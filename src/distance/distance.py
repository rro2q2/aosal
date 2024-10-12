import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import sys
import numpy as np
from scipy.spatial.distance import cdist

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def extract_features(model, dataloader, include_labels=True):
    # self.model.bert.eval()  
    # features = []
    # labels = []
    # with torch.no_grad():
    #     for _, batch in tqdm(enumerate(dataloader), file=sys.stdout):
    #         batch = {key: value.to(device) for key, value in batch.items() if key != "indices"}
    #         outputs = model.forward(**batch)
    #         # Collect features using the [CLS] token representation
    #         batch_features = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
    #         features.append(batch_features)
    #         # Process labels only if they are included and needed
    #         if include_labels and 'labels' in batch:
    #             labels += batch['labels'].cpu().tolist()
    # if include_labels:
    #     assert len(torch.vstack(features)) == len(labels), f"Mismatch in feature length ({len(features)}) and labels length ({len(labels)}) extraction."
    #     return torch.vstack(features), np.array(labels)
    # else:
    #     return torch.vstack(features)

    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc="Features"):
            indices = batch['indices']
            batch = {key: value.to(device) for key, value in batch.items() if key != 'indices'}
            batch_output = model.forward(**batch)
            batch_embs = batch_output.hidden_states[-1][:, 0, :] # CLS token
            embeddings.append(batch_embs)
            labels.extend(batch['labels'].cpu().numpy())
    return embeddings, labels

def compute_class_stats(features, labels):
    unique_labels = np.unique(labels)
    class_means = np.zeros([len(unique_labels), features.shape[1]]) #{}
    sample_counts = []
    # Compute the global covariance matrix that can be used as fallback
    global_covariance = np.cov(features, rowvar=False)
    pooled_covariance = np.zeros_like(global_covariance)
    for l in unique_labels:
        class_features = features[l == labels]
        if class_features.shape[0] > 1:
            class_means[l] = np.mean(class_features, axis=0)
            cov = np.cov(class_features, rowvar=False)
            pooled_covariance += cov * (class_features.shape[0] - 1)
            sample_counts.append(class_features.shape[0] - 1)
        else:
            class_means[l] = np.mean(class_features, axis=0) if class_features.shape[0] > 0 else np.zeros(features.shape[1])
    pooled_covariance = (pooled_covariance / sum(sample_counts)) if sum(sample_counts) > 0 else global_covariance
    return torch.Tensor(class_means), torch.Tensor(pooled_covariance)

def mahalanobis(vector, means_matrix, cov_matrix):
    # Compute the inverse of the covariance matrix
    cov_matrix_inv = torch.linalg.inv(cov_matrix)
    # Center the vector and each row of the means matrix
    vector_centered = vector - means_matrix
    # Compute the Mahalanobis distance for each row in the means matrix
    left_term = torch.matmul(vector_centered, cov_matrix_inv)
    distances = torch.sqrt(torch.sum(left_term * vector_centered, dim=1))
    return distances

def mahalanobis_distance(X, means, cov):
    """
    # X = torch.Tensor(X).to(device)
    # means = [means[label].tolist() for _, label in enumerate(means)]
    # means = torch.Tensor(means).to(device)
    # cov = cov.to(device)
    inv_covmat = torch.linalg.inv(cov)
    print("Inverse of covariance matrix computed.")
    # min_distances = []
    # for i in range(len(X)):
    #     dist = []
    #     for j in range(len(means)):
    #         dist.append(mahalanobis(X[i], means[j], cov).cpu())
    #     min_distances.append(np.min(dist))

    min_distances = []
    for i, x in enumerate(X):
        distances = []
        for mean in means.values():
            dist = cdist([x], [mean], 'mahalanobis', VI=inv_covmat)[0]
            # dist = mahalanobis(x.cpu(), mean, cov.cpu())
            # print(dist)
            distances.append(dist)
        min_distance = np.min(distances)
        min_distances.append(min_distance)
        if i % 500 == 0:  # Log progress every 500 samples
            print(f"Computed distances for {i} samples.")
         
    print("Mahalanobis distances computation completed.")
    return np.array(min_distances)
    """
    distances = []
    for x in tqdm(range(len(X)), desc="MAH"):
        distances.append(mahalanobis(X[x].to(device), means.to(device), cov.to(device)))
    d = torch.min(torch.vstack(distances), dim=1)
    return d.values


def wasserstein(vector, matrix, p=1):
    # Ensure the vector and each row of the matrix are sorted
    vector_sorted, _ = torch.sort(vector)
    matrix_sorted, _ = torch.sort(matrix, dim=1)
    # Compute the Wasserstein distance using the sorted vector and sorted matrix rows
    wasserstein_distances = torch.norm(vector_sorted.unsqueeze(0) - matrix_sorted, p=p, dim=1)
    return wasserstein_distances

def calculate_wasserstein_distance_to_base(features, base_distribution):
    distances = []
    for x in tqdm(range(len(features)), desc='WAS'):
        distances.append(wasserstein(features[x].to(device), base_distribution.to(device), p=2))
    d = torch.min(torch.vstack(distances), dim=1)
    return d.values

def compute_base_distribution(ind_features, ind_labels):
    unique_labels = np.unique(ind_labels)
    class_means = torch.zeros([len(unique_labels), ind_features.shape[1]])
    for l in unique_labels:
        class_features = ind_features[l == ind_labels]
        class_means[l] = torch.Tensor(np.mean(class_features, axis=0))
    return class_means

def norm_dist(distances):
    # z-score + sigmoid normalization
    # z_score_dis = (distances - np.mean(distances)) / np.std(distances)
    # normalized_distances = sigmoid(z_score_dis)

    # max ratio normalization
    if isinstance(distances, torch.Tensor):
        distances = distances.cpu().numpy()

    max_distance = np.max(distances)
    normalized_distances = distances / max_distance if max_distance > 0 else distances
    return normalized_distances

def calculate_fpr_threshold(distances, percentile=95):
    return np.percentile(distances, percentile)

def sigmoid(Z):
    return 1/(1+(np.exp((-Z))))

def run_distance(distance, ind_features, ind_labels, test_features):
    if distance not in {'mahalanobis', 'wasserstein'}:
        raise ValueError(f"The distance option {self.distance} is invalid.\n")
    
    ind_features = np.vstack([f.cpu().tolist() for f in ind_features])
    ind_labels = np.array([l.tolist() for l in ind_labels])
    if isinstance(test_features, dict):
        test_features = torch.Tensor(np.vstack([test_features[f].cpu().tolist() for f in test_features]))
    else:
        test_features = torch.Tensor(np.vstack([f.cpu().tolist() for f in test_features]))
    if distance == 'mahalanobis':
        class_means, pooled_cov = compute_class_stats(ind_features, ind_labels)
        test_distances = mahalanobis_distance(test_features, class_means, pooled_cov)
        return test_distances
    else:
        base_distribution = compute_base_distribution(ind_features, ind_labels)
        test_distances = calculate_wasserstein_distance_to_base(test_features, base_distribution)
        return test_distances

if __name__ == '__main__':
    _ = Distances()
