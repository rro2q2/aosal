import torch
import os
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.covariance import EmpiricalCovariance
from scipy.spatial.distance import cdist
from sklearn.utils import resample
# from test import calculate_fpr_threshold
from datasets import load_dataset
from torch.utils.data.dataloader import default_collate

def consolidate_labels(labels):
    # A dictionary to map variations to a standardized label
    label_map = {}

    # A list of common prefixes to be removed to standardize labels
    prefixes = [
        'please', 'can you', 'help me', 'how do i', 'what\'s', 'what is', 
        'do you know how', 'i need to', 'tell me', 'i want to'
    ]

    for label in labels:
        try:
            # Create a pattern to match any of the prefixes defined above
            prefix_pattern = r'^(?:' + '|'.join(prefixes) + r')[, ]*'
            
            # Remove the matched prefixes and any leading/trailing punctuation
            standardized_label = re.sub(prefix_pattern, '', label, flags=re.I)
            standardized_label = standardized_label.strip().lower()
            
            # Remove any residual punctuation marks at the end or beginning
            standardized_label = re.sub(r'^[!?.\s]+|[!?.\s]+$', '', standardized_label)
            
            # Normalize white spaces to a single space between words
            standardized_label = re.sub(r'\s+', ' ', standardized_label)

            # Map original labels to their standardized form
            label_map[label] = standardized_label
        except Exception as e:
            print(f"Error processing label '{label}': {e}")
            # Optionally, you could map failed labels to a special category
            label_map[label] = "error"

    return label_map

def oversample_minor_classes(texts, labels):
    # Unique classes and the size of the largest class
    unique_classes = np.unique(labels)
    max_size = max(np.bincount(labels))
    
    texts_resampled = []
    labels_resampled = []
    
    # Resample each class to the maximum size
    for cls in unique_classes:
        # Filter texts and labels for the class
        class_texts = [text for text, label in zip(texts, labels) if label == cls]
        class_labels = [label for label in labels if label == cls]
        
        if len(class_texts) < max_size:
            # Resample if the current class size is less than the max size
            class_texts, class_labels = resample(class_texts, class_labels, replace=True, n_samples=max_size)
        
        texts_resampled.extend(class_texts)
        labels_resampled.extend(class_labels)
    
    return texts_resampled, labels_resampled


class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'token_type_ids': self.encodings.get('token_type_ids', None),  # Optional, depending on the model
            'labels': torch.tensor(self.labels[idx]).long()
        }
        return item

    def __len__(self):
        return len(self.labels)




# dataset_path = "./Data/clinc-data.txt"
# texts = []
# labels = []

# with open(dataset_path, 'r', encoding='utf-8') as file:
#     for i, line in enumerate(file, 1):
#         parts = line.strip().split(',', 1)  
#         if len(parts) == 2:
#             text, label = parts
#             texts.append(text.strip())
#             labels.append(label.strip())
#         else:
#             print(f"Error on line {i}: {line}")


# Encode labels
# consolidated_label_map = consolidate_labels(labels)
# new_labels = [consolidated_label_map[label] for label in labels]
# label_encoder = LabelEncoder()
# encoded_labels = label_encoder.fit_transform(new_labels)
# tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# # Filter out 'oos' for training
# ind_indices = [i for i, label in enumerate(encoded_labels) if label != 'oos']
# train_texts = [texts[i] for i in ind_indices]
# train_labels = [encoded_labels[i] for i in ind_indices]

# # Oversample minor classes
# train_texts, train_labels = oversample_minor_classes(train_texts, train_labels)

# train_texts, val_texts, train_encoded_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.1, random_state=42)
# train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
# val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
# train_dataset = Clinic150Dataset(train_encodings, train_encoded_labels)
# val_dataset = Clinic150Dataset(val_encodings, val_labels)



#ROSTD dataset
Dataset = load_dataset("cmaldona/Generalization-MultiClass-CLINC150-ROSTD")
dataset = Dataset['train']
texts = [entry['data'] for entry in dataset]
labels = [entry['labels'] for entry in dataset]
test_dataset = Dataset['validation']
val_texts = [entry['data'] for entry in test_dataset]
val_labels = [entry['labels'] for entry in test_dataset]


all_labels = labels + val_labels  # Combine training and validation labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(all_labels)

ood_labels = ['0', '1', 'ood']
ood_index = [label_encoder.transform([label])[0] for label in ood_labels]

train_encoded_labels = encoded_labels[:len(labels)]
encoded_val_labels = encoded_labels[len(labels):]

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
train_dataset = TextDataset(encodings, train_encoded_labels)

val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
val_dataset = TextDataset(val_encodings, encoded_val_labels)




# Load the pre-trained BERT model with the correct number of IND labels
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy='epoch'
)

# Initialize and run trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# trainer.train()

# model_save_directory = "./model_save"
# tokenizer_save_directory = "./tokenizer_save"

# # Make directories if they don't exist
# os.makedirs(model_save_directory, exist_ok=True)
# os.makedirs(tokenizer_save_directory, exist_ok=True)

# # Save the trained model
# model.save_pretrained(model_save_directory)
# # Save the associated tokenizer
# tokenizer.save_pretrained(tokenizer_save_directory)
print('Model trained and saved')


def extract_features(model, dataloader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = torch.tensor([item[0]['input_ids'] for item in batch]).to(device)
            attention_mask = torch.tensor([item[0]['attention_mask'] for item in batch]).to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Use the [CLS] token embedding for feature representation
            features.append(outputs.last_hidden_state[:, 0].cpu().numpy())
            labels.extend([item[1] for item in batch])
    return np.vstack(features), np.array(labels)

# def extract_features_and_logits(model, dataloader):
#     model.eval()
#     features = []
#     logits = []
#     labels = []
#     with torch.no_grad():
#         for batch in dataloader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             # Get the full model outputs (not just the base BERT model)
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#             # The output object will have a logits attribute after passing through the classification head
#             logits.append(outputs.logits.cpu().numpy())
#             # Extract the pooled features which are typically used for classification tasks
#             features.append(outputs.pooler_output.cpu().numpy())  # pooler_output corresponds to the [CLS] token
#             labels.extend(batch['labels'].cpu().numpy())
    
#     return np.vstack(features), np.vstack(logits), np.array(labels)


# print('OOD labels included')

def custom_collate(batch):
    collated_batch = {}
    for key in batch[0].keys():  # Iterate over keys in the first item of the batch
        if all(key in item for item in batch):  # Check if key is present in all items of the batch
            collated_batch[key] = default_collate([item[key] for item in batch])  # Use default collate if key is present
    return collated_batch


full_dataset_encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
Rtrain_dataset = TextDataset(full_dataset_encodings, encoded_labels)
Rfull_loader = DataLoader(Rtrain_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate)

Rfeatures, Rfull_labels = extract_features(model, Rfull_loader)



test_dataset = Dataset['test']
val_texts = [entry['data'] for entry in test_dataset]
val_labels = [entry['labels'] for entry in test_dataset]

full_dataset_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
full_dataset = TextDataset(full_dataset_encodings, encoded_labels)
full_loader = DataLoader(full_dataset, batch_size=16, shuffle=False)

features, full_labels = extract_features(model, full_loader)
print('Features Extracted')

# features, logits, full_labels = extract_features_and_logits(model, full_loader)
# print('Features and logits extracted')

def compute_class_stats(features, labels):
    print("Starting computation of class statistics.")
    unique_labels = np.unique(labels)
    class_means = {}
    sample_counts = []
    
    # Compute the global covariance matrix that can be used as fallback
    global_covariance = np.cov(features, rowvar=False)
    print("Global covariance computed.")
    pooled_covariance = np.zeros_like(global_covariance)

    for label in unique_labels:
        print(f"Processing class {label}.")
        class_features = features[labels == label]
        if class_features.shape[0] > 1:
            class_means[label] = np.mean(class_features, axis=0)
            cov = np.cov(class_features, rowvar=False)
            pooled_covariance += cov * (class_features.shape[0] - 1)
            sample_counts.append(class_features.shape[0] - 1)
            print(f"Processed class {label} with sufficient samples.")
        else:
            # Use global covariance for classes with insufficient samples
            print(f"Using global covariance for class {label} due to insufficient samples.")
            class_means[label] = np.mean(class_features, axis=0) if class_features.shape[0] > 0 else np.zeros(features.shape[1])

    if sum(sample_counts) > 0:
        pooled_covariance /= sum(sample_counts)
        print("Pooled covariance computed.")
    else:
        pooled_covariance = global_covariance
        print("Using global covariance as fallback for pooled covariance.")

    return class_means, pooled_covariance

def mahalanobis_distance(X, means, cov):
    print("Starting Mahalanobis distance computation.")
    inv_covmat = np.linalg.inv(cov)
    print("Inverse of covariance matrix computed.")
    min_distances = []

    for i, x in enumerate(X):
        distances = []
        for mean in means.values():
            dist = cdist([x], [mean], 'mahalanobis', VI=inv_covmat)[0]
            distances.append(dist)
        min_distance = np.min(distances)
        min_distances.append(min_distance)
        if i % 100 == 0:  # Log progress every 100 samples
            print(f"Computed distances for {i} samples.")
    
    print("Mahalanobis distances computation completed.")
    return np.array(min_distances)

# Compute statistics and Mahalanobis distances
# print("Filtering in-distribution (IND) features and labels...")
# ind_features = features[full_labels != label_encoder.transform(['oos'])[0]]
# ind_labels = full_labels[full_labels != label_encoder.transform(['oos'])[0]]

print("Computing class statistics for IND samples...")
class_means, pooled_cov = compute_class_stats(Rfeatures, Rfull_labels)

print("Calculating Mahalanobis distances for all features...")
distances = mahalanobis_distance(Rfeatures, class_means, pooled_cov)

print('Calculated distance', distances)

np.save('ROSTD_mahalanobis_distances.npy', distances)
print('Distances calculated and saved.')

distances = np.load('./ROSTD_mahalanobis_distances.npy')

# def calculate_fpr_threshold(distances, labels, oos_label_index, percentile=80):
#     # Calculate threshold such that only 20% of IND samples are selected
#     threshold = np.percentile(distances[labels != oos_label_index], percentile)
#     return threshold

# # Calculate the threshold
# threshold = calculate_fpr_threshold(distances, Rfull_labels, label_encoder.transform(['oos'])[0])
# # print("Threshold:", threshold)
# # print('----------------------------')
# selected_ind_samples = (distances < threshold) & (full_labels != label_encoder.transform(['oos'])[0])

# # Select features for these samples
# selected_features = features[selected_ind_samples]
# # print('Selected IND samples', selected_features)


#ROSTD dataset threshold
def calculate_fpr_threshold(distances, labels, ood_label_index, percentile=95):
    # Filter distances to include only IND samples
    ind_distances = distances[labels != ood_label_index]
    
    # Calculate the threshold
    threshold = np.percentile(ind_distances, percentile)
    return threshold


threshold = calculate_fpr_threshold(distances, full_labels, ood_label_index)

print("Calculated FPR Threshold:", threshold)

selected_ind_samples = distances < threshold
selected_features = features[selected_ind_samples]


def calculate_entropy(logits):
    probabilities = F.softmax(torch.tensor(logits), dim=1)
    return -torch.sum(probabilities * torch.log(probabilities + 1e-5), dim=1).numpy()

# Assuming logits are available or computing them here
selected_logits = model(torch.tensor(selected_features).to(device)).logits.detach().cpu()
entropies = calculate_entropy(selected_logits)

# Now use this mask to select the tokenized input IDs and attention masks for the IND samples
selected_indices = np.where(selected_ind_samples)[0]
selected_input_ids = np.array(full_dataset_encodings['input_ids'])[selected_indices]
selected_attention_mask = np.array(full_dataset_encodings['attention_mask'])[selected_indices]

# Convert them to tensors and pass them through the model
selected_input_ids_tensor = torch.tensor(selected_input_ids).to(device)
selected_attention_mask_tensor = torch.tensor(selected_attention_mask).to(device)

model.eval()
with torch.no_grad():
    outputs = model(input_ids=selected_input_ids_tensor, attention_mask=selected_attention_mask_tensor)
    selected_logits = outputs.logits.detach().cpu().numpy()

# Calculate entropy for the selected logits
entropies = calculate_entropy(selected_logits)
# print("Entropies:", entropies)

# Select the top 20% of samples based on entropy
top_20_percent_cutoff = int(0.2 * len(entropies))  # 20% of the total number of samples
selected_indices_top_20_percent = np.argsort(entropies)[-top_20_percent_cutoff:]  # Indices of the top 20% based on entropy

# Now get the entropy values for these top 20% samples
selected_entropies = entropies[selected_indices_top_20_percent]

# From this top 20%, select the top 15% to be annotated
top_15_percent_cutoff = int(0.15 * len(selected_entropies))  # 15% of the top 20% of samples
selected_indices_top_15_percent = np.argsort(selected_entropies)[-top_15_percent_cutoff:]  # Indices of the top 15% within the top 20%

# Get the absolute indices in the original dataset for these samples
absolute_indices_for_annotation = selected_indices_top_20_percent[selected_indices_top_15_percent]
selected_annotation_samples = [texts[i] for i in absolute_indices_for_annotation]

# # Display or log these for annotation
print(f"Number of samples selected for annotation: {len(selected_annotation_samples)}")
print("Samples selected for annotation:", selected_annotation_samples)


#Visualizations


# import matplotlib.pyplot as plt

# # Assuming 'entropies' is your array of entropy values
# plt.hist(entropies, bins=50, alpha=0.75)
# plt.title('Histogram of Entropies')
# plt.xlabel('Entropy')
# plt.ylabel('Number of Samples')
# plt.savefig('plot1.png')  # Saves the plot as a PNG file
# plt.close() 
# print("Graph-1")

# # Extract the Mahalanobis distances for the selected samples (top 20% based on entropy)
# selected_distances = distances[selected_indices][selected_indices_top_20_percent]

# # Ensure that 'selected_distances' and 'selected_entropies' have the same length
# assert len(selected_distances) == len(selected_entropies), "Arrays must be the same length to plot"

# # Now you can plot
# plt.scatter(selected_distances, selected_entropies, alpha=0.5)
# plt.xlabel('Mahalanobis Distance')
# plt.ylabel('Entropy')
# plt.title('Scatter Plot of Selected Distances vs. Entropy')
# plt.savefig('plot2.png')  # Saves the plot as a PNG file
# plt.close() 

# print("Graph-2")



