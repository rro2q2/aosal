import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Any, List
from omegaconf import DictConfig
from torch.distributions.categorical import Categorical
from sklearn.cluster import KMeans

from strategy.sampling_base import QSamplingBase
from distance.distance import * 

class AOSALSampling(QSamplingBase):
    def __init__(
        self,
        model: Any,
        budget_percent: float,
        acquisition_percent: float,
        name: str,
        distance: str,
        inf_measure: str
    ):
        super(AOSALSampling, self).__init__(model, budget_percent, acquisition_percent, name)
        self.distance = distance
        self.inf_measure = inf_measure

    def query(self, n: int, main_dataset: Optional[Dataset], train_idxs: List, unlabeled_idxs: List, threshold: Optional[float] = None):
        # TODO: If the classified IND samples is less than acquisition size,
        # select the top classified OOD samples to fill.

        print("### Running AOSAL Loop ###")
        # Extracts features from IND train set
        train_embeddings, train_labels = self.get_labeled_embeddings(train_idxs, main_dataset)
        # Extract features from the (IND/OOD) unlabeled pool
        unlabeled_embeddings = self.get_unlabeled_embeddings(unlabeled_idxs, main_dataset)
        idx2uidx = {idx: uidx for idx, uidx in enumerate(unlabeled_embeddings)}
        # Run distance-based score function
        unlabeled_distances = run_distance(self.distance, train_embeddings, train_labels, unlabeled_embeddings).cpu().numpy()
        print(unlabeled_distances)
        # Normalize disances
        normalized_distances = norm_dist(unlabeled_distances)
        print(normalized_distances)
        # Sort normalized distances - returns indices array
        sorted_idxs = np.argsort(normalized_distances)
        # # Calculate threshold based on normalized distances
        # threshold = calculate_fpr_threshold(normalized_distances)
        # Get pseudo IND samples that are <= threshold
        pseudo_ind_idxs = [idx for idx in sorted_idxs if normalized_distances[idx] <= threshold]
        # Calculate informative measure (e.g., uncertainty, diversity)
        selected_idxs = self.run_informative_measure(n, pseudo_ind_idxs, main_dataset)
        return [idx2uidx[idx] for idx in selected_idxs]

    def run_informative_measure(self, n, data_idxs, main_dataset):
        if self.inf_measure not in {'uncertainty', 'diversity'}:
            raise ValueError(f"The distance option {self.distance} is invalid.\n")

        if self.inf_measure == 'uncertainty':
            # Calculating Entropy
            probs = self.get_predicted_prob(data_idxs, main_dataset)
            uncertainties = {idx: Categorical(probs=probs[idx]).entropy() for idx in probs}
            sorted_uncertainties = sorted(uncertainties.items(), key=lambda x: x[1], reverse=True)
            return [idx[0] for idx in sorted_uncertainties[:n]]
        else:
            # Calulating BERT-KM
            unlabeled_embeddings = self.get_unlabeled_embeddings(data_idxs, main_dataset)
            # Convert unlabeled id to psuedo IND id
            uidx2pidx = {idx: uidx for idx, uidx in enumerate(unlabeled_embeddings)}
            embeddings = [unlabeled_embeddings[idx].squeeze().cpu().numpy() for idx in unlabeled_embeddings]
            print("Performing KMeans clustering...")
            cluster_learner = KMeans(n_clusters=n)
            cluster_learner.fit(embeddings)
            # Get cluster idxs
            cluster_idxs = cluster_learner.predict(embeddings)
            # Compute center embedding for each cluster
            centers = cluster_learner.cluster_centers_[cluster_idxs]
            # Calculate the distance between each embedding and centers
            dis = (embeddings - centers)**2
            dis = dis.sum(axis=1)
            q_idxs = np.array([np.arange(dis.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])
            return [uidx2pidx[q] for q in q_idxs]
