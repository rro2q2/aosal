# Author: Roland Oruche
# Affiliation: University of Missouri-Columbia
# Year: 2024

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from typing import Optional, Any, List
from sklearn.cluster import KMeans
from tqdm import tqdm
from utils.train_utils import collate_fn
from strategy.sampling_base import QSamplingBase

class BertKMSampling(QSamplingBase):
    def __init__(
        self,
        model: Any,
        budget_percent: float,
        acquisition_percent: float,
        name: str
    ):
        super(BertKMSampling, self).__init__(model, budget_percent, acquisition_percent, name)

    def query(self, n: int, main_dataset: Optional[Dataset], train_idxs: List, unlabeled_idxs: List, threshold: Optional[float] = None):
        # Get embeddings from unlabaled pool
        unlbl_embeddings = self.get_unlabeled_embeddings(unlabeled_idxs, main_dataset)
        idx2uidx = {idx: uidx for idx, uidx in enumerate(unlbl_embeddings)}
        embeddings = [unlbl_embeddings[idx].squeeze().cpu().numpy() for idx in unlbl_embeddings]
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
        return [idx2uidx[q] for q in q_idxs]
