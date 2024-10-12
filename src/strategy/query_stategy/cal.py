import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Optional, Any, List
from copy import copy as copy
from copy import deepcopy as deepcopy
import torch
from tqdm import tqdm
import pdb
from torch.nn import functional as F
import torch.nn as nn
from scipy import stats
from utils.train_utils import collate_fn

from sklearn.neighbors import NearestNeighbors

from strategy.sampling_base import QSamplingBase

class CALSampling(QSamplingBase):
    def __init__(
        self, 
        model: Any,
        budget_percent: float,
        acquisition_percent: float,
        name: str,
        k: int
    ):
        super(CALSampling, self).__init__(model, budget_percent, acquisition_percent, name)
        self.k = k

    def query(self, n: int, main_dataset: Optional[Dataset], train_idxs: List, unlabeled_idxs: List, threshold: Optional[float] = None):
        # Get embeddings of labeled pool
        print("Get embeddings of labeled pool")
        D_L_embeddings = self.get_embeddings(train_idxs, main_dataset, tag="labeled")
        labeled_logits = D_L_embeddings[0]
        labeled_pooled = D_L_embeddings[1]
        labeled_probs = F.softmax(labeled_logits, dim=1)
        
        # Get embeddings of unlabeled pool
        print("Get embeddings of unlabeled pool")
        D_U_embeddings = self.get_embeddings(unlabeled_idxs, main_dataset, tag="unlabeled")
        unlabeled_logits = D_U_embeddings[0]
        unlabeled_pooled = D_U_embeddings[1]
       
        unlabeled_probs = {idx: F.log_softmax(unlabeled_logits[idx], dim=-1) for idx in unlabeled_logits}
        unlabeled_emb_idx = {i: x for i, x in enumerate(unlabeled_idxs)}
        unlabeled_emb = torch.stack([unlabeled_pooled[x].squeeze(0) for _, x in enumerate(unlabeled_pooled)])
        print("Calculating CAL scores...")
        neigh = NearestNeighbors(n_neighbors=self.k)
        neigh.fit(labeled_pooled.cpu().numpy())
        criterion = torch.nn.KLDivLoss(reduction='none')
        kl_scores = dict()
        unlabeled_knn = neigh.kneighbors(unlabeled_emb.cpu().numpy(), return_distance=False)
        for i in range(len(unlabeled_idxs)):
            kl_score = sum([criterion(unlabeled_probs[unlabeled_emb_idx[i]],labeled_probs[j]) for j in unlabeled_knn[i]])
            kl_scores[unlabeled_emb_idx[i]] = kl_score.mean()
        sorted_kl_scores = sorted(kl_scores.items(), key=lambda x: x[1], reverse=True)
        return [idx[0] for idx in sorted_kl_scores[:n]]
    
    def get_embeddings(self, data_idxs, main_dataset, tag="labeled"):
        """
        NOTE: CAL uses [CLS] token embedding.
        """
        self.model.bert.eval()
        data= Subset(main_dataset, data_idxs)
        embeddings = dict()
        loader = DataLoader(
            data,
            batch_size=32,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
            shuffle=False,
        )
        # run inference on the dataset
        if tag == "labeled":
            logits = None
            pooled = None
            with torch.no_grad():
                for batch in tqdm(loader, total=len(loader), desc="Pred Prob"):
                    indices = batch['indices']
                    batch = {key: value.to(self.device) for key, value in batch.items() if key != 'indices'}
                    batch_output = self.model.forward(**batch)
                    batch_pooled = batch_output.hidden_states[-1][:, 0, :]
                    batch_logits = batch_output.logits
                    logits = torch.cat((logits, batch_logits), dim=0) if logits is not None else batch_logits
                    pooled = torch.cat((pooled, batch_pooled), dim=0) if pooled is not None else batch_pooled
            embeddings = [logits, pooled]
        else:
            embeddings["logits"] = dict()
            embeddings["pooled"] = dict()
            with torch.no_grad():
                for batch in tqdm(loader, total=len(loader), desc="Pred Prob"):
                    indices = batch['indices']
                    batch = {key: value.to(self.device) for key, value in batch.items() if key != 'indices'}
                    batch_output = self.model.forward(**batch)
                    batch_pooled = batch_output.hidden_states[-1][:, 0, :]
                    batch_logits = batch_output.logits
                    embeddings["logits"].update({idx: output for idx, output in zip(indices, batch_logits)})
                    embeddings["pooled"].update({idx: logit for idx, logit in zip(indices, batch_pooled)})
            # embeddings.shape = List[Dict[int, torch.Tensor]]
            embeddings = [embeddings["logits"], embeddings["pooled"]]
        return embeddings
