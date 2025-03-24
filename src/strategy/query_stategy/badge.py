# Author: Roland Oruche (Orginal code by: JordanAsh)
# Original code: https://github.com/JordanAsh/badge/blob/master/query_strategies/badge_sampling.py 
# Affiliation: University of Missouri-Columbia
# Year: 2024

import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader
from typing import Optional, Any, List
from copy import copy as copy
from copy import deepcopy as deepcopy
import torch
from tqdm import tqdm
import pdb
from torch.nn import functional as F
from scipy import stats
import numpy as np
from scipy.spatial.distance import cdist
from utils.train_utils import collate_fn

from strategy.sampling_base import QSamplingBase

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# kmeans ++ initialization
def init_centers(embeddings, K):
    print("Calculating KMeans++")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # embeddings = torch.Tensor(embeddings)
    embeddings = embeddings.to(device)
    ind = torch.argmax(torch.norm(embeddings, p=2, dim=1)).item()
    mu = [embeddings[ind]]
    inds_all = [ind]
    cent_inds = [0.] * len(embeddings)
    cent = 0
    print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = torch.cdist(mu[-1].view(1,-1), embeddings, p=2)[0].cpu().numpy()
        else:
            new_D = torch.cdist(mu[-1].view(1,-1), embeddings, p=2)[0].cpu().numpy()
            for i in range(len(embeddings)):
                if D2[i] > new_D[i]:
                    cent_inds[i] = cent
                    D2[i] = new_D[i]
        # set breakpoint for python debugger
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        D_dist = (D2 ** 2)/ sum(D2 ** 2)
        # get idx and corresponding probabilities
        custom_dist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), D_dist))
        ind = custom_dist.rvs(size=1)[0]
        while ind in inds_all: ind = custom_dist.rvs(size=1)[0]
        mu.append(embeddings[ind])
        inds_all.append(ind)
        cent += 1
    return inds_all


class BADGESampling(QSamplingBase):
    def __init__(
        self, 
        model: Any,
        budget_percent: float,
        acquisition_percent: float,
        name: str
    ):
        super(BADGESampling, self).__init__(model, budget_percent, acquisition_percent, name)

    def query(self, n: int, main_dataset: Optional[Dataset], train_idxs: List, unlabeled_idxs: List, threshold: Optional[float] = None):
        grad_embeddings, idx2uidx = self.get_grad_embedding(unlabeled_idxs, main_dataset)
        # Perform K-Means++ clustering
        selected_idxs = init_centers(grad_embeddings, n)    
        selected_uidxs = [idx2uidx[idx] for idx in selected_idxs]
        # Returns the indices from the unlabeled pool
        return selected_uidxs

    
    def get_grad_embedding(self, unlabeled_data_idxs, main_dataset):
        embedding_dim = self.model.bert.config.hidden_size
        self.model.bert.eval()
        # Get the number of labels
        num_labels = self.model.bert.num_labels
        data_unlabeled = Subset(main_dataset, unlabeled_data_idxs)
        embeddings = np.zeros([len(data_unlabeled), embedding_dim * num_labels])
        idx2uidx = dict()
        loader = DataLoader(
            data_unlabeled,
            batch_size=32,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
            shuffle=False,
        )
        # Without updating gradients during inference,
        # pass unlabeled pool into the model and calculate gradients
        with torch.no_grad():
            unlabeled_idx = 0
            for batch in tqdm(loader, total=len(loader), desc="Unlbl Embds"):
                indices = batch['indices']
                batch = {key: value.to(self.device) for key, value in batch.items() if key != 'indices'}
                batch_output = self.model.forward(**batch)
                embs = batch_output.hidden_states[-1][:, 0, :].cpu().numpy()
                probs = F.softmax(batch_output.logits, dim=1).cpu().numpy()
                batch_max_idxs = np.argmax(probs, 1)
                for j in range(len(batch['labels'])):
                    idx2uidx[unlabeled_idx] = indices[j]
                    for c in range(num_labels):
                        if c == batch_max_idxs[j]:
                            embeddings[unlabeled_idx][embedding_dim * c : embedding_dim * (c+1)] = deepcopy(embs[j]) * (1 - probs[j][c])
                        else:
                            embeddings[unlabeled_idx][embedding_dim * c : embedding_dim * (c+1)] = deepcopy(embs[j]) * (-1 * probs[j][c])
                    # Update idx of unlabeled data
                    unlabeled_idx += 1
        return torch.Tensor(embeddings), idx2uidx
    