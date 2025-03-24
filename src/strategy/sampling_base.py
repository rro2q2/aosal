# Author: Roland Oruche
# Affiliation: University of Missouri-Columbia
# Year: 2024

from typing import Optional, Any, List, Union, Dict
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader
from utils.train_utils import collate_fn, collate_fn_binary_labels
from tqdm import tqdm
import numpy as np

class QSamplingBase:
    """ Performs querying sampling by accessing model and dataset. """
    def __init__(
        self,
        model: Any,
        budget_percent: float,
        acquisition_percent: float,
        name: str
    ):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.budget_percent = budget_percent
        self.acquisition_percent = acquisition_percent
        self.name = name

    def query(self, n: int, data_set: Optional[Dataset], train_idxs: List, unlabeled_idxs: List, threshold: Optional[float] = None):
        return

    def update_train_set(self, dataset, ind_query_idxs, main_dataset) -> DataLoader:
        # Update train data indices
        dataset.train_data_idxs = np.concatenate((dataset.train_data_idxs, ind_query_idxs))
        # Update labeled indices 
        dataset.labeled_samples[ind_query_idxs] = True
        # Update unlabeled indices
        # dataset.unlabeled_idxs = [idx for idx in dataset.unlabeled_idxs if not dataset.labeled_samples[idx]]
        dataset.unlabeled_idxs = [idx for idx, _ in enumerate(main_dataset) if not dataset.labeled_samples[idx]]
        # return new train dataloader
        return dataset.train_dataloader(main_dataset)

    def update_val_ood_set(self, dataset, ood_query_idxs, main_dataset, val_data_idxs) -> DataLoader:
        # Update ood val data indices
        dataset.ood_val_data_idxs = np.concatenate((dataset.val_ood_data_idxs, ood_query_idxs))
        # Update labeled indices 
        dataset.labeled_samples[ood_query_idxs] = True
        # Update unlabeled indices
        dataset.unlabeled_idxs = [idx for idx, _ in enumerate(main_dataset) if not dataset.labeled_samples[idx]]
        # return new train dataloader
        return dataset.ood_val_dataloader(main_dataset, val_data_idxs)

    def get_labeled_embeddings(self, train_idxs, main_dataset):
        """ Runs inference on labeled set to get embeddings. """
        self.model.bert.eval()
        data_train = Subset(main_dataset, train_idxs)
        embeddings = []
        labels = []
        loader = DataLoader(
            data_train,
            batch_size=32,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
            shuffle=False,
        )
        with torch.no_grad():
            for batch in tqdm(loader, total=len(loader), desc="Lbl Embds"):
                indices = batch['indices']
                batch = {key: value.to(self.device) for key, value in batch.items() if key != 'indices'}
                batch_output = self.model.forward(**batch)
                batch_embs = batch_output.hidden_states[-1][:, 0, :] # CLS token
                embeddings.append(batch_embs)
                labels.extend(batch['labels'].cpu().numpy())
        return embeddings, labels

    def get_unlabeled_embeddings(self, unlabeled_idxs, main_dataset):
        """ Runs inference on unlabeled set to get embeddings. """
        self.model.bert.eval()
        data_unlabeled = Subset(main_dataset, unlabeled_idxs)
        embeddings = dict()
        loader = DataLoader(
            data_unlabeled,
            batch_size=32,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
            shuffle=False,
        )
        with torch.no_grad():
            for batch in tqdm(loader, total=len(loader), desc="Unlbl Embds"):
                indices = batch['indices']
                batch = {key: value.to(self.device) for key, value in batch.items() if key != 'indices'}
                batch_output = self.model.forward(**batch)
                batch_embs = batch_output.hidden_states[-1][:, 0, :]
                batch_embs = {idx: output for idx, output in zip(indices, batch_embs)}
                embeddings.update(batch_embs)
        return embeddings

    def get_predicted_prob(self, unlabeled_idxs, main_dataset):
        print("Calculating Predicted Probability...")
        self.model.bert.eval()
        data_unlabeled = Subset(main_dataset, unlabeled_idxs)
        probs = dict()
        loader = DataLoader(
            data_unlabeled,
            batch_size=32,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
            shuffle=False,
        )
        with torch.no_grad():
            for batch in tqdm(loader, total=len(loader), desc="Pred Prob"):
                indices = batch['indices']
                batch = {key: value.to(self.device) for key, value in batch.items() if key != 'indices'}
                batch_output = self.model.forward(**batch)
                batch_logits = batch_output.logits
                batch_probs = F.softmax(batch_logits, dim=1)
                batch_probs = {ind: p.cpu() for ind, p in zip(indices, batch_probs)}
                probs.update(batch_probs)
        return probs

if __name__ == '__main__':
    _ = QSamplingBase()
