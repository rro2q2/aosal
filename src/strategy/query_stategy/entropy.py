# Author: Roland Oruche
# Affiliation: University of Missouri-Columbia
# Year: 2024

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Any, List
from omegaconf import DictConfig
from torch.distributions.categorical import Categorical

from strategy.sampling_base import QSamplingBase

class EntropySampling(QSamplingBase):
    def __init__(
        self,
        model: Any,
        budget_percent: float,
        acquisition_percent: float,
        name: str
    ):
        super(EntropySampling, self).__init__(model, budget_percent, acquisition_percent, name)

    def query(self, n: int, main_dataset: Optional[Dataset], train_idxs: List, unlabeled_idxs: List, threshold: Optional[float] = None):
        probs = self.get_predicted_prob(unlabeled_idxs, main_dataset)
        # Calculate uncertainty measure
        uncertainties = {idx: Categorical(probs=probs[idx]).entropy() for idx in probs}
        sorted_uncertainties = sorted(uncertainties.items(), key=lambda x: x[1], reverse=True)
        return [idx[0] for idx in sorted_uncertainties[:n]]
    