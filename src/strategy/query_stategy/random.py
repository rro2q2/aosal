# Author: Roland Oruche
# Affiliation: University of Missouri-Columbia
# Year: 2024

import random
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Any, List
from omegaconf import DictConfig

from strategy.sampling_base import QSamplingBase

class RandomSampling(QSamplingBase):
    def __init__(
        self,
        model: Any,
        budget_percent: float,
        acquisition_percent: float,
        name: str
    ):
        super(RandomSampling, self).__init__(model, budget_percent, acquisition_percent, name)

    def query(self, n: int, main_dataset: Optional[Dataset], train_idxs: List, unlabeled_idxs: List, threshold: Optional[float] = None):
        # perform random sampling
        selected_idxs = torch.randperm(len(unlabeled_idxs))[:n]
        return [unlabeled_idxs[s] for s in selected_idxs]
