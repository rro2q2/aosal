# Author: Roland Oruche
# Affiliation: University of Missouri-Columbia
# Year: 2024

from typing import Dict, Union, List
import torch
from torch.utils.data import Subset

def collate_fn(batch: Subset) -> Dict[str, Union[torch.tensor, List[int]]]:
    max_len = max([len(f[0]["input_ids"]) for f in batch])
    input_ids = [f[0]["input_ids"] + [0] * (max_len - len(f[0]["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f[0]["input_ids"]) + [0.0] * (max_len - len(f[0]["input_ids"])) for f in batch]
    labels = [f[1].item() for f in batch]
    indices = [f[2] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    outputs = {
        "input_ids": input_ids,
        "attention_mask": input_mask,
        "labels": labels,
        "indices": indices
    }
    return outputs

def collate_fn_binary_labels(batch: Subset) -> Dict[str, Union[torch.tensor, List[int]]]:
    max_len = max([len(f[0]["input_ids"]) for f in batch])
    input_ids = [f[0]["input_ids"] + [0] * (max_len - len(f[0]["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f[0]["input_ids"]) + [0.0] * (max_len - len(f[0]["input_ids"])) for f in batch]
    labels = [f[1].item() for f in batch]
    indices = [f[2] for f in batch]
    binary_labels = []
    for l in labels:
        bin_lbl = 0 if l != 999 else 1
        binary_labels.append(bin_lbl)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    binary_labels = torch.tensor(binary_labels, dtype=torch.long)
    outputs = {
        "input_ids": input_ids,
        "attention_mask": input_mask,
        "labels": binary_labels,
        "indices": indices
    }
    return outputs