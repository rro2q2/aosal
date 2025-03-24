# Authors: Roland Oruche, Sai Keerthana Goruganthu
# Affiliation: University of Missouri-Columbia
# Year: 2024

from typing import Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def accuracy(true: list, pred: list) -> Union[int, float]:
    return accuracy_score(true, pred)

def precision(true: list, pred: list, avg: str) -> float:
    return precision_score(true, pred, average=avg)

def recall(true: list, pred: list, avg: str) -> float:
    return recall_score(true, pred, average=avg)

def f1(true: list, pred: list, avg: str) -> float:
    return f1_score(true, pred, average=avg)
