# Author: Roland Oruche
# Affiliation: University of Missouri-Columbia
# Year: 2024

from typing import Optional, Any, Dict
import torch
from torch.utils.data import DataLoader
from torch import optim
import sys
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from metrics import *
from distance.distance import *

class Trainer:
    """ Trains and evaluates model on IND data and OOD data.
    Functions:
        train(): Trains model over specified number of epochs.
        train_epoch(): Performs train and optimization over given epoch.
        evaluate(): Evaluates model on IND data in val set.
        evaluate_ood(): Evaluates model on OOD data in val set and updates FPR threshold.
        test(): Evaluates model on IND data in test set.
    """
    def __init__(
        self,
        model: Any,
        learning_rate: float,
        epochs: int,
        weight_decay: float = 0.0,
        distance: str = 'mahalanobis',
        percentile: int = 95,
        const_threshold: bool = False
    ) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Instantiate hyperparameters
        self.model = model.bert.to(self.device)
        self.lr = learning_rate
        self.epochs = epochs
        self.weight_decay = weight_decay
        # Instantiate optimizer
        self.optimizer = optim.AdamW(params=model.bert.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.current_epoch = 1
        # Get distance
        self.distance = distance
        # Get FPR threshold percentile
        self.percentile = percentile
        self.const_threshold = const_threshold

    def train(self, train_dl: DataLoader, val_dl: DataLoader) -> None:
        for epoch in range(self.current_epoch, self.epochs+1):
            avg_loss = self.train_epoch(train_dl)
            print(f"Average train loss at epoch {epoch}: {avg_loss:.3f}")
            # Evaluate at the end of the epoch
            val_results = self.evaluate(val_dl)
            print(f"Average val loss: {val_results['avg_loss']:.3f} | Average val accuracy: {val_results['avg_acc']:.3f}")

    def train_epoch(self, train_dl: DataLoader) -> float:
        self.model.bert.train()
        total_loss = 0
        for _, batch in tqdm(enumerate(train_dl), file=sys.stdout):
            batch = {key: value.to(self.device) for key, value in batch.items() if key != "indices"}
            labels = batch["labels"]
            # Zero out all gradients
            self.optimizer.zero_grad()
            outputs = self.model.forward(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        average_loss = total_loss / len(train_dl)
        return average_loss

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.bert.eval()
        predictions = []
        ground_truth = []
        results = dict()
        total_loss = 0
        for _, batch in tqdm(enumerate(dataloader), file=sys.stdout):
            batch = {key: value.to(self.device) for key, value in batch.items() if key != "indices"}
            labels = batch["labels"]
            with torch.no_grad():
                outputs = self.model.forward(**batch)
                loss, logits = outputs.loss, outputs.logits
                total_loss += loss.item()
            predictions += torch.argmax(logits, dim=1).long().tolist()
            ground_truth += labels.tolist()
        average_loss = total_loss / len(dataloader)
        results["avg_loss"] = average_loss
        average_acc = accuracy_score(predictions, ground_truth)
        results["avg_acc"] = average_acc
        return results

    def evaluate_ood(self, train_dl: DataLoader, val_ood_dl: DataLoader) -> Tuple[Dict[str, float], float]:
        self.model.bert.eval()
        predictions = []
        ground_truth = []
        ood_results = dict()
        ind_features, ind_labels = extract_features(self.model, train_dl)
        val_features, val_labels = extract_features(self.model, val_ood_dl)
        ground_truth = val_labels
        # Run distance
        val_distances = run_distance(self.distance, ind_features, ind_labels, val_features)
        # Normalize disances
        normalized_distances = norm_dist(val_distances)
        # Calculate threshold
        if self.const_threshold is True:
            threshold = 0.5
        else:
            threshold = calculate_fpr_threshold(normalized_distances, percentile=self.percentile)
        # Get predictions
        predictions = [0 if dist <= threshold else 1 for dist in normalized_distances]
        assert len(predictions) == len(ground_truth), f"Mismatch in predictions length {len(predictions)} and ground truth labels length {len(ground_truth)}."
        average_acc = accuracy_score(predictions, ground_truth)
        print(f"IND/OOD Validation Accuracy: {average_acc * 100:.3f}%")
        ood_results["avg_ood_acc"] = average_acc
        return ood_results, threshold

    def test(self, test_dl: DataLoader) -> Dict[str, float]:
        test_results = self.evaluate(test_dl)
        return test_results
                            
if __name__ == '__main__':
    _ = Trainer()
