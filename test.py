import numpy as np
from sklearn.metrics import roc_curve
from Model import val_dataset,model, val_labels
from Model import extract_features,compute_class_stats
import torch
import os
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.covariance import EmpiricalCovariance
from scipy.spatial.distance import cdist
from sklearn.utils import resample



val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
val_features, val_labels = extract_features(model, val_loader)
print('Features Extracted')

# distances = np.load('./mahalanobis_distances.npy')
class_means, pooled_cov = compute_class_stats(val_features, val_labels)

def calculate_fpr_threshold(features, true_labels, class_means, pooled_cov, desired_fpr=0.05):
    # distances = mahalanobis_distance(features, class_means, pooled_cov)
    # Assuming OOD samples are labeled as 1 in the binary true_labels
    distances = np.load('./mahalanobis_distances.npy')
    fpr, tpr, thresholds = roc_curve(true_labels, distances, pos_label=1)
    # Find the threshold closest to the desired FPR
    threshold = thresholds[np.argmin(np.abs(fpr - desired_fpr))]
    return threshold

def calculate_fpr_threshold(distances, labels, oos_label_index, percentile=80):
    ind_distances = distances[labels!= oos_label_index]
    threshold = np.percentile(ind_distances, percentile)
    return threshold


# Example usage during initial setup
initial_threshold = calculate_fpr_threshold(val_features, val_labels, class_means, pooled_cov, desired_fpr=0.05)

# After human annotation and model update
# updated_features, updated_labels = extract_features(new_model, updated_dataloader)
# updated_class_means, updated_pooled_cov = compute_class_stats(updated_features, updated_labels)
# updated_threshold = calculate_fpr_threshold(updated_features, updated_labels, updated_class_means, updated_pooled_cov, desired_fpr=0.05)

# from sklearn.metrics import confusion_matrix
# # ood_predictions: array of OOD predictions (1 for OOD, 0 for IND)
# # ood_labels: array of true OOD labels (1 for OOD, 0 for IND)
# tn, fp, fn, tp = confusion_matrix(ood_labels, ood_predictions).ravel()

# # OOD detection rate could be calculated as true positive rate (recall)
# ood_detection_rate = tp / (tp + fn)

# from sklearn.metrics import confusion_matrix

# # ood_predictions: array of OOD predictions (1 for OOD, 0 for IND)
# # ood_labels: array of true OOD labels (1 for OOD, 0 for IND)
# tn, fp, fn, tp = confusion_matrix(ood_labels, ood_predictions).ravel()

# # OOD detection rate could be calculated as true positive rate (recall)
# ood_detection_rate = tp / (tp + fn)
