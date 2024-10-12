from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score,average_precision_score
import matplotlib.pyplot as plt
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
from sklearn.metrics import accuracy_score
from Model import val_dataset,train_dataset, val_labels,train_labels,train_texts,val_texts,new_labels, full_labels
from sklearn.preprocessing import label_binarize


# Assume 'encoded_labels' are your label-encoded labels for the entire dataset
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(full_labels)  # new_labels being the original labels

# Number of unique classes
num_classes = 1115
# print(f'Number of classes: {num_classes}')


val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
# Initialize lists to store true labels and prediction probabilities
true_labels = []
prediction_probs = []

model_save_directory = "./model_save"
tokenizer_save_directory = "./tokenizer_save"
model = BertForSequenceClassification.from_pretrained(model_save_directory, num_labels=num_classes)
tokenizer = BertTokenizerFast.from_pretrained(tokenizer_save_directory)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
model.to(device)

for batch in val_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # Store probabilities and true labels for ROC AUC and precision-recall curve
    prediction_probs.extend(probs.cpu().numpy())
    true_labels.extend(labels.cpu().numpy())

# Convert lists to numpy arrays
true_labels = np.array(true_labels)
prediction_probs = np.array(prediction_probs)
true_labels_binarized = label_binarize(true_labels, classes=range(num_classes))
# Calculate ROC AUC score
print(f"true_labels_binarized shape: {true_labels_binarized.shape}")
print(f"prediction_probs shape: {prediction_probs.shape}")
assert prediction_probs.shape[1] == num_classes, "Mismatch in the number of classes for prediction probabilities"
# Now calculate ROC AUC score
roc_auc = roc_auc_score(true_labels_binarized, prediction_probs, multi_class='ovo')

print(f'ROC AUC Score: {roc_auc:.2f}')

# from sklearn.metrics import average_precision_score

# # Calculate macro-average ROC AUC
# roc_auc_macro = roc_auc_score(true_labels_binarized, prediction_probs, multi_class='ovr', average='macro')
# print(f'Macro-Average ROC AUC Score: {roc_auc_macro:.2f}')

# # Calculate macro-average precision-recall score
# precision_macro, recall_macro, _ = precision_recall_curve(true_labels_binarized.ravel(), prediction_probs.ravel())
# average_precision_macro = average_precision_score(true_labels_binarized, prediction_probs, average="macro")

# # Plotting macro-average Precision-Recall curve
# plt.figure(figsize=(8, 6))
# plt.plot(recall_macro, precision_macro, label=f'Macro-average Precision-recall (area = {average_precision_macro:.2f})')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Macro-Average Precision-Recall Curve')
# plt.legend(loc="best")
# plt.savefig('macro_avg_precision_recall_curve.png')
# plt.close()

# Plotting macro-average ROC curve
fpr_macro, tpr_macro, _ = roc_curve(true_labels_binarized.ravel(), prediction_probs.ravel())
roc_auc_macro = auc(fpr_macro, tpr_macro)

plt.figure(figsize=(8, 6))
plt.plot(fpr_macro, tpr_macro, label=f'Macro-average ROC (area = {roc_auc_macro:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Macro-Average ROC Curve')
plt.legend(loc="lower right")
plt.savefig('macro_avg_roc_curve.png')
plt.close()


# Micro-average Precision-Recall and ROC AUC
roc_auc_micro = roc_auc_score(true_labels_binarized, prediction_probs, multi_class='ovr', average='micro')
print(f'Micro-Average ROC AUC Score: {roc_auc_micro:.2f}')

precision_micro, recall_micro, _ = precision_recall_curve(true_labels_binarized.ravel(), prediction_probs.ravel())
average_precision_micro = average_precision_score(true_labels_binarized, prediction_probs, average="micro")

# Plotting micro-average Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall_micro, precision_micro, label=f'Micro-average Precision-recall (area = {average_precision_micro:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Micro-Average Precision-Recall Curve')
plt.legend(loc="best")
plt.savefig('micro_avg_precision_recall_curve.png')
plt.close()

# Plotting micro-average ROC curve
fpr_micro, tpr_micro, _ = roc_curve(true_labels_binarized.ravel(), prediction_probs.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

plt.figure(figsize=(8, 6))
plt.plot(fpr_micro, tpr_micro, label=f'Micro-average ROC (area = {roc_auc_micro:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Micro-Average ROC Curve')
plt.legend(loc="lower right")
plt.savefig('micro_avg_roc_curve.png')
plt.close()
