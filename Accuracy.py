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
from sklearn.metrics import accuracy_score, f1_score
from Model import val_dataset,train_dataset, val_labels,train_labels,train_texts,val_texts

model_save_directory = "./model_save"
tokenizer_save_directory = "./tokenizer_save"
model = BertForSequenceClassification.from_pretrained(model_save_directory)
tokenizer = BertTokenizerFast.from_pretrained(tokenizer_save_directory)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
model.to(device)

# Ensure DataLoader is set up correctly
val_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)

val_preds = []
vval_labels = []

for batch in val_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    # Print shapes for debugging
    # print("Shape of input_ids:", input_ids.shape)
    # print("Shape of attention_mask:", attention_mask.shape)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1)

    # Move logits and labels to CPU and convert to numpy arrays
    predictions = predictions.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    # Ensure labels are at least 1-D
    labels = np.atleast_1d(labels)

    # Store predictions and true labels
    val_preds.extend(predictions)
    vval_labels.extend(labels)

# Calculate accuracy
val_accuracy = accuracy_score(vval_labels, val_preds)
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')
f1 = f1_score(vval_labels, val_preds, average='weighted')  
print(f'Validation F1-Score: {f1:.2f}')