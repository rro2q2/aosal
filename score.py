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
from scipy.stats import wasserstein_distance
from sklearn.utils import resample
# from test import calculate_fpr_threshold

def consolidate_labels(labels):
    # A dictionary to map variations to a standardized label
    label_map = {}

    # A list of common prefixes to be removed to standardize labels
    prefixes = [
        'please', 'can you', 'help me', 'how do i', 'what\'s', 'what is', 
        'do you know how', 'i need to', 'tell me', 'i want to'
    ]

    for label in labels:
        try:
            # Create a pattern to match any of the prefixes defined above
            prefix_pattern = r'^(?:' + '|'.join(prefixes) + r')[, ]*'
            
            # Remove the matched prefixes and any leading/trailing punctuation
            standardized_label = re.sub(prefix_pattern, '', label, flags=re.I)
            standardized_label = standardized_label.strip().lower()
            
            # Remove any residual punctuation marks at the end or beginning
            standardized_label = re.sub(r'^[!?.\s]+|[!?.\s]+$', '', standardized_label)
            
            # Normalize white spaces to a single space between words
            standardized_label = re.sub(r'\s+', ' ', standardized_label)

            # Map original labels to their standardized form
            label_map[label] = standardized_label
        except Exception as e:
            print(f"Error processing label '{label}': {e}")
            # Optionally, you could map failed labels to a special category
            label_map[label] = "error"

    return label_map

def oversample_minor_classes(texts, labels):
    # Unique classes and the size of the largest class
    unique_classes = np.unique(labels)
    max_size = max(np.bincount(labels))
    
    texts_resampled = []
    labels_resampled = []
    
    # Resample each class to the maximum size
    for cls in unique_classes:
        # Filter texts and labels for the class
        class_texts = [text for text, label in zip(texts, labels) if label == cls]
        class_labels = [label for label in labels if label == cls]
        
        if len(class_texts) < max_size:
            # Resample if the current class size is less than the max size
            class_texts, class_labels = resample(class_texts, class_labels, replace=True, n_samples=max_size)
        
        texts_resampled.extend(class_texts)
        labels_resampled.extend(class_labels)
    
    return texts_resampled, labels_resampled


class Clinic150Dataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


dataset_path = "./Data/clinc-data.txt"
texts = []
labels = []

with open(dataset_path, 'r', encoding='utf-8') as file:
    for i, line in enumerate(file, 1):
        parts = line.strip().split(',', 1)  
        if len(parts) == 2:
            text, label = parts
            texts.append(text.strip())
            labels.append(label.strip())
        else:
            print(f"Error on line {i}: {line}")


# Encode labels
consolidated_label_map = consolidate_labels(labels)
new_labels = [consolidated_label_map[label] for label in labels]
label_encoder = LabelEncoder()

encoded_labels = label_encoder.fit_transform(new_labels)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Filter out 'oos' for training
ind_indices = [i for i, label in enumerate(encoded_labels) if label != 'oos']
train_texts = [texts[i] for i in ind_indices]
train_labels = [encoded_labels[i] for i in ind_indices]

print(len(train_texts),len(train_labels))
# Oversample minor classes
train_texts, train_labels = oversample_minor_classes(train_texts, train_labels)

# Split data into training and validation
train_texts, val_texts, train_encoded_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.1, random_state=42)


# Tokenize text
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# Create datasets
train_dataset = Clinic150Dataset(train_encodings, train_encoded_labels)
val_dataset = Clinic150Dataset(val_encodings, val_labels)


print("Train")
print(len(train_encoded_labels))
print(len(train_dataset))
print("val")
print(len(val_dataset))
