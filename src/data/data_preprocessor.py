# Author: Roland Oruche
# Affiliation: University of Missouri-Columbia
# Year: 2024

from typing import Optional, Tuple, List, Union
import numpy as np
import random
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
from utils.train_utils import collate_fn, collate_fn_binary_labels

# Intialize random seed
random.seed(42)
np.random.seed(42)

class MainDataset(Dataset):
    """ Instantiates text dataset.
    Functions:
        __len__(): returns length of the dataset.
        __getitem__(): retrieves a input and target sample based on index.
    """
    def __init__(self, data) -> None:
        self.inputs = data[0]
        self.targets = data[1]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index) -> Tuple[str, torch.LongTensor, int]:
        input_sample = self.inputs[index]
        target_sample = torch.LongTensor([self.targets[index]])
        return input_sample, target_sample, index


class DataPreprocessor:
    """
    Procceses raw dataset and prepares IND data and OOD data for partitioning and tokenization.
    Functions:
        get_ind_data(): Filters and retrieves IND data from full dataset.
        get_ood_data(): Filters and retrieves OOD data from full dataset.
        prepare_dataset(): Prepares IND/OOD data paritions via splitting and tokenizing.
        train_dataloader(): Returns dataloader for train split.
        val_dataloader(): Returns dataloader for val split.
        test_dataloader(): Returns dataloader for test split.
        ood_val_dataloader(): Returns dataloader for ood data in validation set.
    """
    def __init__(
        self,
        data_dir: str,
        train_test_split: Tuple[float, float],
        ind_dataset: str,
        ind_ratio: float,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        max_seq_length: int,
        ood_dataset: Optional[str] = None,
        ood_ratio: Optional[float] = None,
        noise_ratio: Optional[float] = None
    ) -> None:
        self.data_dir = data_dir
        self.train_test_split = train_test_split
        self.ind_ratio = ind_ratio
        self.ood_ratio = ood_ratio
        self.ind_dataset = ind_dataset
        self.ood_dataset = ood_dataset
        self.noise_ratio = noise_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_seq_length = max_seq_length

        self.ind_samples = []
        self.ood_samples = []
        self.labeled_samples = []
        # Intialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.train_data_idxs = []
        self.val_ood_data_idxs: Optional[List] = None

    def get_ind_data(self) -> List[Dict[str, Union[int, str]]]:
        ind_data_full = dict()
        ind_data_full['train'] = dict()
        with open(f'{self.data_dir}/{self.ind_dataset}-data.txt', 'r') as fp:
            data = fp.readlines()
        # Get IND labels
        self.labels = dict()
        label_idx = 0
        # Process text and labels
        for d in data:
            sample = d.strip()
            # Check if sample is OOD
            if sample.split(',')[-1] == 'oos':
                self.ood_samples.append(sample)
            else: # sample is IND
                text = ','.join(sample.split(',')[:-1])
                label = sample.split(',')[-1].strip()
                if label not in self.labels:
                    self.labels[label] = label_idx
                    label_idx += 1
                self.ind_samples.append({'text': text, 'label': self.labels[label]})

        train_len = int(len(self.ind_samples) * self.train_test_split[0])
        # Set initial train length from IND samples
        self.init_train_len = int(self.ind_ratio * train_len)
        # Set validation length from IND samples
        self.val_len = int(len(self.ind_samples) * (self.train_test_split[1] / 2))
        return self.ind_samples

    def get_ood_data(self) -> List[Dict[str, Union[int, str]]]:
        ood_data_full = dict()
        if self.ood_dataset:
            with open(f'{self.data_dir}/{self.ood_dataset}-data.txt', 'r') as fp:
                data = fp.readlines()
            data += self.ood_samples
        else:
            data = self.ood_samples
        self.ood_samples = []
        # Process OOD text samples and labels
        for d in data:
            sample = d.strip()
            text = ','.join(sample.split(',')[:-1])
            label = sample.split(',')[-1].strip()
            # if label not in self.labels:
            self.ood_samples.append({'text': text, 'label': 999})
        # Set length OOD labeled for validation set 
        self.ood_lbl_len = int(len(self.ood_samples) * self.ood_ratio)
        return self.ood_samples

    def prepare_dataset(self) -> Tuple[MainDataset, np.ndarray, np.ndarray]:
        print("Preparing the dataset...")
        self.ind_samples = self.get_ind_data()
        # Add OOD dataset if present, otherwise prepare IND dataset only.
        if self.ood_dataset or self.ood_samples:
            ood_data = self.get_ood_data()
            full_dataset = self.ind_samples + ood_data
        else:
            full_dataset = self.ind_samples
        # Tokenize
        text = [self.tokenizer(sample["text"], max_length=self.max_seq_length, truncation=True) for _, sample in enumerate(full_dataset)]
        labels = [sample["label"] for _,sample in enumerate(full_dataset)]
        data = (text, labels)
        # Intialize full dataset
        main_dataset = MainDataset(data)
        # Create boolean array for tracking when data is labeled (True = labeled, False = unlabeled).
        self.labeled_samples = np.array([False] * main_dataset.__len__())

        ##### Perform data splits #####
        # Get IND (labeled) splits
        ind_idxs = [idx for idx, sample in enumerate(main_dataset) if sample[1].item() != 999]

        # Initialize train labeled set.
        self.train_data_idxs = np.random.choice(ind_idxs, self.init_train_len, replace=False)
        self.labeled_samples[self.train_data_idxs] = True
        ind_idxs = [ind_idxs[i] for i in range(len(ind_idxs)) if not self.labeled_samples[ind_idxs[i]]]

        # Initialize val labeled set.
        val_data_idxs = np.random.choice(ind_idxs, self.val_len, replace=False)
        self.labeled_samples[val_data_idxs] = True
        ind_idxs = [ind_idxs[i] for i in range(len(ind_idxs)) if not self.labeled_samples[ind_idxs[i]]]

        # Initialize test labeled set.
        test_data_idxs = np.random.choice(ind_idxs, self.val_len, replace=False)
        self.labeled_samples[test_data_idxs] = True

        # Remaining unlabeled IND instances
        self.unlabeled_ind_idxs = [ind_idxs[i] for i in range(len(ind_idxs)) if not self.labeled_samples[ind_idxs[i]]]
    
        if self.ood_dataset or self.ood_samples:
            # Get OOD (labeled) split
            ood_idxs = [idx for idx, sample in enumerate(main_dataset) if sample[1].item() == 999]
            # Initialize OOD val labeled set.
            self.val_ood_data_idxs = np.random.choice(ood_idxs, self.ood_lbl_len)
            self.labeled_samples[self.val_ood_data_idxs] = True
        if self.ood_dataset:
            # Remaining unlabled OOD instances
            ood_idxs = [ood_idxs[i] for i in range(len(ood_idxs)) if not self.labeled_samples[ood_idxs[i]]]
            # Adjust length of OOD unlabeled set with the noise ratio
            adjusted_ood_len = int(abs(len(self.unlabeled_ind_idxs) - (len(self.unlabeled_ind_idxs) + \
                (len(self.unlabeled_ind_idxs) * self.noise_ratio))) / (1-self.noise_ratio))
            self.unlabeled_ood_idxs = ood_idxs[:adjusted_ood_len]
            removed_ood_idxs = np.array(ood_idxs[adjusted_ood_len:])
            self.labeled_samples[removed_ood_idxs] = True
        # Get unlabeled set
        self.unlabeled_idxs = [idx for idx, _ in enumerate(main_dataset) if not self.labeled_samples[idx]]
        return main_dataset, val_data_idxs, test_data_idxs
        
        
    def train_dataloader(self, main_dataset: MainDataset) -> DataLoader:
        data_train = Subset(main_dataset, self.train_data_idxs)
        return DataLoader(
            dataset=data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self, main_dataset: MainDataset, data_idxs: np.ndarray) -> DataLoader:
        data_val = Subset(main_dataset, data_idxs)
        return DataLoader(
            dataset=data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=True
        )

    def test_dataloader(self, main_dataset: MainDataset, data_idxs: np.ndarray) -> DataLoader:
        data_test = Subset(main_dataset, data_idxs)
        return DataLoader(
            dataset=data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=True
        )

    def ood_val_dataloader(self, main_dataset: MainDataset, val_idxs: np.ndarray) -> DataLoader:
        ood_data_val = Subset(main_dataset, np.concatenate((val_idxs, self.val_ood_data_idxs)))
        return DataLoader(
            dataset=ood_data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn_binary_labels,
            shuffle=False,
            drop_last=True
        )

def print_dataset_statistics(dataset: DataPreprocessor) -> None:
    """ Prints the statistics of the dataset.
    @param dataset - Text dataset instantiated by `MainDataset` class.
    """
    print("IND dataset: ", dataset.ind_dataset)
    print("OOD dataset: ", dataset.ood_dataset)

    print("----- Total length of IND dataset: {} | train/val/test split on IND: {} -----\n\
        Percentage of IND samples assigned for training: {} | Length of IND train (labeled): {}\n\
        Percentage of IND samples assigned to unlabeled pool: {:.2f} Length of IND train (unlabeled): {}\n\
        Length of IND val: {} \n\
        Length of IND test: {} \n\
        ".format(
            len(dataset.ind_samples),
            (dataset.train_test_split[0], (dataset.train_test_split[1] / 2), (dataset.train_test_split[1] / 2)),
            dataset.ind_ratio,
            dataset.init_train_len,
            100*(len(dataset.unlabeled_ind_idxs) / (len(dataset.unlabeled_ind_idxs) + len(dataset.unlabeled_ood_idxs))),
            len(dataset.unlabeled_ind_idxs),
            dataset.val_len,
            dataset.val_len
        )
    )
    if dataset.ood_dataset or dataset.ood_samples:
        print("----- Total length of OOD dataset: {} -----\n\
            Percentage of OOD samples assigned for validation: {} | Length of OOD (labeled): {} \n\
            Percentage of OOD samples assigned to unlabeled pool: {:.2f} (noise ratio) | Length of OOD (unlabeled): {} \n\
            ".format(
                len(dataset.ood_samples),
                dataset.ood_ratio,
                dataset.ood_lbl_len,
                100*(len(dataset.unlabeled_ood_idxs) / (len(dataset.unlabeled_ind_idxs) + len(dataset.unlabeled_ood_idxs))),
                len(dataset.unlabeled_ood_idxs)
            )
        )

if __name__ == '__main__':
    _ = DataPreprocessor()
