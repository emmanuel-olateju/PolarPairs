import math
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

class PolarizationDataset(Dataset):
    def __init__(self,texts,labels,tokenizer,max_length =128, n_classes=2):
        self.texts=texts
        self.labels=labels
        self.tokenizer= tokenizer
        self.max_length = max_length # Store max_length
        self.n_classes = n_classes

    def __len__(self):
        return len(self.texts)

    def __getitem__(self,idx):
        text=self.texts[idx]
        label=self.labels[idx]
        encoding=self.tokenizer(text,truncation=True,padding=False,max_length=self.max_length,return_tensors='pt')

        # Ensure consistent tensor conversion for all items
        item = {key: encoding[key].squeeze() for key in encoding.keys()}
        if self.n_classes > 2:
            item['labels'] = torch.tensor(label, dtype=torch.float)    
        else:
            item['labels'] = torch.tensor(label, dtype=torch.long)
        return item


def load_and_split_bilingual_data(subtask, source_lang, target_lang, test_size=0.2, random_state=42, mode='train', verbose=True):
    """
    Load and split bilingual polarization data with stratified sampling.

    Args:
        subtask (str): Subtask name (e.g., 'subtask1', 'subtask2')
        source_lang (str): Source language code (e.g., 'swa', 'eng')
        target_lang (str): Target language code (e.g., 'eng', 'swa')
        test_size (float): Proportion of validation set (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)
        verbose (bool): Print distribution statistics (default: True)

    Returns:
        tuple: (train_df, val_df) with columns:
            - source_text
            - polarization (source label)
            - target_text
            - target_polarization (target label)
    """

    # Load the DataFrames
    source = pd.read_csv(f'data/{subtask}/{mode}/{source_lang}.csv')
    target = pd.read_csv(f'data/{subtask}/{mode}/{target_lang}.csv')

    # --- 1. Shape Matching and Alignment ---
    source_len = source.shape[0]
    target_len = target.shape[0]

    if target_len < source_len:
        # Case A: Target is shorter than Source (Repeat/Tile the Target)
        repeat_factor = math.ceil(source_len / target_len)
        target_aligned = pd.concat([target] * repeat_factor, ignore_index=True).iloc[0:source_len]

    elif target_len > source_len:
        # Case B: Target is longer than Source (Truncate the Target)
        target_aligned = target.iloc[0:source_len]

    else:
        # Case C: Target and Source are already the same length
        target_aligned = target.copy()

    # --- 2. Data Combination ---
    data = pd.DataFrame({
        'source_text': source['text'],
        'polarization': source['polarization'],
        'target_text': target_aligned['text'],
        'target_polarization': target_aligned['polarization']
    })

    # --- 3. Stratified Splitting ---
    sss_source = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    # Split based on source polarization
    for train_index, val_index in sss_source.split(data, data['polarization']):
        train = data.iloc[train_index].reset_index(drop=True)
        val = data.iloc[val_index].reset_index(drop=True)

    # Stratified split for target language in validation set
    full_target_pool = data[['target_text', 'target_polarization']].copy()
    val_size = val.shape[0]
    test_ratio = val_size / data.shape[0]

    sss_target = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
    for _, new_val_target_index in sss_target.split(full_target_pool, full_target_pool['target_polarization']):
        new_val_target = full_target_pool.iloc[new_val_target_index].reset_index(drop=True)

    # Verify and update validation target data
    if new_val_target.shape[0] == val.shape[0]:
        val['target_text'] = new_val_target['target_text'].values
        val['target_polarization'] = new_val_target['target_polarization'].values
    else:
        raise ValueError(f"Stratified target split size ({new_val_target.shape[0]}) "
                        f"does not match validation set size ({val.shape[0]}).")

    # --- 4. Print Results (if verbose) ---
    if verbose:
        print("="*60)
        print(f"Dataset: {subtask} | {source_lang} → {target_lang}")
        print("="*60)
        print(f"Total samples: {data.shape[0]}")
        print(f"Train set size: {train.shape[0]} ({train.shape[0]/data.shape[0]*100:.1f}%)")
        print(f"Validation set size: {val.shape[0]} ({val.shape[0]/data.shape[0]*100:.1f}%)")

        print("\n--- Polarization Distribution ---")
        print("\nTrain Set (Source):")
        print(train['polarization'].value_counts().sort_index())
        print(f"  Ratio: {train['polarization'].value_counts(normalize=True).sort_index().to_dict()}")

        print("\nValidation Set (Source):")
        print(val['polarization'].value_counts().sort_index())
        print(f"  Ratio: {val['polarization'].value_counts(normalize=True).sort_index().to_dict()}")

        print("\nValidation Set (Target):")
        print(val['target_polarization'].value_counts().sort_index())
        print(f"  Ratio: {val['target_polarization'].value_counts(normalize=True).sort_index().to_dict()}")
        print("="*60)

    return train, val


class PolarPairsDataset(Dataset):
    """Combined dataset with stratified target sampling"""
    def __init__(self, source_texts, target_texts, source_labels, target_labels,
                 tokenizer, max_length=128, subset_size=20):
        self.source_labels = source_labels
        self.target_labels = np.array(target_labels)
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_length = max_length
        self.subset_size = subset_size

        # Pre-compute indices by class for stratified sampling
        self.pos_indices = np.where(self.target_labels == 1)[0]
        self.neg_indices = np.where(self.target_labels == 0)[0]

        self.mode = 'train'

        print(f"Target distribution - Positive: {len(self.pos_indices)}, "
              f"Negative: {len(self.neg_indices)}")

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source = str(self.source_texts[idx])
        source_label = int(self.source_labels[idx])

        # Stratified sampling: ensure balanced positive/negative targets
        n_pos = self.subset_size // 2
        n_neg = self.subset_size - n_pos

        # Sample with replacement if needed
        pos_sample = np.random.choice(self.pos_indices, size=n_pos,
                                     replace=len(self.pos_indices) < n_pos)
        neg_sample = np.random.choice(self.neg_indices, size=n_neg,
                                     replace=len(self.neg_indices) < n_neg)

        target_indices = np.concatenate([pos_sample, neg_sample])
        np.random.shuffle(target_indices)

        target_texts_batch = [str(self.target_texts[i]) for i in target_indices]
        target_labels_batch = [int(self.target_labels[i]) for i in target_indices]

        # Encode
        source_encoding = self.tokenizer(
            source, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )

        target_encoding = self.tokenizer(
            target_texts_batch, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )

        if self.mode == 'train':
          return {
              'x_input_ids': source_encoding['input_ids'].squeeze(0),
              'x_attention_mask': source_encoding['attention_mask'].squeeze(0),
              'x_hat_input_ids': target_encoding['input_ids'],
              'x_hat_attention_mask': target_encoding['attention_mask'],
              'polar_labels': torch.tensor(source_label, dtype=torch.long),
              'hat_labels': torch.tensor(target_labels_batch, dtype=torch.long)
          }
        else:
          return {
              'x_input_ids': source_encoding['input_ids'].squeeze(0),
              'x_attention_mask': source_encoding['attention_mask'].squeeze(0),
              'polar_labels': torch.tensor(source_label, dtype=torch.long),
          }
