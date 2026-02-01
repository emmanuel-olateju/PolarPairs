from typing import List, Tuple, Union

import math
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

class PolarizationDataset(Dataset):
    def __init__(self, texts, labels,tokenizer, max_length =128, n_classes=2):
        self.texts=texts
        self.labels=labels
        self.tokenizer= tokenizer
        self.max_length = max_length # Store max_length
        self.n_classes = n_classes

    def __len__(self):
        return len(self.texts)

    def __getitem__(self,idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, 
            truncation=True, 
            padding=False, 
            max_length=self.max_length, 
            return_tensors='pt')

        # Only squeeze the batch dimension if it exists, leaving the sequence dimension alone
        item = {key: torch.as_tensor(encoding[key]).squeeze(0) for key in encoding.keys()}
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

        'polarization': source['polarization'],
        'target_text': target_aligned['text'],
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

def load_multilingual_data(
    subtask: str, 
    languages: List[str], 
    target_lang: str, 
    target_train_ratio: float = 0.5,
    test_size: float = 0.2, 
    random_state: int = 42, 
    mode: str = 'train', 
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load multilingual data with target language split between train and validation.
    
    Supports MULTI-LABEL, MULTI-CLASS classification.
    
    Args:
        subtask (str): Subtask name (e.g., 'subtask1', 'subtask2')
        languages (list): List of all language codes including target (e.g., ['swa', 'eng', 'fra'])
        target_lang (str): Target language code for validation (e.g., 'eng')
        target_train_ratio (float): Proportion of target language to include in training (default: 0.5)
        test_size (float): Proportion of target language for validation (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)
        mode (str): Data mode - 'train' or 'test' (default: 'train')
        verbose (bool): Print distribution statistics (default: True)
    
    Returns:
        tuple: (train_df, val_df) with columns:
            - text
            - polarization (label - can be list for multi-label)
            - language (source language code)
    """
    
    def parse_labels(label):
        """Parse label that could be string, list, or comma-separated"""
        if isinstance(label, list):
            return label
        elif isinstance(label, str):
            label = label.strip('[]')
            if ',' in label:
                return [int(x.strip()) for x in label.split(',')]
            else:
                return [int(label)]
        else:
            return [int(label)]
    
    def count_labels(df):
        """Count individual label occurrences in multi-label dataset"""
        all_labels = [label for labels in df['polarization_parsed'] for label in labels]
        return Counter(all_labels)
    
    all_data = []
    
    # --- 1. Load all language data ---
    for lang in languages:
        df = pd.read_csv(f'data/{subtask}/{mode}/{lang}.csv')
        df['language'] = lang
        all_data.append(df[['text', 'polarization', 'language']])
    
    # Combine all languages
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # --- 2. Separate target language from other languages ---
    target_mask = combined_df['language'] == target_lang
    target_df = combined_df[target_mask].reset_index(drop=True)
    other_langs_df = combined_df[~target_mask].reset_index(drop=True)
    
    # --- 3. Split target language (stratified for multi-label) ---
    # Parse labels
    target_df['polarization_parsed'] = target_df['polarization'].apply(parse_labels)
    
    # Create binary matrix for stratification
    mlb = MultiLabelBinarizer()
    y_binary = mlb.fit_transform(target_df['polarization_parsed'])
    
    # Stratified split using multi-label stratification
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, 
        test_size=(1 - target_train_ratio), 
        random_state=random_state
    )
    
    for train_idx, val_idx in msss.split(target_df, y_binary):
        target_train = target_df.iloc[train_idx].reset_index(drop=True)
        target_val = target_df.iloc[val_idx].reset_index(drop=True)
    
    # Clean up temporary column
    target_train = target_train.drop('polarization_parsed', axis=1)
    target_val = target_val.drop('polarization_parsed', axis=1)
    
    # --- 4. Construct final train and validation sets ---
    # Training set: All other languages + subset of target language
    train_df = pd.concat([other_langs_df, target_train], ignore_index=True)
    
    # Shuffle training set
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Validation set: Remaining subset of target language only
    val_df = target_val.copy()
    
    # --- 5. Print statistics (if verbose) ---
    if verbose:
        print("="*60)
        print(f"Dataset: {subtask} | Languages: {languages}")
        print(f"Target Language (for validation): {target_lang}")
        print("="*60)
        
        print(f"\nTotal samples across all languages: {len(combined_df)}")
        print(f"  - Target language ({target_lang}): {len(target_df)}")
        print(f"  - Other languages: {len(other_langs_df)}")
        
        print(f"\nTrain set size: {len(train_df)} ({len(train_df)/len(combined_df)*100:.1f}%)")
        print(f"  - From target language ({target_lang}): {len(target_train)} ({len(target_train)/len(train_df)*100:.1f}%)")
        print(f"  - From other languages: {len(other_langs_df)} ({len(other_langs_df)/len(train_df)*100:.1f}%)")
        
        print(f"\nValidation set size: {len(val_df)} ({len(val_df)/len(combined_df)*100:.1f}%)")
        print(f"  - From target language ({target_lang}): {len(target_val)} (100%)")
        
        print("\n--- Language Distribution in Training Set ---")
        print(train_df['language'].value_counts().sort_index())
        print(f"Percentages: {train_df['language'].value_counts(normalize=True).sort_index().to_dict()}")
        
        print("\n--- Polarization Distribution (Multi-Label) ---")
        
        # Parse labels for statistics
        train_df['polarization_parsed'] = train_df['polarization'].apply(parse_labels)
        val_df['polarization_parsed'] = val_df['polarization'].apply(parse_labels)
        
        print("\nTrain Set (individual label frequencies):")
        train_counts = count_labels(train_df)
        for label, count in sorted(train_counts.items()):
            print(f"  Label {label}: {count} ({count/len(train_df)*100:.2f}%)")
        
        print("\nValidation Set (individual label frequencies):")
        val_counts = count_labels(val_df)
        for label, count in sorted(val_counts.items()):
            print(f"  Label {label}: {count} ({count/len(val_df)*100:.2f}%)")
        
        # Show label combination distribution
        print("\n--- Label Combinations (Top 10) ---")
        print("\nTrain Set:")
        combo_counts = train_df['polarization_parsed'].apply(lambda x: str(sorted(x))).value_counts()
        for combo, count in combo_counts.head(10).items():
            print(f"  {combo}: {count} ({count/len(train_df)*100:.2f}%)")
        
        print("\nValidation Set:")
        combo_counts = val_df['polarization_parsed'].apply(lambda x: str(sorted(x))).value_counts()
        for combo, count in combo_counts.head(10).items():
            print(f"  {combo}: {count} ({count/len(val_df)*100:.2f}%)")
        
        print("\n--- Per-Language Polarization in Training (Individual Labels) ---")
        for lang in train_df['language'].unique():
            lang_data = train_df[train_df['language'] == lang]
            print(f"\n{lang}:")
            lang_counts = count_labels(lang_data)
            for label, count in sorted(lang_counts.items()):
                print(f"  Label {label}: {count} ({count/len(lang_data)*100:.2f}%)")
        
        # Clean up
        train_df = train_df.drop('polarization_parsed', axis=1)
        val_df = val_df.drop('polarization_parsed', axis=1)
        
        print("="*60)
    
    return train_df, val_df


# --- Alternative: Simpler version without target language in training ---

def load_multilingual_data_strict(
    subtask: str, 
    source_languages: List[str],
    target_lang: str, 
    test_size: float = 0.2, 
    random_state: int = 42, 
    mode: str = 'train', 
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load multilingual data where training uses ONLY source languages,
    and validation uses ONLY target language.
    
    Supports MULTI-LABEL, MULTI-CLASS classification.
    
    Args:
        subtask (str): Subtask name (e.g., 'subtask1', 'subtask2')
        source_languages (list): List of source language codes (e.g., ['swa', 'fra', 'ara'])
        target_lang (str): Target language code for validation (e.g., 'eng')
        test_size (float): Proportion of target language for validation (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)
        mode (str): Data mode - 'train' or 'test' (default: 'train')
        verbose (bool): Print distribution statistics (default: True)
    
    Returns:
        tuple: (train_df, val_df) with columns:
            - text
            - polarization (label - can be list for multi-label)
            - language (source language code)
    """
    
    def parse_labels(label):
        """Parse label that could be string, list, or comma-separated"""
        if isinstance(label, list):
            return label
        elif isinstance(label, str):
            # Handle formats like "[0, 1]" or "0,1" or "0"
            label = label.strip('[]')
            if ',' in label:
                return [int(x.strip()) for x in label.split(',')]
            else:
                return [int(label)]
        else:
            return [int(label)]
    
    def count_labels(df):
        """Count individual label occurrences in multi-label dataset"""
        all_labels = [label for labels in df['polarization_parsed'] for label in labels]
        return Counter(all_labels)
    
    # --- 1. Load source languages for training ---
    train_data = []
    for lang in source_languages:
        df = pd.read_csv(f'data/{subtask}/{mode}/{lang}.csv')
        df['language'] = lang
        train_data.append(df[['text', 'polarization', 'language']])
    
    train_df = pd.concat(train_data, ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # --- 2. Load target language and split for validation ---
    target_df = pd.read_csv(f'data/{subtask}/{mode}/{target_lang}.csv')
    target_df['language'] = target_lang
    
    # --- 3. Parse multi-label format ---
    target_df['polarization_parsed'] = target_df['polarization'].apply(parse_labels)
    
    # --- 4. Create binary matrix for stratification ---
    mlb = MultiLabelBinarizer()
    y_binary = mlb.fit_transform(target_df['polarization_parsed'])
    
    # --- 5. Stratified split for multi-label ---
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, 
        test_size=test_size, 
        random_state=random_state
    )
    
    for train_idx, val_idx in msss.split(target_df, y_binary):
        target_train = target_df.iloc[train_idx].reset_index(drop=True)
        val_df = target_df.iloc[val_idx].reset_index(drop=True)
    
    # Clean up temporary column
    target_train = target_train.drop('polarization_parsed', axis=1)
    val_df = val_df.drop('polarization_parsed', axis=1)
    
    # Optionally: Add some target language to training
    # train_df = pd.concat([train_df, target_train], ignore_index=True)
    # train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # --- 6. Print statistics (if verbose) ---
    if verbose:
        print("=" * 60)
        print(f"Dataset: {subtask}")
        print(f"Source Languages (training): {source_languages}")
        print(f"Target Language (validation): {target_lang}")
        print("=" * 60)
        print(f"\nTrain set size: {len(train_df)}")
        print(f"Validation set size: {len(val_df)}")
        
        print("\n--- Language Distribution in Training Set ---")
        print(train_df['language'].value_counts().sort_index())
        
        print("\n--- Polarization Distribution (Multi-Label) ---")
        
        # Parse labels for statistics
        train_df['polarization_parsed'] = train_df['polarization'].apply(parse_labels)
        val_df['polarization_parsed'] = val_df['polarization'].apply(parse_labels)
        
        print("\nTrain Set (individual label frequencies):")
        train_counts = count_labels(train_df)
        for label, count in sorted(train_counts.items()):
            print(f"  Label {label}: {count} ({count/len(train_df)*100:.2f}%)")
        
        print("\nValidation Set (individual label frequencies):")
        val_counts = count_labels(val_df)
        for label, count in sorted(val_counts.items()):
            print(f"  Label {label}: {count} ({count/len(val_df)*100:.2f}%)")
        
        # Show label combination distribution
        print("\n--- Label Combinations (Top 10) ---")
        print("\nTrain Set:")
        combo_counts = train_df['polarization_parsed'].apply(lambda x: str(sorted(x))).value_counts()
        for combo, count in combo_counts.head(10).items():
            print(f"  {combo}: {count} ({count/len(train_df)*100:.2f}%)")
        
        print("\nValidation Set:")
        combo_counts = val_df['polarization_parsed'].apply(lambda x: str(sorted(x))).value_counts()
        for combo, count in combo_counts.head(10).items():
            print(f"  {combo}: {count} ({count/len(val_df)*100:.2f}%)")
        
        # Clean up
        train_df = train_df.drop('polarization_parsed', axis=1)
        val_df = val_df.drop('polarization_parsed', axis=1)
        
        print("=" * 60)
    
    return train_df, val_df

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
