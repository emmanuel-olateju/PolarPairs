from typing import List, Tuple, Union

import math
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

# Define task labels
TASKS_LABELS_NAMES = {
    'subtask1': 'polarization',
    'subtask2': ['gender/sexual', 'political', 'religious', 'racial/ethnic', 'other'],
    'subtask3': ['vilification', 'extreme_language', 'stereotype', 'invalidation', 'lack_of_empathy', 'dehumanization']
}

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
    languages: List[str],  # SOURCE languages only (NOT including target)
    target_lang: str,      # Target language (loaded separately)
    target_train_ratio: float = 0.5,
    test_size: float = 0.2, 
    random_state: int = 42, 
    mode: str = 'train', 
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    MODIFIED: Load multilingual data where SOURCE languages and TARGET language 
    are specified separately. Target language is split between train and validation.
    
    Key difference from original:
    - 'languages' should NOT include target_lang
    - target_lang is loaded separately and split according to target_train_ratio
    
    Example:
        languages = ['swa', 'hau', 'amh']  # Source languages only
        target_lang = 'eng'                 # Target language (separate)
        target_train_ratio = 0.3            # 30% of eng → train, 70% → val
    
    Supports MULTI-LABEL, MULTI-CLASS classification across all subtasks.
    
    Args:
        subtask (str): Subtask name ('subtask1', 'subtask2', 'subtask3')
        languages (List[str]): Source language codes (NOT including target)
        target_lang (str): Target language code (will be loaded separately)
        target_train_ratio (float): Proportion of target language for training (default: 0.5)
        test_size (float): Deprecated, kept for compatibility (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)
        mode (str): 'train' or 'test' (default: 'train')
        verbose (bool): Print distribution statistics (default: True)
    
    Returns:
        tuple: (train_df, val_df)
            - train_df: Source languages + target_train_ratio of target
            - val_df: (1 - target_train_ratio) of target only
    """
    
    # Get label columns for this subtask
    label_cols = TASKS_LABELS_NAMES[subtask]
    is_multilabel = isinstance(label_cols, list)
    
    if verbose:
        print("\n" + "="*80)
        print(f"Loading {subtask} data - Multilingual mode")
        print(f"Source languages (train): {languages}")
        print(f"Target language (split): {target_lang}")
        print(f"Target train ratio: {target_train_ratio:.1%} train, {(1-target_train_ratio):.1%} val")
        print(f"Label type: {'Multi-label' if is_multilabel else 'Single/Multi-class'}")
        print("="*80)
    
    def parse_labels_subtask1(label):
        """Parse label for subtask1"""
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
    
    def count_labels(df, label_cols):
        """Count label occurrences"""
        if isinstance(label_cols, list):
            counts = {}
            for col in label_cols:
                counts[col] = df[col].sum()
            return counts
        else:
            all_labels = [label for labels in df['polarization_parsed'] for label in labels]
            return Counter(all_labels)
    
    # --- 1. Load SOURCE languages ---
    source_data = []
    
    if not languages:
        raise ValueError(
            "languages list is empty! "
            "Provide at least one source language (e.g., ['swa', 'hau'])"
        )
    
    # Ensure target is not in source languages
    if target_lang in languages:
        raise ValueError(
            f"Target language '{target_lang}' found in source languages list {languages}!\n"
            f"Please remove it from 'languages' parameter.\n"
            f"Correct usage: languages=['swa', 'hau', 'amh'], target_lang='eng'"
        )
    
    for lang in languages:
        try:
            df = pd.read_csv(f'data/{subtask}/{mode}/{lang}.csv')
            df['language'] = lang
            
            if is_multilabel:
                cols_to_keep = ['text', 'language'] + label_cols
            else:
                cols_to_keep = ['text', 'polarization', 'language']
            
            source_data.append(df[cols_to_keep])
            
            if verbose:
                print(f"\n✓ Loaded source language {lang}: {len(df)} samples")
                if is_multilabel:
                    for col in label_cols:
                        count = df[col].sum()
                        print(f"    {col}: {count}")
                        
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Source language file not found: data/{subtask}/{mode}/{lang}.csv"
            )
        except Exception as e:
            raise RuntimeError(f"Error loading {lang}: {str(e)}")
    
    # Combine source languages
    source_df = pd.concat(source_data, ignore_index=True)
    
    if verbose:
        print(f"\n✓ Combined source languages: {len(source_df)} samples")
    
    # --- 2. Load TARGET language separately ---
    try:
        target_df = pd.read_csv(f'data/{subtask}/{mode}/{target_lang}.csv')
        
        if len(target_df) == 0:
            raise ValueError(f"Target language file is empty: data/{subtask}/{mode}/{target_lang}.csv")
        
        target_df['language'] = target_lang
        
        if is_multilabel:
            cols_to_keep = ['text', 'language'] + label_cols
        else:
            cols_to_keep = ['text', 'polarization', 'language']
        
        target_df = target_df[cols_to_keep]
        
        if verbose:
            print(f"\n✓ Loaded target language {target_lang}: {len(target_df)} samples")
            if is_multilabel:
                for col in label_cols:
                    count = target_df[col].sum()
                    print(f"    {col}: {count}")
                    
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Target language file not found: data/{subtask}/{mode}/{target_lang}.csv"
        )
    except Exception as e:
        raise RuntimeError(f"Error loading target language {target_lang}: {str(e)}")
    
    # --- 3. Split target language (stratified) ---
    if is_multilabel:
        # Subtask 2 or 3: use binary columns directly
        y_binary = target_df[label_cols].values
    else:
        # Subtask 1: parse and binarize
        target_df['polarization_parsed'] = target_df['polarization'].apply(parse_labels_subtask1)
        mlb = MultiLabelBinarizer()
        y_binary = mlb.fit_transform(target_df['polarization_parsed'])
    
    # Validate we have enough samples
    if len(target_df) < 2:
        raise ValueError(
            f"Not enough target language samples! "
            f"Need at least 2 samples, got {len(target_df)}"
        )
    
    # Stratified split
    val_ratio = 1.0 - target_train_ratio
    
    try:
        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1, 
            test_size=val_ratio,
            random_state=random_state
        )
        
        for train_idx, val_idx in msss.split(target_df, y_binary):
            target_train = target_df.iloc[train_idx].reset_index(drop=True)
            target_val = target_df.iloc[val_idx].reset_index(drop=True)
        
        if verbose:
            print(f"\n✓ Split target language:")
            print(f"    → Train portion: {len(target_train)} samples ({target_train_ratio:.1%})")
            print(f"    → Val portion: {len(target_val)} samples ({val_ratio:.1%})")
            
    except Exception as e:
        raise RuntimeError(f"Error during stratified split: {str(e)}")
    
    # Clean up temporary column for subtask1
    if not is_multilabel:
        if 'polarization_parsed' in target_train.columns:
            target_train = target_train.drop('polarization_parsed', axis=1)
        if 'polarization_parsed' in target_val.columns:
            target_val = target_val.drop('polarization_parsed', axis=1)
    
    # --- 4. Construct final train and validation sets ---
    # Train = ALL source languages + target_train portion
    train_df = pd.concat([source_df, target_train], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Validation = target_val portion only
    val_df = target_val.copy()
    
    # --- 5. Print final statistics ---
    if verbose:
        print("\n" + "="*80)
        print("FINAL DATASET STATISTICS")
        print("="*80)
        
        print(f"\nTraining set: {len(train_df)} samples")
        lang_dist = train_df['language'].value_counts().to_dict()
        for lang, count in sorted(lang_dist.items()):
            print(f"  {lang}: {count} ({count/len(train_df)*100:.1f}%)")
        
        print(f"\nValidation set: {len(val_df)} samples")
        print(f"  {target_lang}: {len(val_df)} (100%)")
        
        # Label distribution
        if is_multilabel:
            print("\nTraining set label distribution:")
            train_counts = count_labels(train_df, label_cols)
            for label, count in train_counts.items():
                print(f"  {label}: {count} ({count/len(train_df)*100:.2f}%)")
            
            print("\nValidation set label distribution:")
            val_counts = count_labels(val_df, label_cols)
            for label, count in val_counts.items():
                print(f"  {label}: {count} ({count/len(val_df)*100:.2f}%)")
        else:
            # For subtask1
            print("\nTraining set polarization distribution:")
            train_polar_dist = train_df['polarization'].value_counts()
            for label, count in sorted(train_polar_dist.items()):
                print(f"  Label {label}: {count} ({count/len(train_df)*100:.2f}%)")
            
            print("\nValidation set polarization distribution:")
            val_polar_dist = val_df['polarization'].value_counts()
            for label, count in sorted(val_polar_dist.items()):
                print(f"  Label {label}: {count} ({count/len(val_df)*100:.2f}%)")
        
        print("="*80 + "\n")
    
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
    
    Supports MULTI-LABEL, MULTI-CLASS classification across all subtasks.
    
    Args:
        subtask (str): Subtask name ('subtask1', 'subtask2', 'subtask3')
        source_languages (list): List of source language codes (e.g., ['swa', 'fra', 'ara'])
        target_lang (str): Target language code for validation (e.g., 'eng')
        test_size (float): Proportion of target language for validation (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)
        mode (str): Data mode - 'train' or 'test' (default: 'train')
        verbose (bool): Print distribution statistics (default: True)
    
    Returns:
        tuple: (train_df, val_df)
    """
    
    # Get label columns for this subtask
    label_cols = TASKS_LABELS_NAMES[subtask]
    is_multilabel = isinstance(label_cols, list)
    
    def parse_labels_subtask1(label):
        """Parse label for subtask1 (single or multi-class)"""
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
    
    def count_labels(df, label_cols):
        """Count label occurrences"""
        if isinstance(label_cols, list):
            # Multi-label case (subtask2, subtask3)
            counts = {}
            for col in label_cols:
                counts[col] = df[col].sum()
            return counts
        else:
            # Single label case (subtask1)
            all_labels = [label for labels in df['polarization_parsed'] for label in labels]
            return Counter(all_labels)
    
    # --- 1. Load source languages for training ---
    train_data = []
    
    if not source_languages:
        raise ValueError("source_languages cannot be empty!")
    
    for lang in source_languages:
        file_path = f'data/{subtask}/{mode}/{lang}.csv'
        
        try:
            df = pd.read_csv(file_path)
            df['language'] = lang
            
            # Select appropriate columns based on subtask
            if is_multilabel:
                # Subtask 2 or 3: multi-label binary columns
                cols_to_keep = ['text', 'language'] + label_cols
            else:
                # Subtask 1: single polarization column
                cols_to_keep = ['text', 'polarization', 'language']
            
            train_data.append(df[cols_to_keep])
            
        except FileNotFoundError:
            print(f"WARNING: File not found: {file_path}")
        except KeyError as e:
            print(f"ERROR: Missing columns in {file_path}: {e}")
            print(f"  Available columns: {df.columns.tolist()}")
            raise
    
    if not train_data:
        raise ValueError(
            f"No data loaded! Check that files exist in 'data/{subtask}/{mode}/' "
            f"for languages: {source_languages}"
        )
    
    train_df = pd.concat(train_data, ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # --- 2. Load target language and split for validation ---
    target_file_path = f'data/{subtask}/{mode}/{target_lang}.csv'
    
    try:
        target_df = pd.read_csv(target_file_path)
        target_df['language'] = target_lang
    except FileNotFoundError:
        raise FileNotFoundError(f"Target language file not found: {target_file_path}")
    
    # --- 3. Prepare labels for stratification ---
    if is_multilabel:
        # Subtask 2 or 3: Use binary columns directly
        y_binary = target_df[label_cols].values
    else:
        # Subtask 1: Parse and binarize
        target_df['polarization_parsed'] = target_df['polarization'].apply(parse_labels_subtask1)
        mlb = MultiLabelBinarizer()
        y_binary = mlb.fit_transform(target_df['polarization_parsed'])
    
    # --- 4. Stratified split for multi-label ---
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, 
        test_size=test_size, 
        random_state=random_state
    )
    
    for train_idx, val_idx in msss.split(target_df, y_binary):
        target_train = target_df.iloc[train_idx].reset_index(drop=True)
        val_df = target_df.iloc[val_idx].reset_index(drop=True)
    
    # Clean up temporary column for subtask1
    if not is_multilabel:
        target_train = target_train.drop('polarization_parsed', axis=1, errors='ignore')
        val_df = val_df.drop('polarization_parsed', axis=1, errors='ignore')
    
    # --- 5. Print statistics (if verbose) ---
    if verbose:
        print("=" * 60)
        print(f"Dataset: {subtask}")
        print(f"Source Languages (training): {source_languages}")
        print(f"Target Language (validation): {target_lang}")
        print(f"Label type: {'Multi-label' if is_multilabel else 'Multi-class'}")
        print("=" * 60)
        print(f"\nTrain set size: {len(train_df)}")
        print(f"Validation set size: {len(val_df)}")
        
        print("\n--- Language Distribution in Training Set ---")
        print(train_df['language'].value_counts().sort_index())
        
        if is_multilabel:
            # Subtask 2 or 3 statistics
            print("\n--- Label Distribution (Multi-Label Binary) ---")
            
            print("\nTrain Set:")
            train_label_counts = count_labels(train_df, label_cols)
            for label, count in train_label_counts.items():
                print(f"  {label}: {count} ({count/len(train_df)*100:.2f}%)")
            
            print("\nValidation Set:")
            val_label_counts = count_labels(val_df, label_cols)
            for label, count in val_label_counts.items():
                print(f"  {label}: {count} ({count/len(val_df)*100:.2f}%)")
            
            # Show label combinations
            print("\n--- Label Combinations (Top 10) ---")
            train_df['label_combo'] = train_df[label_cols].apply(
                lambda row: str([col for col in label_cols if row[col] == 1]), axis=1
            )
            val_df['label_combo'] = val_df[label_cols].apply(
                lambda row: str([col for col in label_cols if row[col] == 1]), axis=1
            )
            
            print("\nTrain Set:")
            combo_counts = train_df['label_combo'].value_counts()
            for combo, count in combo_counts.head(10).items():
                print(f"  {combo}: {count} ({count/len(train_df)*100:.2f}%)")
            
            print("\nValidation Set:")
            combo_counts = val_df['label_combo'].value_counts()
            for combo, count in combo_counts.head(10).items():
                print(f"  {combo}: {count} ({count/len(val_df)*100:.2f}%)")
            
            # Clean up
            train_df = train_df.drop('label_combo', axis=1)
            val_df = val_df.drop('label_combo', axis=1)
            
        else:
            # Subtask 1 statistics
            print("\n--- Polarization Distribution (Multi-Class) ---")
            
            train_df['polarization_parsed'] = train_df['polarization'].apply(parse_labels_subtask1)
            val_df['polarization_parsed'] = val_df['polarization'].apply(parse_labels_subtask1)
            
            print("\nTrain Set (individual label frequencies):")
            train_counts = count_labels(train_df, 'polarization')
            for label, count in sorted(train_counts.items()):
                print(f"  Label {label}: {count} ({count/len(train_df)*100:.2f}%)")
            
            print("\nValidation Set (individual label frequencies):")
            val_counts = count_labels(val_df, 'polarization')
            for label, count in sorted(val_counts.items()):
                print(f"  Label {label}: {count} ({count/len(val_df)*100:.2f}%)")
            
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


class CrossLingualDataset(Dataset):
    """Dataset for TN_PolarPairs with multi-label support"""
    def __init__(self, dataframe, tokenizer, subtask, max_length=128, mode='train'):
        """
        Args:
            dataframe: DataFrame with columns ['text', 'language'] + label columns
            tokenizer: HuggingFace tokenizer or tokenizer name
            subtask: 'subtask1', 'subtask2', or 'subtask3'
            max_length: Maximum sequence length
            mode: 'train' or 'eval'
        """
        self.data = dataframe.reset_index(drop=True)
        
        # Handle tokenizer
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
            
        self.max_length = max_length
        self.subtask = subtask
        self.mode = mode
        
        # Get label columns based on subtask
        self.label_cols = TASKS_LABELS_NAMES[subtask]
        self.is_multilabel = isinstance(self.label_cols, list)
        
        # Validate data
        if 'text' not in self.data.columns:
            raise ValueError("DataFrame must have 'text' column")
        
        if self.is_multilabel:
            missing_cols = [col for col in self.label_cols if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"Missing label columns: {missing_cols}")
        else:
            if self.label_cols not in self.data.columns:
                raise ValueError(f"Missing label column: {self.label_cols}")
        
        print(f"Dataset initialized: {len(self.data)} samples")
        print(f"Mode: {mode}, Subtask: {subtask}, Multi-label: {self.is_multilabel}")
        print(f"Label columns: {self.label_cols}")
        
        # Print label distribution
        if self.is_multilabel:
            print("Label distribution:")
            for col in self.label_cols:
                count = self.data[col].sum()
                print(f"  {col}: {count} ({count/len(self.data)*100:.1f}%)")
        else:
            print(f"Label distribution:\n{self.data[self.label_cols].value_counts()}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            row = self.data.iloc[idx]
            text = str(row['text']).strip()
            
            # Handle empty text
            if not text:
                text = "[EMPTY]"
            
            # Tokenize - return lists, NOT tensors
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
            )
            
            # Prepare item - keep as lists (NO .squeeze(0))
            item = {
                'x_input_ids': encoding['input_ids'].squeeze(0),        # List
                'x_attention_mask': encoding['attention_mask'].squeeze(0),  # List
            }
            
            if self.is_multilabel:
                labels = [int(row[col]) for col in self.label_cols]
                item['polar_labels'] = torch.tensor(labels, dtype=torch.float)
            else:
                # ... existing logic ...
                label_value = row[self.label_cols]
                item['polar_labels'] = torch.tensor(label_value, dtype=torch.long)
                
                if isinstance(label_value, str):
                    # Parse string format like "[0, 1]" or "0"
                    label_value = label_value.strip('[]')
                    if ',' in label_value:
                        # Multi-class (multiple labels) - return as LIST
                        labels = [int(x.strip()) for x in label_value.split(',')]
                    else:
                        # Single class - return as INT
                        labels = int(label_value)
                else:
                    # Already numeric - return as INT
                    labels = int(label_value)
            
            item['polar_labels'] = labels  # This is now a list or int, NOT a tensor
            
            # Add language info if available (optional)
            if 'language' in row:
                item['language'] = row['language']
            
            return item
            
        except Exception as e:
            print(f"\nERROR at index {idx}:")
            print(f"  Error: {e}")
            print(f"  Row: {self.data.iloc[idx].to_dict()}")
            raise
