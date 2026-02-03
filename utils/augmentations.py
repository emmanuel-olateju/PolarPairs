import random
import pandas as pd

def aeda_5_line(text, punc_ratio=0.3):
    puncs = ['.', ',', '!', '?', ';', ':']
    words = text.split()
    # 1. Choose random positions 2. Insert random puncs 3. Join back
    for _ in range(max(1, int(len(words) * punc_ratio))):
        words.insert(random.randint(0, len(words)), random.choice(puncs))
    return " ".join(words)

def augment_minority_classes(df, target_cols, method, n_aug=2):
    """
    df: Your training dataframe
    target_cols: List of column names representing minority classes (e.g., ['Dehumanization', 'Vilification'])
    n_aug: Number of augmented versions to create per identified row
    """
    # 1. Identify rows where ANY of the target columns have a 1
    minority_mask = (df[target_cols] == 1).any(axis=1)
    minority_df = df[minority_mask]
    
    new_rows = []
    
    # 2. Iterate only over the identified minority rows
    for _, row in minority_df.iterrows():
        for _ in range(n_aug):
            aug_row = row.copy()
            # Apply your preferred augmentation (AEDA or nlpaug)
            aug_row['text'] = method(row['text']) 
            new_rows.append(aug_row)
    
    return new_rows
