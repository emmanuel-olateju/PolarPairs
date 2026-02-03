import random
import pandas as pd

def aeda_5_line(text, punc_ratio=0.3):
    puncs = ['.', ',', '!', '?', ';', ':']
    words = text.split()
    # 1. Choose random positions 2. Insert random puncs 3. Join back
    for _ in range(max(1, int(len(words) * punc_ratio))):
        words.insert(random.randint(0, len(words)), random.choice(puncs))
    return " ".join(words)

# Simple expansion logic
def aeda_minority_classes(df, target_classes=[], n_aug=2):
    new_rows = []
    for _, row in df[df['label'].isin(target_classes)].iterrows():
        for _ in range(n_aug):
            aug_row = row.copy()
            aug_row['text'] = aeda_5_line(row['text']) # or nlpaug
            new_rows.append(aug_row)
    
    return new_rows
