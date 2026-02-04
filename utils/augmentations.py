import random
import pandas as pd

import nltk # type: ignore

# Download the necessary resources for WordNet and POS tagging
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

import nlpaug.augmenter.word as naw # type: ignore

def aeda_5_line(text, punc_ratio=0.3):
    puncs = ['.', ',', '!', '?', ';', ':']
    words = text.split()
    # 1. Choose random positions 2. Insert random puncs 3. Join back
    for _ in range(max(1, int(len(words) * punc_ratio))):
        words.insert(random.randint(0, len(words)), random.choice(puncs))
    return " ".join(words)

# Initialize the augmenter once outside the function for efficiency
# This targets WordNet synonyms for Adjectives
adjective_aug = naw.SynonymAug(
    aug_src='wordnet',
    aug_p=0.3, # Change 30% of eligible words
    pos_tag_pattern='(JJ|JJR|JJS)', # Only target Adjectives
    stopwords=['not', 'no', 'never'] # Protection for sentiment logic
)

def wordswap_adjectives(text):
    """
    Wraps nlpaug to be used as a standalone function in your pipeline.
    """
    # nlpaug returns a list, so we take the first element
    augmented_list = adjective_aug.augment(text)
    return augmented_list[0] if augmented_list else text

def augment_minority_classes(df, target_cols, methods, n_aug=2):
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
            
            selected_method = random.choice(methods)
            aug_row['text'] = selected_method(row['text']) 
            new_rows.append(aug_row)
    
    return new_rows
