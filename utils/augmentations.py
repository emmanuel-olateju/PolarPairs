import random
import pandas as pd

import nltk # type: ignore
# Download the necessary resources for WordNet and POS tagging
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
import nlpaug.augmenter.word as naw # type: ignore

from transformers import MarianMTModel, MarianTokenizer # type: ignore
import torch # type: ignore
# Mapping dictionary
LANG_MAP = {
    'eng': 'en',
    'swa': 'sw',
    'hau': 'ha',
    'amh': 'am'
}

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
    # pos_tag_pattern='(JJ|JJR|JJS)', # Only target Adjectives
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
        is_english = row.get('lang', 'eng') == 'eng'

        for _ in range(n_aug):
            aug_row = row.copy()

            selected_method = random.choice(methods)

            if selected_method == back_translate:
                lang = aug_row['language']
                aug_row['text'] = back_translate(row['text'], source_code=lang)
            elif selected_method == wordswap_adjectives:
                if is_english:
                    aug_row['text'] = wordswap_adjectives(row['text']) 
                else:
                    aug_row['text'] = aeda_5_line(row['text'])
            elif selected_method == aeda_5_line:
                aug_row['text'] = aeda_5_line(row['text'])
            else:
                aug_row['text'] = row['text']

                
            new_rows.append(aug_row)
    
    return new_rows

# ======================
# BACK TRANSLATION
# ======================

def get_marian_translator(source_lang, target_lang):
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer

def local_translate(text, model, tokenizer):
    if not text: return text
    # Tokenize and Generate
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        translated_tokens = model.generate(**inputs)
    # Decode
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

# Cache for loaded models to prevent re-loading every time
MODEL_CACHE = {}

def get_cached_translator(src, tgt):
    pair = f"{src}-{tgt}"
    if pair not in MODEL_CACHE:
        model_name = f'Helsinki-NLP/opus-mt-{src}-{tgt}'
        print(f"--- Loading model: {model_name} ---")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
        MODEL_CACHE[pair] = (model, tokenizer)
    return MODEL_CACHE[pair]

def back_translate(text, source_code):
    """
    Translates from source_code -> English -> source_code
    Example: source_code='swa'
    """
    if source_code == 'eng' or source_code not in LANG_MAP:
        # If English, we use a bridge like French or German
        src, bridge = 'en', 'fr'
    else:
        src = LANG_MAP[source_code]
        bridge = 'en'

    try:
        # 1. Forward to Bridge
        f_model, f_tokenizer = get_cached_translator(src, bridge)
        intermediate_text = local_translate(text, f_model, f_tokenizer)
        
        # 2. Back to Source
        b_model, b_tokenizer = get_cached_translator(bridge, src)
        return local_translate(intermediate_text, b_model, b_tokenizer)
    except Exception as e:
        print(f"Back-translation failed for {source_code}: {e}")
        return text
