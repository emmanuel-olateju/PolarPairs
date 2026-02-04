import random
import pandas as pd

import nltk # type: ignore
# Download the necessary resources for WordNet and POS tagging
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')
import nlpaug.augmenter.word as naw # type: ignore

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM # type: ignore
import torch # type: ignore

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

# Load the "distilled" 600M version - it's fast and fits in Colab easily
MODEL_NAME = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to("cuda" if torch.cuda.is_available() else "cpu")

# NLLB uses specific language codes
NLLB_CODES = {
    'eng': 'eng_Latn',
    'swa': 'swh_Latn',
    'hau': 'hau_Latn',
    'amh': 'amh_Ethi'
}

def back_translate(text, source_code):
    src_lang = NLLB_CODES.get(source_code, 'eng_Latn')
    tgt_lang = 'eng_Latn' if src_lang != 'eng_Latn' else 'fra_Latn' # Bridge to French if original is English

    try:
        # Forward: Source -> English
        tokenizer.src_lang = src_lang
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        translated_tokens = model.generate(
            **inputs, 
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang), 
            max_length=128
        )
        intermediate_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

        # Backward: English -> Source
        tokenizer.src_lang = tgt_lang
        inputs = tokenizer(intermediate_text, return_tensors="pt").to(model.device)
        back_translated_tokens = model.generate(
            **inputs, 
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(src_lang), 
            max_length=128
        )
        return tokenizer.batch_decode(back_translated_tokens, skip_special_tokens=True)[0]
    except Exception as e:
        print(f"NLLB failed for {source_code}: {e}")
        return text
    
def batch_backtranslate_minority_classes(df, target_cols, batch_size=64):
    """
    Filters for minority classes and performs batch back-translation.
    Returns a list of new augmented rows.
    """
    # 1. Identify minority samples
    minority_mask = (df[target_cols] == 1).any(axis=1)
    minority_df = df[minority_mask].copy()
    
    if minority_df.empty:
        return []

    new_rows = []
    
    # 2. Process by language group (NLLB needs specific src_lang per batch)
    for lang_code, group in minority_df.groupby('language'):
        texts = group['text'].tolist()
        src_lang = NLLB_CODES.get(lang_code, 'eng_Latn')
        tgt_lang = 'eng_Latn' if src_lang != 'eng_Latn' else 'fra_Latn'
        
        print(f"--- Batch Translating {len(texts)} samples for language: {lang_code} ---")
        
        translated_results = []
        
        # 3. GPU Batch Loop
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Forward Batch (Source -> English/French)
            tokenizer.src_lang = src_lang
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
            
            with torch.no_grad():
                translated_tokens = model.generate(
                    **inputs, 
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                    max_length=128
                )
            intermediate_texts = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            
            # Backward Batch (English/French -> Source)
            tokenizer.src_lang = tgt_lang
            inputs = tokenizer(intermediate_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            
            with torch.no_grad():
                back_tokens = model.generate(
                    **inputs, 
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(src_lang),
                    max_length=128
                )
            translated_results.extend(tokenizer.batch_decode(back_tokens, skip_special_tokens=True))
        
        # 4. Create new rows with the augmented text
        for idx, (original_idx, row) in enumerate(group.iterrows()):
            aug_row = row.copy()
            aug_row['text'] = translated_results[idx]
            new_rows.append(aug_row)
            
    return new_rows
