import os

import numpy as np # type: ignore
import pandas as pd

from dataset_loader import CrossLingualDataset
from trainers_collators_methods import (
    TN_PolarPairsTrainer,
    TN_PolarPairsCollator)
from metrics import subtask2_codabench_compute_metrics_multilabel

from transformers import ( # type: ignore
    Trainer, 
    AutoTokenizer,
    AutoModelForSequenceClassification
) # type: ignore

import torch # type: ignore
from torch.utils import DataLoader # type: ignore


TASKS_LABELS_NAMES = {
    'subtask1': 'polarization',
    'subtask2': ['gender/sexual','political','religious','racial/ethnic','other'],
    'subtask3': ['vilification','extreme_language','stereotype','invalidation','lack_of_empathy','dehumanization']
}


def subtask2_codabench_evaluation(model, tokenizer, language, training_args, dir, mode='cross-lingual', eval_mode='test'):
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     model_name,
    #     num_labels=5
    # )

    test = pd.read_csv(f'data/subtask2/{eval_mode}/{language}.csv')
    texts = test['text'].tolist()
    ids = test['id'].tolist()
    labels = np.zeros((len(texts), 5)).tolist()
    test[TASKS_LABELS_NAMES['subtask2']] = labels

    if mode == 'mono-lingual':
        pass
    elif mode == 'cross-lingual':
        dataset = CrossLingualDataset(test, tokenizer, subtask='subtask2', mode='eval')
        
        trainer = TN_PolarPairsTrainer(
            model = model,
            args = training_args,
            eval_dataset = dataset,
            compute_metrics = subtask2_codabench_compute_metrics_multilabel,
            data_collator = TN_PolarPairsCollator
        )

        predictions = trainer.predict(dataset)
        probs = torch.sigmoid(torch.from_numpy(predictions.predictions))
        preds = (probs > 0.2).int().numpy()

        results_df = pd.DataFrame({
            'id': ids,
            'gender/sexual': preds[:, 0],
            'political': preds[:, 1],
            'religious': preds[:, 2],
            'racial/ethnic': preds[:, 3],
            'other': preds[:, 4]
        })
        os.makedirs(f'{dir}/subtask_2', exist_ok=True)
        results_df.to_csv(f'{dir}/subtask_2/pred_{language}.csv', index=False)