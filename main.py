from utils.dataset_loader import load_and_split_bilingual_data, PolarizationDataset
from utils.metrics import subtask1_codabench_compute_metrics

from utils.experiment_tracker import Experiment, Parameter

import argparse

import numpy as np
import torch as torch
from transformers.training_args import TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from transformers import DataCollatorWithPadding

def parse_args():
    parser = argparse.ArgumentParser(description='PolarPairs Training')

    parser.add_argument('--mode', type=str, default='mono-lingual',
        choices=['mono-lingual', 'cross-lingual']
        )
    
    parser.add_argument('--task', type=str, default='subtask-1',
                        choices=['subtask-1', 'subtask-2', 'subtask-3', 'combined'])
    parser.add_argument('--language', type=str, default='eng', 
        choices=['eng', 'amh', 'swa', 'hau', 'all']
    )
    
    parser.add_argument('--lr', type=float, default=2E-6)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_epochs', type=int, default=3)

    parser.add_argument('--experiment_version', type=str, default='v0.0')
    parser.add_argument('--experiment_baseline', type=str, default='None')
    parser.add_argument('--experiment_description', type=str, default='Baseline')
    parser.add_argument('--experiment_dir', type=str, default='./experiments/')

    parser.add_argument('--teacher_model', type=str, default='microsoft/deberta-v3-small')
    parser.add_argument('--student_model', type=str, default='microsoft/deberta-v3-small')

    parser.add_argument('--save_models', help='Save models to huggingFace repository', action='store_true')

    return parser.parse_args()


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    args = parse_args()
    experiment = Experiment(
        args.experiment_version,
        args.experiment_dir,
        args.experiment_description,
        args.experiment_baseline
    )

    if args.task == 'subtask-1':
        if args.language == 'all':
            languages = ['eng', 'amh', 'swa', 'hau']
        else:
            languages = [args.language]

        if args.mode == 'mono-lingual':

            training_args = TrainingArguments(
                output_dir = './results',
                num_train_epochs = args.n_epochs,
                per_device_train_batch_size = args.batch_size,  # mT5-small can handle 2-4
                per_device_eval_batch_size = args.batch_size,
                gradient_accumulation_steps = 8,
                eval_strategy = 'epoch',
                save_strategy = 'no',
                logging_steps = 10,
                learning_rate = args.lr,
                max_grad_norm = 1.0,
                lr_scheduler_type = 'linear',
                warmup_ratio = 0.1,
                fp16 = False,
                weight_decay = 0.1,
                # dataloader_num_workers = 0,
                # load_best_model_at_end = False,
                # eval_accumulation_steps = 1,
                # gradient_checkpointing = False,
                # metric_for_best_model="eval_loss"
            )

        for language in languages:

            # Load the tokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.student_model, use_fast=True)

            train, val = load_and_split_bilingual_data(
                subtask = 'subtask1',
                source_lang = language,
                target_lang = 'eng'
            )

            # Create datasets
            train_dataset = PolarizationDataset(train['source_text'].tolist(), train['polarization'].tolist(), tokenizer, n_classes=2)
            val_dataset = PolarizationDataset(val['source_text'].tolist(), val['polarization'].tolist(), tokenizer, n_classes=2)

            # Load the model
            model = AutoModelForSequenceClassification.from_pretrained(
                args.student_model,
                num_labels = 2,
                ignore_mismatched_sizes = True
            )

            # Initialize the Trainer
            trainer = Trainer(
                model=model,                         # the instantiated 🤗 Transformers model to be trained
                args=training_args,                  # training arguments, defined above
                train_dataset=train_dataset,         # training dataset
                eval_dataset=val_dataset,            # evaluation dataset
                compute_metrics=subtask1_codabench_compute_metrics,     # the callback that computes metrics of interest
                data_collator=DataCollatorWithPadding(tokenizer), # Data collator for dynamic padding
                # callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
            )

            # Train the model
            trainer.train()

            eval_results = trainer.evaluate()
            print(f"Macro F1 score on {language} validation set: {eval_results['eval_f1_macro']}")

            if args.save_models:
                model.push_to_hub(f"olateju/PolarPairs-{args.experiment_version}_{language}")
                tokenizer.push_to_hub(f"olateju/PolarPairs-{args.experiment_version}_{language}")

            # ===== SAVE THE FINE-TUNED MODEL =====
            # save_path = f'finetuned_models/{language}_{tokenizer_param.get_value()}'
            # model.save_pretrained(save_path)
            # tokenizer.save_pretrained(save_path)
            # print(f"Saved {language} model to {save_path}")
            # ====================================

            eval_results_param = Parameter(eval_results, f"{language}_eval_results", "Performance")
            experiment.add_params([eval_results_param])




if __name__ == '__main__':
    main()