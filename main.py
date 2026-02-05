from utils.dataset_loader import (
    load_and_split_bilingual_data, 
    load_multilingual_data, 
    load_multilingual_data_strict, 
    PolarizationDataset, 
    CrossLingualDataset)
from utils.metrics import (
    subtask1_codabench_compute_metrics, 
    subtask2_codabench_compute_metrics_multilabel, 
    compute_metrics)
from utils.trainers_collators_methods import (
    TN_PolarPairsCollator, TN_PolarPairs, TN_PolarPairsTrainer)
from utils.augmentations import (
    aeda_5_line,
    augment_minority_classes, 
    wordswap_adjectives, 
    back_translate, 
    batch_backtranslate_minority_classes)
from utils.evaluations import (
    subtask2_codabench_evaluation
)

from utils.experiment_tracker import Experiment, Parameter

import yaml
import argparse

import numpy as np # type: ignore
import pandas as pd
import torch as torch # type: ignore
from transformers.training_args import TrainingArguments # type: ignore
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer # type: ignore
from transformers import DataCollatorWithPadding # type: ignore
from transformers import EarlyStoppingCallback # type: ignore

import os
from dotenv import load_dotenv # type: ignore
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

from huggingface_hub import login # type: ignore
if HF_TOKEN:
    login(token=HF_TOKEN) 
else:
    print("Error: HF_TOKEN not found in .env file")
    exit()

TASKS_METRIC = {
    'subtask1': subtask1_codabench_compute_metrics,
    'subtask2': subtask2_codabench_compute_metrics_multilabel,
    'subtask3': compute_metrics
}
TASKS_LABELS_NAMES = {
    'subtask1': 'polarization',
    'subtask2': ['gender/sexual','political','religious','racial/ethnic','other'],
    'subtask3': ['vilification','extreme_language','stereotype','invalidation','lack_of_empathy','dehumanization']
}

def parse_args():
    parser = argparse.ArgumentParser(description='PolarPairs Training')

    parser.add_argument('--mode', type=str, default='mono-lingual',
        choices=['mono-lingual', 'cross-lingual']
        )
    
    parser.add_argument('--task', type=str, default='subtask-1',
                        choices=['subtask1', 'subtask2', 'subtask3'])
    parser.add_argument('--language', type=str, default='eng', 
        choices=['eng', 'amh', 'swa', 'hau', 'all']
    )

    parser.add_argument('--data_augment', help='Augment training set using methods in config.yaml', action='store_true')

    parser.add_argument('--save_models', help='Save models to huggingFace repository', action='store_true')

    return parser.parse_args()


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    
    with open('./config.yaml', 'r') as f:
        configs = yaml.safe_load(f)
    
    training_params = configs['training']
    languages = configs['languages']
    n_labels = configs['n_labels']
    # experiment_params = configs['experiment']
    experiment = Experiment(**configs['experiment'])
    print(f"Experiment Directory: {experiment.dir}")

    args = parse_args()
    experiment.dict_2_params(training_params, 'Training')
    experiment.dict_2_params(configs['models'], 'Training')

    if args.language == 'all':
        languages = languages
    else:
        languages = [args.language]

    training_args = TrainingArguments(
        output_dir = './results',
        num_train_epochs = training_params['n_epochs'],
        per_device_train_batch_size = training_params['batch_size'],  # mT5-small can handle 2-4
        per_device_eval_batch_size = training_params['batch_size'],
        gradient_accumulation_steps = 1,
        eval_strategy = 'epoch',
        save_strategy = 'epoch',
        load_best_model_at_end = True,
        metric_for_best_model = 'eval_loss',
        greater_is_better = False,
        save_total_limit = 1,
        logging_steps = 50,
        learning_rate = training_params['lr'],
        max_grad_norm = 1.0,
        lr_scheduler_type = 'cosine',
        # warmup_ratio = 0.1,
        fp16 = True,
        # weight_decay = 0.1,
        dataloader_num_workers = 0,
        gradient_checkpointing = False,
        eval_accumulation_steps = 1,
        # metric_for_best_model="eval_loss",
        disable_tqdm=False
    )

    for language in languages:
        if args.mode == 'mono-lingual':

            # Load the tokenizer and model
            model_name = configs['models']['slave_model']
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels = n_labels[args.task],
                ignore_mismatched_sizes = True
            )
            # # Freeze all layers
            # for param in model.base_model.parameters():
            #     param.requires_grad = False

            # # Unfreeze last 2 encoder layers (for BERT-like models)
            # num_layers = model.config.num_hidden_layers
            # for i in range(num_layers - 4, num_layers):
            #     for param in model.base_model.encoder.layer[i].parameters():
            #         param.requires_grad = True

            # Create datasets
            train, val = load_and_split_bilingual_data( # Need to update this to make source language = all other languages asides target language 
                subtask = args.task,
                source_lang = language,
                target_lang = 'eng'
            )
            train_dataset = PolarizationDataset(train['source_text'].tolist(), train['polarization'].tolist(), tokenizer, n_classes=n_labels[args.task])
            val_dataset = PolarizationDataset(val['source_text'].tolist(), val['polarization'].tolist(), tokenizer, n_classes=n_labels[args.task])

            # Initialize the Trainer & Train the model
            trainer = Trainer(
                model=model,                         # the instantiated 🤗 Transformers model to be trained
                args=training_args,                  # training arguments, defined above
                train_dataset=train_dataset,         # training dataset
                eval_dataset=val_dataset,            # evaluation dataset
                compute_metrics=TASKS_METRIC[args.task],     # the callback that computes metrics of interest
                data_collator=DataCollatorWithPadding(tokenizer), # Data collator for dynamic padding
                # callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
            )
            trainer.train()

            # Evaluate Model
            eval_results = trainer.evaluate()
            print(f"Macro F1 score on {language} validation set: {eval_results['eval_f1_macro']}")

            # Save modelto hugging-face and  experiment details to experiment-tracker
            # if args.save_models:
            #     model.push_to_hub(f"olateju/PolarPairs-{model_name}_{language}")
            #     tokenizer.push_to_hub(f"olateju/PolarPairs-{model_name}_{language}")

            eval_results_param = Parameter(eval_results, f"{language}_eval_results", "Performance")
            experiment.add_params([eval_results_param])

        elif args.mode == 'cross-lingual':

            # Load the tokenizer and model
            student_model_name = configs['models']['slave_model']
            student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
            student_model = AutoModel.from_pretrained(student_model_name)

            # # Freeze n-layers of student model
            # for param in student_model.base_model.parameters():
            #     param.requires_grad = False
            # freeze_n_layers = configs['models']['freeze_slave_n_layers']
            # if freeze_n_layers > 0:
            #     num_layers = student_model.config.num_hidden_layers
            #     for i in range(num_layers - freeze_n_layers, num_layers):
            #         for param in student_model.base_model.encoder.layer[i].parameters():
            #             param.requires_grad = True

            teacher_model_name = configs['models']['anchor_model']
            teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
            teacher_model = AutoModel.from_pretrained(teacher_model_name,)

            # Freeze n-layers of teacher model
            freeze_n_layers = configs['models']['freeze_anchor_n_layers']
            if freeze_n_layers > 0:
              for param in teacher_model.base_model.parameters():
                  param.requires_grad = False

              num_layers = teacher_model.config.num_hidden_layers
              for i in range(num_layers - freeze_n_layers, num_layers):
                  for param in teacher_model.base_model.encoder.layer[i].parameters():
                      param.requires_grad = True

            model = TN_PolarPairs(
                student_model, teacher_model,
                num_labels=configs['n_labels'][args.task], 
                embedding_size=768
            )

            # Create datasets
            source_langs = configs['languages']
            source_langs.remove(language)
            print("Source Languages: ", source_langs)
            print("Languages: ", languages)
            train, val = load_multilingual_data(
                subtask=args.task, 
                languages=source_langs, 
                target_lang=language,
                target_train_ratio=0.7,
                mode='train', 
                verbose=False)
            
            if args.data_augment:
                augmentations = configs['augmentations']
                minority_classes = configs['minority_classes'][args.task]
                # task_classes = TASKS_LABELS_NAMES[args.task]

                bt_rows = 0
                if 'backtranslate' in augmentations:
                    print("Generating Back-Translation Augmentations")
                    bt_rows = batch_backtranslate_minority_classes(train, minority_classes, batch_size=64)
                    print("!!! Back-translatioan augmentations generated for minority classes !!!")

                augmentation_methods = []
                if 'aeda' in augmentations:
                    augmentation_methods.append(aeda_5_line)
                    print("Added AEDA Augmentations Technique")
                if 'wordswap' in augmentations:
                    augmentation_methods.append(wordswap_adjectives)
                    print("Added Wordswap Augmentation TEchnique")

                print("Augmenting Minortiy Classes In Train Set")
                augmented_rows = augment_minority_classes(train, minority_classes, methods=augmentation_methods, n_aug=4)    
                            
                train = pd.concat([train, pd.DataFrame(augmented_rows)], ignore_index=True)
                if 'backtranslate' in augmentations:
                    train = pd.concat([train, pd.DataFrame(bt_rows)], ignore_index=True)
                print("!!! Augmentation Done !!!")

            train_dataset = CrossLingualDataset(
              dataframe = train, 
              subtask = args.task,
              tokenizer = teacher_tokenizer,
              mode='train')
            val_dataset = CrossLingualDataset(
              dataframe = val, 
              subtask = args.task,
              tokenizer = teacher_tokenizer,
              mode='eval')
            

            # Initialize the Trainer & Train the model
            trainer = TN_PolarPairsTrainer(
                model=model,                         # the instantiated 🤗 Transformers model to be trained
                args=training_args,                  # training arguments, defined above
                train_dataset=train_dataset,         # training dataset
                eval_dataset=val_dataset,            # evaluation dataset
                compute_metrics=TASKS_METRIC[args.task],     # the callback that computes metrics of interest
                data_collator=TN_PolarPairsCollator(), # Data collator for dynamic padding
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
                **training_params['TNCSE'])
            trainer.train()

            # Evaluate the model on the validation set
            print("Training done, evaluating on validation set")
            eval_results = trainer.evaluate()
            print(f"Macro F1 score on validation set for Subtask 2: {eval_results['eval_f1_macro']}")

            # Save modelto hugging-face and  experiment details to experiment-tracker
            if args.save_models:
                print("Saving Models")
                trainer.save_model(f'olateju/PolarPairs-{args.task}.{experiment.version}.{language}.model')
                # model.push_to_hub(f'olateju/PolarPairs-{args.task}.{experiment.version}.{language}.model')
                # tokenizer.push_to_hub(f'olateju/PolarPairs-{args.task}.{experiment.version}.{language}.model') # type: ignore
                print("Models Saved")

            eval_results_param = Parameter(eval_results, f"{language}_eval_results", "Performance")
            experiment.add_params([eval_results_param])

            if args.task == 'subtask2':
                print("Generating submission for subtask2")
                subtask2_codabench_evaluation(
                    model, teacher_tokenizer, language, 
                    training_args, experiment.dir, 
                    args.mode, eval_mode='test'
                )
        
    experiment.save()


if __name__ == '__main__':
    main()