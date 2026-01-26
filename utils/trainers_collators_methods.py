from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn as nn

from transformers import AutoModel, Trainer
from transformers import DataCollatorWithPadding

from  losses import polar_pairs_contrastive_loss

class CustomDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        # Convert labels to long (int64)
        if 'labels' in batch:
            batch['labels'] = batch['labels'].long()
        return batch

class MultiLingualPolarPairsAlignment(nn.Module):

    def __init__(self, encoder_model_name, pretrained_encoder_name, num_labels, alignment_normalization=True, alignment_residual=True):
        super(MultiLingualPolarPairsAlignment, self).__init__()

        self.encoder = AutoModel.from_pretrained(encoder_model_name)
        self.encoder.train()
        self.pretrained_encoder = AutoModel.from_pretrained(pretrained_encoder_name)
        self.pretrained_encoder.eval()

        encoder_config = self.encoder.config
        embedding_size = encoder_config.hidden_size

        self.alignment_head = nn.Linear(embedding_size, embedding_size)
        self.classification_head = nn.Linear(embedding_size, num_labels)
        self.alignment_layer_norm = nn.LayerNorm(embedding_size)
        self.alignment_normalization = alignment_normalization
        self.alignment_residual = True

    def get_cls_embeddings(self, x_input_ids, x_attention_mask, polar_labels):
        x1_hidden = self.encoder(
            input_ids=x_input_ids,
            attention_mask=x_attention_mask
        ).last_hidden_state
        return x1_hidden[:, 0, :]

    def cls_resnet(self, x):
        return self.alignment_head(x) + x

    def cls_normalization(self, x):
        return self.alignment_layer_norm(x)

    def get_aligned_embedding(self, x_input_ids, x_attention_mask, polar_labels):
        x_cls = self.get_cls_embeddings(x_input_ids, x_attention_mask, polar_labels)

        if self.alignment_residual:
            x_cls = self.cls_resnet(x_cls)
        else:
            x_cls = self.alignment_head(x_cls)

        if self.alignment_normalization:
            x_cls = self.cls_normalization(x_cls)

        return x_cls

    def forward(self, x_input_ids, x_attention_mask, x_hat_input_ids, x_hat_attention_mask, polar_labels, hat_labels):
        # Get Aligned Embedding for source language
        x1_aligned = self.get_aligned_embedding(x_input_ids, x_attention_mask, polar_labels)

        # Flatten targets before encoding
        batch_size, subset_size, seq_len = x_hat_input_ids.shape
        x_hat_input_ids_flat = x_hat_input_ids.view(batch_size * subset_size, seq_len)
        x_hat_attention_mask_flat = x_hat_attention_mask.view(batch_size * subset_size, seq_len)
        x2_hidden_flat = self.pretrained_encoder(
            input_ids=x_hat_input_ids_flat,
            attention_mask=x_hat_attention_mask_flat
        ).last_hidden_state  # ✓ Shape: (batch_size * subset_size, seq_len, hidden_size)

        # Reshape back correctly
        hidden_size = x2_hidden_flat.size(-1)
        x2_hidden = x2_hidden_flat.view(batch_size, subset_size, seq_len, hidden_size)
        # Extract CLS token
        x2_cls = x2_hidden[:, :, 0, :]  # (batch_size, subset_size, hidden_size)

        # # Apply alignment to all target embeddings
        # batch_size, subset_size, hidden_size = x2_cls.shape
        # x2_cls_flat = x2_cls.view(batch_size * subset_size, hidden_size)
        # x2_aligned_flat = F.leaky_relu(self.alignment_head(x2_cls_flat)) + x2_cls_flat
        # if self.alignment_normalization:
        #   x2_aligned_flat = self.alignment_layer_norm(x2_cls_flat + x2_aligned_flat)
        # x2_aligned = x2_aligned_flat.view(batch_size, subset_size, hidden_size)

        # Use target embeddings directly (no transformation)
        x2_aligned = x2_cls  # (batch_size, subset_size, hidden_size)

        # Classification logits (only from source)
        logits = self.classification_head(x1_aligned)

        return logits, x1_aligned, x2_aligned

    def compute_embeddings(self, x_input_ids, x_attention_mask, polar_labels):
        x_embd = self.get_aligned_embedding(x_input_ids, x_attention_mask, polar_labels)
        return {
            'embeddings': x_embd,
            'labels': polar_labels
        }

    def compute_logits(self, x_input_ids, x_attention_mask, polar_labels):
        embeddings = self.compute_embeddings(x_input_ids, x_attention_mask, polar_labels)
        logits = self.compute_embeddings(embeddings['embeddings'])
        return {
            'logits': logits,
            'labels': polar_labels
        }


class PolarPairsTrainer(Trainer):
    def __init__(self, lambda_align=0.5, temperature=0.07, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_align = lambda_align
        self.temperature = temperature
        self.label_names = ['polar_labels', 'hat_labels']

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        logits, x1_aligned, x2_aligned = model(
            x_input_ids=inputs['x_input_ids'],
            x_attention_mask=inputs['x_attention_mask'],
            x_hat_input_ids=inputs['x_hat_input_ids'],
            x_hat_attention_mask=inputs['x_hat_attention_mask'],
            polar_labels=inputs['polar_labels'],
            hat_labels=inputs['hat_labels']
        )

        total_loss, classification_loss, alignment_loss = polar_pairs_contrastive_loss(
            logits=logits,
            x1_aligned=x1_aligned,
            x2_aligned=x2_aligned,
            polar_labels=inputs['polar_labels'],
            hat_labels=inputs['hat_labels'],
            lambda_align=self.lambda_align,
            temperature=self.temperature
        )

        # Log component losses
        if self.state.global_step % 10 == 0:
            self.log({
                'classification_loss': classification_loss.item(),
                'alignment_loss': alignment_loss.item(),
                'lambda': self.lambda_align
            })

        if return_outputs:
            outputs = {'logits': logits}
            return total_loss, outputs
        else:
            return total_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            logits = outputs['logits']

        labels = inputs['polar_labels'].detach().cpu()

        if prediction_loss_only:
            return (loss.detach(), None, None)

        return (loss.detach(), logits.detach().cpu(), labels)

@dataclass
class PolarPairsCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}

        batch['x_input_ids'] = torch.stack([f['x_input_ids'] for f in features])
        batch['x_attention_mask'] = torch.stack([f['x_attention_mask'] for f in features])
        batch['x_hat_input_ids'] = torch.stack([f['x_hat_input_ids'] for f in features])
        batch['x_hat_attention_mask'] = torch.stack([f['x_hat_attention_mask'] for f in features])

        polar_labels = []
        for f in features:
            label = f['polar_labels']
            if isinstance(label, torch.Tensor):
                polar_labels.append(label.item() if label.dim() == 0 else label[0].item())
            else:
                polar_labels.append(int(label))

        batch['polar_labels'] = torch.tensor(polar_labels, dtype=torch.long)
        batch['hat_labels'] = torch.stack([f['hat_labels'] for f in features])

        return batch