import torch
import torch.nn as nn
import torch.nn.functional as F

def polar_pairs_contrastive_loss(
                logits, x1_aligned, x2_aligned, polar_labels, 
                hat_labels, lambda_align: float = 0.5, 
                temperature: float = 0.07 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Improved loss with:
        1. Classification loss
        2. Contrastive alignment loss (pull same class together, push different apart)
        3. Class balance handling
        """

        # 1. Classification loss with class weights
        num_classes = logits.size(1)  # Get number of classes from logits
        class_counts = torch.bincount(polar_labels, minlength=num_classes)
        class_weights = 1.0 / (class_counts.float() + 1e-6)
        class_weights = class_weights / class_weights.sum()

        # Move weights to same device as logits
        class_weights = class_weights.to(logits.device)

        classification_loss = F.cross_entropy(logits, polar_labels, weight=class_weights)

        # 2. Contrastive alignment loss
        batch_size, subset_size, hidden_size = x2_aligned.shape

        # Expand source embeddings to match all targets
        x1_expanded = x1_aligned.unsqueeze(1).expand(-1, subset_size, -1)
        x1_flat = x1_expanded.reshape(batch_size * subset_size, hidden_size)
        x2_flat = x2_aligned.reshape(batch_size * subset_size, hidden_size)

        # Expand source labels to match
        source_labels_expanded = polar_labels.unsqueeze(1).expand(-1, subset_size)
        source_labels_flat = source_labels_expanded.reshape(-1)
        target_labels_flat = hat_labels.reshape(-1)

        # Compute similarity matrix
        x1_norm = F.normalize(x1_flat, p=2, dim=1)
        x2_norm = F.normalize(x2_flat, p=2, dim=1)
        similarity = torch.mm(x1_norm, x2_norm.t()) / temperature

        # Create masks for positive and negative pairs
        labels_equal = (source_labels_flat.unsqueeze(1) == target_labels_flat.unsqueeze(0))

        # InfoNCE-style contrastive loss
        exp_sim = torch.exp(similarity)

        # Mask out self-similarities (diagonal)
        mask_self = torch.eye(similarity.size(0), device=similarity.device).bool()
        exp_sim = exp_sim.masked_fill(mask_self, 0)

        # Positive pairs: same label
        pos_sim = (exp_sim * labels_equal.float()).sum(dim=1)

        # All pairs (excluding self)
        all_sim = exp_sim.sum(dim=1)

        # Contrastive loss: -log(pos / (pos + neg))
        alignment_loss = -torch.log((pos_sim + 1e-8) / (all_sim + 1e-8))
        alignment_loss = alignment_loss.mean()

        # 3. Total loss
        total_loss = (1.0 - lambda_align) * classification_loss + lambda_align * alignment_loss

        return total_loss, classification_loss, alignment_loss