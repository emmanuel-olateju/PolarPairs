import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore

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
        class_counts = torch.bincount(polar_labels.long(), minlength=num_classes)
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

def tnc_contrastive_loss(logits, x1_cls, x2_cls, x1_pool, x2_pool, polar_labels, tau):
    
    if polar_labels.dim() > 1 and polar_labels.size(1) > 1:
        # Multi-label classification
        classification_loss = F.binary_cross_entropy_with_logits(logits, polar_labels.float())
    else:
        # Single-label classification
        if polar_labels.dim() > 1:
            polar_labels = polar_labels.squeeze(-1)
        
        num_classes = logits.size(1)
        class_counts = torch.bincount(polar_labels, minlength=num_classes)
        class_weights = 1.0 / (class_counts.float() + 1e-6)
        class_weights = class_weights / class_weights.sum()
        class_weights = class_weights.to(logits.device)
        classification_loss = F.cross_entropy(logits, polar_labels, weight=class_weights)
    
    # For contrastive loss, we need to compare if ANY label matches
    # labels_equal[i,j] = True if samples i and j share ANY label
    labels_equal = (polar_labels.unsqueeze(1) * polar_labels.unsqueeze(0)).sum(dim=-1) > 0
    
    mask = torch.eye(logits.size(0), device=logits.device).bool()
    
    # Interaction Constraint InfoNCE Loss
    x1_cls_norm = F.normalize(x1_cls, p=2, dim=1)
    x2_cls_norm = F.normalize(x2_cls, p=2, dim=1)
    cls_similarity = torch.mm(x1_cls_norm, x2_cls_norm.t()) / tau
    cls_similarity = torch.exp(cls_similarity)
    cls_similarity = cls_similarity.masked_fill(mask, 0)
    pos = (cls_similarity * labels_equal.float()).sum(dim=1)
    all = cls_similarity.sum(dim=1)
    icnce_loss = -torch.log((pos + 1e-8) / (all + 1e-8))
    icnce_loss = icnce_loss.mean()
    
    # tnce Loss
    diff = x1_pool.unsqueeze(1) - x2_pool.unsqueeze(0)
    diff_norms = F.normalize(diff, p=2, dim=2)
    x1_pool_norm = F.normalize(x1_pool, p=2, dim=1)
    x2_pool_norm = F.normalize(x2_pool, p=2, dim=1)
    sum_norms = x1_pool_norm.unsqueeze(1) + x2_pool_norm.unsqueeze(0)
    tensor_norm_loss = diff_norms / (sum_norms + 1e-8)
    tensor_norm_loss = tensor_norm_loss.sum(dim=-1)  # Sum over embedding dim
    tensor_norm_loss = tensor_norm_loss * labels_equal.float()
    n_positives = labels_equal.sum(dim=1).float()
    tnc_loss = -torch.log(cls_similarity + 1e-8) * tensor_norm_loss
    tnc_loss_per_sample = tnc_loss.sum(dim=1) / (n_positives + 1e-8)
    tnce_loss = tnc_loss_per_sample.mean()
    
    return classification_loss, tnce_loss, icnce_loss