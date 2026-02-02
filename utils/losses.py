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
    """
    FIXED: Added numerical stability and proper handling
    """
    
    # Handle multi-label vs single-label
    if polar_labels.dim() > 1 and polar_labels.size(1) > 1:
        target = polar_labels.float()
        
        # Calculate positive weights: (N_neg / N_pos) for each class
        # This balances the '0's and '1's within each of the 5 categories
        num_pos = target.sum(dim=0)
        num_neg = target.size(0) - num_pos
        pos_weight = (num_neg / (num_pos + 1e-6)) * 3.0
        
        # Clamp weights to prevent extreme gradients
        pos_weight = torch.clamp(pos_weight, max=10.0).to(logits.device)

        classification_loss = F.binary_cross_entropy_with_logits(logits, target)
        labels_equal = (polar_labels.unsqueeze(1) * polar_labels.unsqueeze(0)).sum(dim=-1) > 0
    else:
        if polar_labels.dim() > 1:
            polar_labels = polar_labels.squeeze(-1)
        
        num_classes = logits.size(1)
        class_counts = torch.bincount(polar_labels, minlength=num_classes)
        class_weights = 1.0 / (class_counts.float() + 1e-6)
        class_weights = class_weights / class_weights.sum()
        class_weights = class_weights.to(logits.device)
        classification_loss = F.cross_entropy(logits, polar_labels, weight=class_weights)
        labels_equal = (polar_labels.unsqueeze(1) == polar_labels.unsqueeze(0))
    
    # Check if we have any positive pairs
    n_positives = labels_equal.sum(dim=1).float()
    has_positives = (n_positives > 0).float()
    
    # If no positives in batch, skip contrastive losses
    if has_positives.sum() == 0:
        return classification_loss, torch.tensor(0.0, device=logits.device), torch.tensor(0.0, device=logits.device)
    
    mask = torch.eye(logits.size(0), device=logits.device).bool()
    
    # ============================================
    # ICNCE Loss (Interaction Constraint InfoNCE)
    # ============================================
    x1_cls_norm = F.normalize(x1_cls, p=2, dim=1)
    x2_cls_norm = F.normalize(x2_cls, p=2, dim=1)
    
    # Compute similarity
    cls_similarity = torch.mm(x1_cls_norm, x2_cls_norm.t()) / tau
    
    # Clamp for numerical stability BEFORE exp
    cls_similarity = torch.clamp(cls_similarity, min=-10, max=10)
    
    cls_similarity = torch.exp(cls_similarity)
    cls_similarity = cls_similarity.masked_fill(mask, 0)
    
    # Positive and all similarities
    pos = (cls_similarity * labels_equal.float()).sum(dim=1)
    all_sim = cls_similarity.sum(dim=1)
    
    # Add stronger epsilon and clamp
    eps = 1e-6
    pos = torch.clamp(pos, min=eps)
    all_sim = torch.clamp(all_sim, min=eps)
    
    # InfoNCE loss with clamping
    ratio = pos / all_sim
    ratio = torch.clamp(ratio, min=eps, max=1.0)
    icnce_loss = -torch.log(ratio)
    
    # Only average over samples with positives
    icnce_loss = (icnce_loss * has_positives).sum() / (has_positives.sum() + eps)
    
    # ============================================
    # TNC Loss (Tensor Norm Constraint)
    # ============================================
    # Compute differences
    diff = x1_pool.unsqueeze(1) - x2_pool.unsqueeze(0)  # [batch, batch, dim]
    diff_norms = torch.norm(diff, p=2, dim=2)  # [batch, batch]
    
    # Compute sum of norms
    x1_pool_norm = torch.norm(x1_pool, p=2, dim=1)  # [batch]
    x2_pool_norm = torch.norm(x2_pool, p=2, dim=1)  # [batch]
    sum_norms = x1_pool_norm.unsqueeze(1) + x2_pool_norm.unsqueeze(0)  # [batch, batch]
    
    # Avoid division by zero with stronger epsilon
    eps = 1e-4
    sum_norms = torch.clamp(sum_norms, min=eps)
    
    # Tensor norm loss
    tensor_norm_loss = diff_norms / sum_norms  # [batch, batch]
    
    # Only consider positive pairs
    tensor_norm_loss = tensor_norm_loss * labels_equal.float()
    
    # Weighted by similarity (with clamping)
    cls_similarity_clamped = torch.clamp(cls_similarity, min=eps, max=1e6)
    weighted_tnc = (cls_similarity_clamped + eps) * tensor_norm_loss
    
    # Average per sample (only over samples with positives)
    tnc_per_sample = weighted_tnc.sum(dim=1) / (n_positives + eps)
    tnce_loss = (tnc_per_sample * has_positives).sum() / (has_positives.sum() + eps)
    
    # Final clamping to prevent explosion
    icnce_loss = torch.clamp(icnce_loss, max=10.0)
    tnce_loss = torch.clamp(tnce_loss, max=10.0)
    
    return classification_loss, tnce_loss, icnce_loss