import numpy as np
from sklearn.metrics import f1_score
import torch

def subtask1_codabench_compute_metrics(p):
  preds = np.argmax(p.predictions, axis=1)
  return {'f1_macro': f1_score(p.label_ids, preds, average='macro')}

def subtask2_codabench_compute_metrics_multilabel(p):
  # Sigmoid the predictions to get probabilities
  probs = torch.sigmoid(torch.from_numpy(p.predictions))
  # Convert probabilities to predicted labels (0 or 1)
  preds = (probs > 0.2).int().numpy()
  # Compute macro F1 score
  return {'f1_macro': f1_score(p.label_ids, preds, average='macro')}

def compute_metrics(p):
    """Simple, robust compute_metrics function"""

    # Predictions should be (n_samples, n_classes)
    logits = p.predictions
    preds = np.argmax(logits, axis=1)

    # Labels should be (n_samples,)
    labels = p.label_ids

    # Ensure labels are 1D
    if isinstance(labels, (list, tuple)):
        labels = np.array(labels)

    if labels.ndim > 1:
        labels = labels.flatten()

    # Calculate F1
    return {
        'f1_macro': f1_score(labels, preds, average='macro'),
        'f1_weighted': f1_score(labels, preds, average='weighted')
    }