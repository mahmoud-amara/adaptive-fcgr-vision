"""Comprehensive evaluation metrics for genomic sequence classification.

Metrics:
- Macro-F1 (primary metric)
- Multiclass Accuracy
- Balanced Accuracy
- Expected Calibration Error (ECE)
- Multiclass Brier Score
- Inference Latency and Throughput
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import time
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
)


def compute_multiclass_ece(
    y_true_indices: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute multiclass Expected Calibration Error (ECE).

    Parameters
    ----------
    y_true_indices : np.ndarray
        Ground truth integer class indices, shape (N,).
    y_prob : np.ndarray
        Predicted probabilities, shape (N, C).
    n_bins : int
        Number of confidence bins.

    Returns
    -------
    float
        ECE score in [0, 1].
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true_indices).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(confidences)

    if n_total == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_multiclass_brier(
    y_true_indices: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int,
) -> float:
    """Compute multiclass Brier score: mean squared difference from one-hot truth."""
    N = len(y_true_indices)
    if N == 0:
        return 0.0
    one_hot = np.zeros((N, n_classes), dtype=float)
    one_hot[np.arange(N), y_true_indices] = 1.0
    brier = np.mean(np.sum((y_prob - one_hot) ** 2, axis=1))
    return float(brier)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute standard research evaluation metrics suite.

    Parameters
    ----------
    y_true : np.ndarray
        Array of ground truth class labels (strings or ints).
    y_pred : np.ndarray
        Array of predicted class labels.
    y_prob : Optional[np.ndarray]
        Array of predicted class probability distributions, shape (N, C).
    classes : Optional[List[str]]
        List of unique class names corresponding to columns of y_prob.

    Returns
    -------
    Dict[str, Any]
        Dictionary of computed metric values.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if classes is None:
        unique_classes = sorted(list(np.unique(np.concatenate([y_true_arr, y_pred_arr]))))
    else:
        unique_classes = list(classes)

    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    y_true_idx = np.array([class_to_idx[c] for c in y_true_arr])
    y_pred_idx = np.array([class_to_idx[c] for c in y_pred_arr])

    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
    accuracy = float(accuracy_score(y_true_arr, y_pred_arr))
    balanced_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=unique_classes)

    metrics = {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "num_samples": len(y_true_arr),
        "classes": unique_classes,
        "confusion_matrix": cm.tolist(),
    }

    if y_prob is not None:
        ece = compute_multiclass_ece(y_true_idx, y_prob)
        brier = compute_multiclass_brier(y_true_idx, y_prob, len(unique_classes))
        metrics["ece"] = ece
        metrics["brier_score"] = brier

    return metrics
