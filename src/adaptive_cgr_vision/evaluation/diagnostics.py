"""Diagnostic analyses for classification errors and sequence similarity."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def compute_error_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sequence_lengths: Optional[np.ndarray] = None,
    ambiguity_fractions: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Analyze errors as a function of sequence length and ambiguity."""
    is_error = (y_true != y_pred).astype(int)
    results: Dict[str, Any] = {
        "total_samples": len(y_true),
        "total_errors": int(np.sum(is_error)),
        "error_rate": float(np.mean(is_error)),
    }

    if sequence_lengths is not None:
        results["mean_length_correct"] = float(np.mean(sequence_lengths[is_error == 0])) if np.sum(is_error == 0) > 0 else 0.0
        results["mean_length_error"] = float(np.mean(sequence_lengths[is_error == 1])) if np.sum(is_error == 1) > 0 else 0.0

    if ambiguity_fractions is not None:
        results["mean_ambig_correct"] = float(np.mean(ambiguity_fractions[is_error == 0])) if np.sum(is_error == 0) > 0 else 0.0
        results["mean_ambig_error"] = float(np.mean(ambiguity_fractions[is_error == 1])) if np.sum(is_error == 1) > 0 else 0.0

    return results
