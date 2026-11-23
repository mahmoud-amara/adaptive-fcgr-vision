"""Evaluation, statistical inference, and diagnostics module."""

from adaptive_cgr_vision.evaluation.metrics import (
    evaluate_predictions,
    compute_multiclass_ece,
    compute_multiclass_brier,
)
from adaptive_cgr_vision.evaluation.uncertainty import (
    bootstrap_metric_ci,
    paired_difference_test,
)
from adaptive_cgr_vision.evaluation.diagnostics import compute_error_breakdown

__all__ = [
    "evaluate_predictions",
    "compute_multiclass_ece",
    "compute_multiclass_brier",
    "bootstrap_metric_ci",
    "paired_difference_test",
    "compute_error_breakdown",
]
