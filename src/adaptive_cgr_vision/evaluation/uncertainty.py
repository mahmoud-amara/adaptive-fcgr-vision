"""Bootstrap confidence intervals and statistical comparison tests.

Implements non-parametric bootstrap resampling for confidence intervals (95% CI)
and paired permutation tests for comparing model differences.
"""

from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import f1_score


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """Compute non-parametric bootstrap confidence interval for a metric.

    Returns
    -------
    Tuple[float, float, float]
        (point_estimate, ci_lower, ci_upper)
    """
    if metric_fn is None:
        def metric_fn(yt, yp):
            return float(f1_score(yt, yp, average="macro", zero_division=0))

    rng = np.random.default_rng(random_state)
    n = len(y_true)
    point_est = metric_fn(y_true, y_pred)

    boot_scores = []
    for _ in range(n_bootstraps):
        boot_idx = rng.choice(n, size=n, replace=True)
        score = metric_fn(y_true[boot_idx], y_pred[boot_idx])
        boot_scores.append(score)

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_scores, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_scores, 100 * (1.0 - alpha / 2.0)))

    return point_est, ci_lower, ci_upper


def paired_difference_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> Dict[str, float]:
    """Compute paired permutation test for difference in macro-F1 between two models.

    H0: Model A and Model B have identical performance.
    """
    rng = np.random.default_rng(random_state)
    f1_a = float(f1_score(y_true, y_pred_a, average="macro", zero_division=0))
    f1_b = float(f1_score(y_true, y_pred_b, average="macro", zero_division=0))
    observed_diff = f1_a - f1_b

    n = len(y_true)
    count_more_extreme = 0

    for _ in range(n_permutations):
        # Swap predictions with probability 0.5
        swap_mask = rng.random(n) < 0.5
        perm_a = np.where(swap_mask, y_pred_b, y_pred_a)
        perm_b = np.where(swap_mask, y_pred_a, y_pred_b)

        perm_f1_a = f1_score(y_true, perm_a, average="macro", zero_division=0)
        perm_f1_b = f1_score(y_true, perm_b, average="macro", zero_division=0)
        perm_diff = perm_f1_a - perm_f1_b

        if np.abs(perm_diff) >= np.abs(observed_diff):
            count_more_extreme += 1

    p_val = count_more_extreme / n_permutations

    return {
        "f1_model_a": f1_a,
        "f1_model_b": f1_b,
        "observed_diff": observed_diff,
        "p_value": float(p_val),
    }
