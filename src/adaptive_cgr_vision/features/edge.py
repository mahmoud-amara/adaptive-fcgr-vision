"""Local contrast and edge statistics feature extraction for FCGR matrices.

Family D in Feature Registry:
- edge_magnitude_mean, edge_magnitude_std
- edge_density_t{t}
- local_var_mean_s{s}, local_var_std_s{s}
- contrast_range, contrast_iqr
"""

from typing import Dict, List, Optional
import numpy as np
from scipy.ndimage import sobel, uniform_filter


def extract_edge_features(
    matrix: np.ndarray,
    thresholds: Optional[List[float]] = None,
    windows: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Extract local contrast and edge statistics from a 2D FCGR matrix.

    Parameters
    ----------
    matrix : np.ndarray
        2D float FCGR matrix.
    thresholds : List[float], optional
        Relative gradient magnitude threshold fractions (default: [0.01, 0.05, 0.1]).
    windows : List[int], optional
        Local window sizes for variance filtering (default: [3, 5]).

    Returns
    -------
    Dict[str, float]
        Dictionary of edge and contrast features.
    """
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {matrix.shape}")

    if thresholds is None:
        thresholds = [0.01, 0.05, 0.1]
    if windows is None:
        windows = [3, 5]

    arr = np.asarray(matrix, dtype=np.float64)
    features: Dict[str, float] = {}

    if arr.size == 0 or np.all(np.isclose(arr, 0.0)) or np.isclose(np.min(arr), np.max(arr)):
        features["edge_magnitude_mean"] = 0.0
        features["edge_magnitude_std"] = 0.0
        for t in thresholds:
            t_str = str(t).replace(".", "")
            features[f"edge_density_t{t_str}"] = 0.0
        for w in windows:
            features[f"local_var_mean_s{w}"] = 0.0
            features[f"local_var_std_s{w}"] = 0.0
        features["contrast_range"] = 0.0
        features["contrast_iqr"] = 0.0
        return features

    # 1. Global Contrast
    c_min = float(np.min(arr))
    c_max = float(np.max(arr))
    features["contrast_range"] = c_max - c_min

    q75, q25 = np.percentile(arr, [75, 25])
    features["contrast_iqr"] = float(q75 - q25)

    # 2. Sobel Edge Gradient Magnitude
    # Sobel across row and column directions
    grad_r = sobel(arr, axis=0, mode="reflect")
    grad_c = sobel(arr, axis=1, mode="reflect")
    grad_mag = np.hypot(grad_r, grad_c)

    features["edge_magnitude_mean"] = float(np.mean(grad_mag))
    features["edge_magnitude_std"] = float(np.std(grad_mag))

    max_mag = float(np.max(grad_mag))
    for t in thresholds:
        t_str = str(t).replace(".", "")
        if max_mag > 1e-15:
            density = float(np.mean(grad_mag > (t * max_mag)))
        else:
            density = 0.0
        features[f"edge_density_t{t_str}"] = density

    # 3. Local Variance Statistics
    for w in windows:
        # Local variance = E[X^2] - (E[X])^2 over window w x w
        # Constrain window to not exceed matrix dimensions
        eff_w = min(w, min(arr.shape))
        if eff_w < 1:
            eff_w = 1
        mean_local = uniform_filter(arr, size=eff_w, mode="reflect")
        mean_sq_local = uniform_filter(arr ** 2, size=eff_w, mode="reflect")
        local_var = np.clip(mean_sq_local - (mean_local ** 2), 0.0, None)

        features[f"local_var_mean_s{w}"] = float(np.mean(local_var))
        features[f"local_var_std_s{w}"] = float(np.std(local_var))

    return features
