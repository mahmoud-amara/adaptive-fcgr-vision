"""Spatial distribution and moments feature extraction for FCGR matrices.

Family B in Feature Registry:
- spatial_centroid_row, spatial_centroid_col
- spatial_var_row, spatial_var_col, spatial_covar
- spatial_mass_q00, spatial_mass_q01, spatial_mass_q10, spatial_mass_q11
- spatial_entropy
- spatial_row_marginal_std, spatial_col_marginal_std
"""

from typing import Dict
import numpy as np


def extract_spatial_features(matrix: np.ndarray) -> Dict[str, float]:
    """Extract spatial distribution and moment features from a 2D FCGR matrix.

    Parameters
    ----------
    matrix : np.ndarray
        2D float FCGR matrix (non-negative).

    Returns
    -------
    Dict[str, float]
        Dictionary of spatial features.
    """
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {matrix.shape}")

    arr = np.asarray(matrix, dtype=np.float64)
    arr = np.clip(arr, 0.0, None)  # Ensure non-negative

    total_mass = float(np.sum(arr))

    # Handle zero/empty mass safely
    if total_mass <= 0.0 or not np.isfinite(total_mass):
        return {
            "spatial_centroid_row": 0.0,
            "spatial_centroid_col": 0.0,
            "spatial_var_row": 0.0,
            "spatial_var_col": 0.0,
            "spatial_covar": 0.0,
            "spatial_mass_q00": 0.0,
            "spatial_mass_q01": 0.0,
            "spatial_mass_q10": 0.0,
            "spatial_mass_q11": 0.0,
            "spatial_entropy": 0.0,
            "spatial_row_marginal_std": 0.0,
            "spatial_col_marginal_std": 0.0,
        }

    nrows, ncols = arr.shape
    r_coords, c_coords = np.mgrid[0:nrows, 0:ncols]

    # Probability distribution
    p = arr / total_mass

    # Centroids (means)
    mu_r = float(np.sum(r_coords * p))
    mu_c = float(np.sum(c_coords * p))

    # Variances and Covariance
    var_r = float(np.sum(((r_coords - mu_r) ** 2) * p))
    var_c = float(np.sum(((c_coords - mu_c) ** 2) * p))
    covar = float(np.sum((r_coords - mu_r) * (c_coords - mu_c) * p))

    # Quadrant masses (split evenly along row and column midpoints)
    mid_r = nrows // 2
    mid_c = ncols // 2

    q00 = float(np.sum(arr[:mid_r, :mid_c]) / total_mass)
    q01 = float(np.sum(arr[:mid_r, mid_c:]) / total_mass)
    q10 = float(np.sum(arr[mid_r:, :mid_c]) / total_mass)
    q11 = float(np.sum(arr[mid_r:, mid_c:]) / total_mass)

    # Spatial 2D entropy: -sum(p * log2(p)) where p > 0
    mask = p > 1e-15
    entropy = -float(np.sum(p[mask] * np.log2(p[mask])))

    # Marginals and their standard deviations
    row_marginals = np.sum(p, axis=1)  # sum over columns
    col_marginals = np.sum(p, axis=0)  # sum over rows
    row_std = float(np.std(row_marginals))
    col_std = float(np.std(col_marginals))

    return {
        "spatial_centroid_row": mu_r,
        "spatial_centroid_col": mu_c,
        "spatial_var_row": var_r,
        "spatial_var_col": var_c,
        "spatial_covar": covar,
        "spatial_mass_q00": q00,
        "spatial_mass_q01": q01,
        "spatial_mass_q10": q10,
        "spatial_mass_q11": q11,
        "spatial_entropy": entropy,
        "spatial_row_marginal_std": row_std,
        "spatial_col_marginal_std": col_std,
    }
