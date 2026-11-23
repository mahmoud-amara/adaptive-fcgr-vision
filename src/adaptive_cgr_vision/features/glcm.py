"""Gray-Level Co-occurrence Matrix (GLCM) / Haralick texture feature extraction.

Family A in Feature Registry:
- contrast
- dissimilarity
- homogeneity
- energy (ASM)
- correlation
- entropy
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def quantize_matrix(
    matrix: np.ndarray,
    n_levels: int = 32,
    apply_log1p: bool = True,
) -> np.ndarray:
    """Quantize a 2D continuous matrix to discrete gray levels in [0, n_levels - 1].

    Parameters
    ----------
    matrix : np.ndarray
        2D float array.
    n_levels : int
        Number of quantization bins (default: 32).
    apply_log1p : bool
        Whether to apply np.log1p before linear quantization.

    Returns
    -------
    np.ndarray
        2D uint8 array with values in [0, n_levels - 1].
    """
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {matrix.shape}")

    arr = np.asarray(matrix, dtype=np.float64)

    if apply_log1p:
        arr = np.log1p(np.clip(arr, 0.0, None))

    min_val = np.min(arr)
    max_val = np.max(arr)

    # Constant matrix check
    if np.isclose(min_val, max_val):
        return np.zeros(arr.shape, dtype=np.uint8)

    scaled = (arr - min_val) / (max_val - min_val)
    quantized = np.clip(np.floor(scaled * n_levels), 0, n_levels - 1).astype(np.uint8)
    return quantized


def compute_glcm_entropy(P: np.ndarray) -> np.ndarray:
    """Compute GLCM entropy across 4D GLCM tensor [levels, levels, distances, angles].

    Formula: -sum(P(i, j) * log2(P(i, j))) where 0 * log(0) = 0.

    Returns
    -------
    np.ndarray of shape (num_distances, num_angles)
    """
    # P shape: (n_levels, n_levels, n_distances, n_angles)
    eps = 1e-15
    # Mask out zeros to avoid log2(0) warnings
    mask = P > eps
    log_p = np.zeros_like(P)
    log_p[mask] = np.log2(P[mask])
    # Entropy summed over first two axes (i, j)
    entropy = -np.sum(P * log_p, axis=(0, 1))
    return entropy


def extract_glcm_features(
    matrix: np.ndarray,
    distances: Optional[List[int]] = None,
    angles: Optional[List[float]] = None,
    n_levels: int = 32,
    apply_log1p: bool = True,
    symmetric: bool = True,
    normed: bool = True,
    aggregate_angles: bool = True,
) -> Dict[str, float]:
    """Extract GLCM texture features from a 2D FCGR matrix.

    Parameters
    ----------
    matrix : np.ndarray
        2D float FCGR matrix.
    distances : List[int], optional
        Pixel offset distances (default: [1, 2, 3]).
    angles : List[float], optional
        Pixel offset angles in radians (default: [0, pi/4, pi/2, 3pi/4]).
    n_levels : int
        Number of gray levels for quantization (default: 32).
    apply_log1p : bool
        Whether to log-transform before quantization (default: True).
    symmetric : bool
        Whether to accumulate co-occurrences symmetrically (default: True).
    normed : bool
        Whether to normalize GLCM probabilities to sum to 1 (default: True).
    aggregate_angles : bool
        Whether to take the mean across angles for each distance (default: True).

    Returns
    -------
    Dict[str, float]
        Dictionary mapping feature names to extracted float values.
    """
    if distances is None:
        distances = [1, 2, 3]
    if angles is None:
        angles = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    angle_degrees = [int(round(np.degrees(a))) for a in angles]

    # Check for empty/constant matrix
    if matrix.size == 0 or np.all(np.isclose(matrix, 0.0)) or np.all(np.isclose(matrix, matrix.flat[0])):
        # Return zeros for all requested feature combinations
        features: Dict[str, float] = {}
        props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "entropy"]
        for prop in props:
            for d in distances:
                if aggregate_angles:
                    features[f"glcm_{prop}_d{d}"] = 0.0
                else:
                    for deg in angle_degrees:
                        features[f"glcm_{prop}_d{d}_a{deg}"] = 0.0
        return features

    q_img = quantize_matrix(matrix, n_levels=n_levels, apply_log1p=apply_log1p)

    glcm = graycomatrix(
        q_img,
        distances=distances,
        angles=angles,
        levels=n_levels,
        symmetric=symmetric,
        normed=normed,
    )

    features = {}

    standard_props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    for prop in standard_props:
        vals = graycoprops(glcm, prop)  # shape (len(distances), len(angles))
        # Handle possible NaN in correlation when variance is zero
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

        for i, d in enumerate(distances):
            if aggregate_angles:
                features[f"glcm_{prop}_d{d}"] = float(np.mean(vals[i, :]))
            else:
                for j, deg in enumerate(angle_degrees):
                    features[f"glcm_{prop}_d{d}_a{deg}"] = float(vals[i, j])

    # Entropy
    entropy_vals = compute_glcm_entropy(glcm)  # shape (len(distances), len(angles))
    entropy_vals = np.nan_to_num(entropy_vals, nan=0.0, posinf=0.0, neginf=0.0)

    for i, d in enumerate(distances):
        if aggregate_angles:
            features[f"glcm_entropy_d{d}"] = float(np.mean(entropy_vals[i, :]))
        else:
            for j, deg in enumerate(angle_degrees):
                features[f"glcm_entropy_d{d}_a{deg}"] = float(entropy_vals[i, j])

    return features
