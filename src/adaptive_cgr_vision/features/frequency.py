"""Multiscale frequency feature extraction (DCT and 2D Discrete Wavelet Transform).

Family C in Feature Registry:
- dct_energy_s0, dct_energy_s1, dct_energy_s2, dct_energy_s3
- dct_ratio_low
- wavelet_energy_L{l}_{subband}
- wavelet_entropy_L{l}
"""

from typing import Dict, List, Optional
import numpy as np
from scipy.fft import dctn
import pywt


def extract_dct_features(
    matrix: np.ndarray,
    n_bands: int = 4,
) -> Dict[str, float]:
    """Extract 2D DCT frequency band energy features.

    Parameters
    ----------
    matrix : np.ndarray
        2D float FCGR matrix.
    n_bands : int
        Number of concentric radial frequency bands (default: 4).

    Returns
    -------
    Dict[str, float]
        Dictionary with dct_energy_s{s} and dct_ratio_low.
    """
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {matrix.shape}")

    arr = np.asarray(matrix, dtype=np.float64)
    nrows, ncols = arr.shape

    features: Dict[str, float] = {}

    if arr.size == 0 or np.all(np.isclose(arr, 0.0)):
        for s in range(n_bands):
            features[f"dct_energy_s{s}"] = 0.0
        features["dct_ratio_low"] = 0.0
        return features

    # Compute 2D Discrete Cosine Transform (orthonormal)
    dct_coeff = dctn(arr, type=2, norm="ortho")
    energy_map = dct_coeff ** 2
    total_energy = float(np.sum(energy_map))

    # Frequency coordinates (normalized distance from DC component at (0,0))
    u_coords, v_coords = np.mgrid[0:nrows, 0:ncols]
    # Normalized radial distance in [0, 1]
    max_radius = np.sqrt((nrows - 1) ** 2 + (ncols - 1) ** 2)
    if max_radius == 0:
        max_radius = 1.0
    radii = np.sqrt(u_coords ** 2 + v_coords ** 2) / max_radius

    # Split into n_bands concentric radial bands
    band_edges = np.linspace(0.0, 1.0 + 1e-9, n_bands + 1)

    low_energy = 0.0
    for s in range(n_bands):
        mask = (radii >= band_edges[s]) & (radii < band_edges[s + 1])
        band_energy = float(np.sum(energy_map[mask]))
        features[f"dct_energy_s{s}"] = band_energy
        if s == 0:
            low_energy = band_energy

    if total_energy > 1e-15:
        features["dct_ratio_low"] = float(low_energy / total_energy)
    else:
        features["dct_ratio_low"] = 0.0

    return features


def extract_wavelet_features(
    matrix: np.ndarray,
    wavelet: str = "haar",
    max_levels: int = 3,
) -> Dict[str, float]:
    """Extract multiscale 2D Discrete Wavelet Transform energy and entropy features.

    Parameters
    ----------
    matrix : np.ndarray
        2D float FCGR matrix.
    wavelet : str
        Wavelet family name (default: 'haar').
    max_levels : int
        Maximum decomposition level (default: 3).

    Returns
    -------
    Dict[str, float]
        Dictionary with wavelet subband energies and level entropies.
    """
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {matrix.shape}")

    arr = np.asarray(matrix, dtype=np.float64)
    nrows, ncols = arr.shape

    features: Dict[str, float] = {}

    # Determine maximum valid decomposition level for this shape
    max_possible = pywt.dwt_max_level(min(nrows, ncols), pywt.Wavelet(wavelet).dec_len)
    levels = min(max_levels, max(1, max_possible))

    if arr.size == 0 or np.all(np.isclose(arr, 0.0)):
        for lvl in range(1, max_levels + 1):
            features[f"wavelet_energy_L{lvl}_H"] = 0.0
            features[f"wavelet_energy_L{lvl}_V"] = 0.0
            features[f"wavelet_energy_L{lvl}_D"] = 0.0
            features[f"wavelet_entropy_L{lvl}"] = 0.0
        features[f"wavelet_energy_approx"] = 0.0
        return features

    # pywt.wavedec2 returns [cAn, (cHn, cVn, cDn), ..., (cH1, cV1, cD1)]
    coeffs = pywt.wavedec2(arr, wavelet=wavelet, level=levels)
    c_approx = coeffs[0]
    features["wavelet_energy_approx"] = float(np.sum(c_approx ** 2))

    detail_tuples = coeffs[1:]  # from level N down to 1
    # detail_tuples has length = levels, where detail_tuples[0] is level=levels, detail_tuples[-1] is level=1

    for idx, (cH, cV, cD) in enumerate(reversed(detail_tuples)):
        lvl = idx + 1  # level 1 to levels
        e_H = float(np.sum(cH ** 2))
        e_V = float(np.sum(cV ** 2))
        e_D = float(np.sum(cD ** 2))

        features[f"wavelet_energy_L{lvl}_H"] = e_H
        features[f"wavelet_energy_L{lvl}_V"] = e_V
        features[f"wavelet_energy_L{lvl}_D"] = e_D

        # Level entropy across subband energies
        sub_energies = np.array([e_H, e_V, e_D], dtype=np.float64)
        lvl_total = np.sum(sub_energies)
        if lvl_total > 1e-15:
            p_sub = sub_energies / lvl_total
            mask = p_sub > 1e-15
            lvl_entropy = -float(np.sum(p_sub[mask] * np.log2(p_sub[mask])))
        else:
            lvl_entropy = 0.0

        features[f"wavelet_entropy_L{lvl}"] = lvl_entropy

    # Fill in any remaining levels up to max_levels with zeros for stable schema
    for lvl in range(levels + 1, max_levels + 1):
        features[f"wavelet_energy_L{lvl}_H"] = 0.0
        features[f"wavelet_energy_L{lvl}_V"] = 0.0
        features[f"wavelet_energy_L{lvl}_D"] = 0.0
        features[f"wavelet_entropy_L{lvl}"] = 0.0

    return features


def extract_frequency_features(
    matrix: np.ndarray,
    n_bands: int = 4,
    wavelet: str = "haar",
    wavelet_levels: int = 3,
) -> Dict[str, float]:
    """Extract combined frequency features (DCT + Wavelet)."""
    dct_feats = extract_dct_features(matrix, n_bands=n_bands)
    wv_feats = extract_wavelet_features(matrix, wavelet=wavelet, max_levels=wavelet_levels)
    combined = {}
    combined.update(dct_feats)
    combined.update(wv_feats)
    return combined
