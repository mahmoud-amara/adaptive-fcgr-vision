"""Feature registry and schema management for Adaptive CGR-Vision.

Ensures strict naming, family categorization, and stability across runs.
"""

from typing import Dict, List, Set
from dataclasses import dataclass


FEATURE_FAMILIES = {
    "glcm": "Family A: Texture (GLCM / Haralick)",
    "spatial": "Family B: Spatial Distribution and Moments",
    "dct": "Family C1: 2D DCT Multiscale Frequency",
    "wavelet": "Family C2: 2D Discrete Wavelet Multiscale Frequency",
    "frequency": "Family C: Multiscale Frequency Summaries (DCT + Wavelet)",
    "edge": "Family D: Local Contrast / Edge Statistics",
}


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str
    description: str
    dtype: str = "float64"


def get_feature_family(feature_name: str) -> str:
    """Determine the feature family from a standardized feature name."""
    if feature_name.startswith("glcm_"):
        return "glcm"
    elif feature_name.startswith("spatial_"):
        return "spatial"
    elif feature_name.startswith("dct_"):
        return "dct"
    elif feature_name.startswith("wavelet_"):
        return "wavelet"
    elif feature_name.startswith("edge_") or feature_name.startswith("local_var_") or feature_name.startswith("contrast_"):
        return "edge"
    else:
        raise ValueError(f"Unknown feature family for feature name: '{feature_name}'")


def group_features_by_family(feature_names: List[str]) -> Dict[str, List[str]]:
    """Group a list of feature names by their respective feature family."""
    grouped: Dict[str, List[str]] = {
        "glcm": [],
        "spatial": [],
        "frequency": [],
        "edge": [],
    }
    for name in feature_names:
        family = get_feature_family(name)
        if family in ("dct", "wavelet"):
            grouped["frequency"].append(name)
        else:
            grouped[family].append(name)
    return grouped
