"""Image-processing feature extraction modules for FCGR representations."""

from adaptive_cgr_vision.features.glcm import extract_glcm_features, quantize_matrix
from adaptive_cgr_vision.features.spatial import extract_spatial_features
from adaptive_cgr_vision.features.frequency import (
    extract_dct_features,
    extract_wavelet_features,
    extract_frequency_features,
)
from adaptive_cgr_vision.features.edge import extract_edge_features
from adaptive_cgr_vision.features.extractor import FeatureExtractor
from adaptive_cgr_vision.features.registry import (
    FEATURE_FAMILIES,
    get_feature_family,
    group_features_by_family,
)

__all__ = [
    "extract_glcm_features",
    "quantize_matrix",
    "extract_spatial_features",
    "extract_dct_features",
    "extract_wavelet_features",
    "extract_frequency_features",
    "extract_edge_features",
    "FeatureExtractor",
    "FEATURE_FAMILIES",
    "get_feature_family",
    "group_features_by_family",
]
