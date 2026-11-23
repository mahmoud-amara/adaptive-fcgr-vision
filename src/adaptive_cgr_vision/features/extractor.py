"""Unified FeatureExtractor for Adaptive CGR-Vision.

Extracts all or selective feature families from an FCGR matrix or DNA sequence,
producing deterministic, strictly-ordered feature vectors and dataframes.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from adaptive_cgr_vision.representations.fcgr import compute_fcgr, FCGRResult
from adaptive_cgr_vision.features.glcm import extract_glcm_features
from adaptive_cgr_vision.features.spatial import extract_spatial_features
from adaptive_cgr_vision.features.frequency import extract_frequency_features
from adaptive_cgr_vision.features.edge import extract_edge_features
from adaptive_cgr_vision.features.registry import group_features_by_family


class FeatureExtractor:
    """Extracts multiscale image features from FCGR matrices."""

    def __init__(
        self,
        families: Optional[List[str]] = None,
        glcm_distances: Optional[List[int]] = None,
        glcm_n_levels: int = 32,
        dct_bands: int = 4,
        wavelet: str = "haar",
        wavelet_levels: int = 3,
        edge_thresholds: Optional[List[float]] = None,
        edge_windows: Optional[List[int]] = None,
    ):
        if families is None:
            self.families = ["glcm", "spatial", "frequency", "edge"]
        else:
            self.families = families

        self.glcm_distances = glcm_distances or [1, 2, 3]
        self.glcm_n_levels = glcm_n_levels
        self.dct_bands = dct_bands
        self.wavelet = wavelet
        self.wavelet_levels = wavelet_levels
        self.edge_thresholds = edge_thresholds or [0.01, 0.05, 0.1]
        self.edge_windows = edge_windows or [3, 5]

        self._feature_names: Optional[List[str]] = None

    def extract_from_matrix(self, matrix: np.ndarray) -> Dict[str, float]:
        """Extract features from a 2D FCGR matrix.

        Returns
        -------
        Dict[str, float]
            Ordered dictionary of extracted features.
        """
        features: Dict[str, float] = {}

        if "glcm" in self.families:
            glcm_feats = extract_glcm_features(
                matrix,
                distances=self.glcm_distances,
                n_levels=self.glcm_n_levels,
                aggregate_angles=True,
            )
            features.update(glcm_feats)

        if "spatial" in self.families:
            spatial_feats = extract_spatial_features(matrix)
            features.update(spatial_feats)

        if "frequency" in self.families:
            freq_feats = extract_frequency_features(
                matrix,
                n_bands=self.dct_bands,
                wavelet=self.wavelet,
                wavelet_levels=self.wavelet_levels,
            )
            features.update(freq_feats)

        if "edge" in self.families:
            edge_feats = extract_edge_features(
                matrix,
                thresholds=self.edge_thresholds,
                windows=self.edge_windows,
            )
            features.update(edge_feats)

        # Sort feature keys alphabetically for strict deterministic column ordering
        sorted_keys = sorted(features.keys())
        ordered_features = {k: features[k] for k in sorted_keys}

        if self._feature_names is None:
            self._feature_names = sorted_keys

        return ordered_features

    def extract_from_sequence(
        self,
        sequence: str,
        k: int = 6,
        normalization: str = "count",
        ambiguity_policy: str = "split_runs",
    ) -> Dict[str, float]:
        """Convenience method: compute FCGR from sequence and extract features."""
        fcgr_res = compute_fcgr(
            sequence=sequence,
            k=k,
            normalization=normalization,
            ambiguity_policy=ambiguity_policy,
        )
        return self.extract_from_matrix(fcgr_res.matrix)

    @property
    def feature_names(self) -> List[str]:
        """Get the stable list of feature names."""
        if self._feature_names is None:
            # Generate dummy matrix to discover feature names
            dummy = np.ones((64, 64), dtype=np.float32) / (64 * 64)
            self.extract_from_matrix(dummy)
        return list(self._feature_names)
