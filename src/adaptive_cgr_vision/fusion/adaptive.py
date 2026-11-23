"""Adaptive multiscale feature fusion transformer.

Learns family and feature weighting strictly on training fold data.
Guarantees zero leakage into validation/test folds.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler

from adaptive_cgr_vision.features.registry import group_features_by_family, get_feature_family


class AdaptiveFusionTransformer(BaseEstimator, TransformerMixin):
    """Adaptive multiscale feature fusion transformer.

    Parameters
    ----------
    method : str
        Weighting strategy: 'supervised_f_score', 'family_variance', or 'uniform'
    smooth_factor : float
        Epsilon smoothing added to weights to avoid zeroing out informative features.
    normalize_weights : bool
        Whether to normalize family weights to sum to 1.
    """

    def __init__(
        self,
        method: str = "supervised_f_score",
        smooth_factor: float = 1e-3,
        normalize_weights: bool = True,
    ):
        self.method = method
        self.smooth_factor = smooth_factor
        self.normalize_weights = normalize_weights

        self.scalers_: Dict[str, StandardScaler] = {}
        self.family_indices_: Dict[str, List[int]] = {}
        self.weights_: Optional[np.ndarray] = None
        self.family_weights_: Dict[str, float] = {}
        self.feature_names_: Optional[List[str]] = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, List[str]]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "AdaptiveFusionTransformer":
        """Fit feature scalers and learn weights strictly on training fold data."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            arr = X.values.astype(np.float64)
        else:
            arr = np.asarray(X, dtype=np.float64)
            if feature_names is not None:
                self.feature_names_ = list(feature_names)
            elif self.feature_names_ is None:
                self.feature_names_ = [f"f_{i}" for i in range(arr.shape[1])]

        n_samples, n_features = arr.shape

        # Group indices by family
        self.family_indices_ = {}
        for idx, feat_name in enumerate(self.feature_names_):
            try:
                fam = get_feature_family(feat_name)
            except ValueError:
                fam = "default"
            if fam not in self.family_indices_:
                self.family_indices_[fam] = []
            self.family_indices_[fam].append(idx)

        # Fit per-family standard scalers
        self.scalers_ = {}
        scaled_X = np.zeros_like(arr)
        for fam, indices in self.family_indices_.items():
            scaler = StandardScaler()
            scaled_X[:, indices] = scaler.fit_transform(arr[:, indices])
            self.scalers_[fam] = scaler

        # Learn weights strictly on training set
        if self.method == "supervised_f_score" and y is not None:
            y_arr = np.asarray(y)
            # Compute ANOVA F-value
            f_scores, _ = f_classif(scaled_X, y_arr)
            # Replace NaNs/Infs with zero
            f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
            f_scores = f_scores + self.smooth_factor

            # Calculate family aggregate weights
            self.family_weights_ = {}
            for fam, indices in self.family_indices_.items():
                fam_mean_f = float(np.mean(f_scores[indices]))
                self.family_weights_[fam] = fam_mean_f

            if self.normalize_weights:
                total_w = sum(self.family_weights_.values())
                if total_w > 0:
                    self.family_weights_ = {k: v / total_w for k, v in self.family_weights_.items()}

            # Assign feature weights according to family weight
            self.weights_ = np.ones(n_features, dtype=np.float64)
            for fam, indices in self.family_indices_.items():
                w = self.family_weights_[fam]
                self.weights_[indices] = np.sqrt(w)

        elif self.method == "family_variance":
            self.family_weights_ = {}
            for fam, indices in self.family_indices_.items():
                fam_var = float(np.mean(np.var(arr[:, indices], axis=0)))
                # Inverse variance weighting
                self.family_weights_[fam] = 1.0 / (fam_var + self.smooth_factor)

            if self.normalize_weights:
                total_w = sum(self.family_weights_.values())
                if total_w > 0:
                    self.family_weights_ = {k: v / total_w for k, v in self.family_weights_.items()}

            self.weights_ = np.ones(n_features, dtype=np.float64)
            for fam, indices in self.family_indices_.items():
                w = self.family_weights_[fam]
                self.weights_[indices] = np.sqrt(w)

        else:
            # Uniform baseline weighting
            self.weights_ = np.ones(n_features, dtype=np.float64)
            self.family_weights_ = {fam: 1.0 / len(self.family_indices_) for fam in self.family_indices_}

        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Transform features using fitted scalers and learned weights."""
        if isinstance(X, pd.DataFrame):
            arr = X.values.astype(np.float64)
        else:
            arr = np.asarray(X, dtype=np.float64)

        transformed = np.zeros_like(arr)
        for fam, indices in self.family_indices_.items():
            scaler = self.scalers_[fam]
            transformed[:, indices] = scaler.transform(arr[:, indices])

        if self.weights_ is not None:
            transformed = transformed * self.weights_

        return transformed
