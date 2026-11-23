"""Fixed feature fusion baseline.

Standard feature concatenation with per-family standard scaling.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class FixedFusionTransformer(BaseEstimator, TransformerMixin):
    """Fixed concatenation baseline with standard scaling."""

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.scaler = StandardScaler(with_mean=self.with_mean, with_std=self.with_std)
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None) -> "FixedFusionTransformer":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            arr = X.values.astype(np.float64)
        else:
            arr = np.asarray(X, dtype=np.float64)
            self.feature_names_ = [f"f_{i}" for i in range(arr.shape[1])]

        self.scaler.fit(arr)
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            arr = X.values.astype(np.float64)
        else:
            arr = np.asarray(X, dtype=np.float64)

        return self.scaler.transform(arr)
