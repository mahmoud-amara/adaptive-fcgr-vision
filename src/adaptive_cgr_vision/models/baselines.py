"""Baseline classifiers for genomic sequence classification comparison.

Baselines:
1. Raw FCGR: Direct flattened FCGR matrix classifier.
2. Canonical K-mer Spectrum: Standard k-mer frequency vector classifier.
3. Smith-Waterman Alignment: 1-Nearest Neighbor alignment classifier using local alignment scores.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from Bio.Align import PairwiseAligner

from adaptive_cgr_vision.representations.fcgr import compute_fcgr


class RawFCGRBaseline(BaseEstimator, ClassifierMixin):
    """Direct flattened FCGR vector baseline classifier."""

    def __init__(
        self,
        base_estimator: Optional[BaseEstimator] = None,
        k: int = 6,
        normalization: str = "count",
    ):
        self.base_estimator = base_estimator
        self.k = k
        self.normalization = normalization
        self.estimator_: Optional[BaseEstimator] = None

    def fit(self, sequences: List[str], y: np.ndarray) -> "RawFCGRBaseline":
        X = np.array([
            compute_fcgr(s, k=self.k, normalization=self.normalization).matrix.flatten()
            for s in sequences
        ], dtype=np.float32)

        from sklearn.svm import SVC
        self.estimator_ = self.base_estimator or SVC(kernel="rbf", probability=True, random_state=42)
        self.estimator_.fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, sequences: List[str]) -> np.ndarray:
        X = np.array([
            compute_fcgr(s, k=self.k, normalization=self.normalization).matrix.flatten()
            for s in sequences
        ], dtype=np.float32)
        return self.estimator_.predict(X)

    def predict_proba(self, sequences: List[str]) -> np.ndarray:
        X = np.array([
            compute_fcgr(s, k=self.k, normalization=self.normalization).matrix.flatten()
            for s in sequences
        ], dtype=np.float32)
        return self.estimator_.predict_proba(X)


class SmithWatermanKNNClassifier(BaseEstimator, ClassifierMixin):
    """Smith-Waterman 1-Nearest Neighbor local alignment classifier baseline.

    Uses Bio.Align.PairwiseAligner (Smith-Waterman local mode).
    """

    def __init__(
        self,
        match_score: float = 2.0,
        mismatch_score: float = -3.0,
        open_gap_score: float = -5.0,
        extend_gap_score: float = -2.0,
    ):
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        self.open_gap_score = open_gap_score
        self.extend_gap_score = extend_gap_score

        self.train_sequences_: List[str] = []
        self.train_labels_: np.ndarray = np.array([])
        self.classes_: np.ndarray = np.array([])

    def fit(self, sequences: List[str], y: np.ndarray) -> "SmithWatermanKNNClassifier":
        self.train_sequences_ = list(sequences)
        self.train_labels_ = np.asarray(y)
        self.classes_ = np.unique(self.train_labels_)
        return self

    def _get_aligner(self) -> PairwiseAligner:
        aligner = PairwiseAligner()
        aligner.mode = "local"
        aligner.match_score = self.match_score
        aligner.mismatch_score = self.mismatch_score
        aligner.open_gap_score = self.open_gap_score
        aligner.extend_gap_score = self.extend_gap_score
        return aligner

    def predict(self, query_sequences: List[str]) -> np.ndarray:
        """Predict class by finding the training sequence with maximum SW score."""
        aligner = self._get_aligner()
        predictions = []

        for q_seq in query_sequences:
            best_score = -float("inf")
            best_label = self.train_labels_[0]

            for train_seq, train_lbl in zip(self.train_sequences_, self.train_labels_):
                score = aligner.score(q_seq, train_seq)
                if score > best_score:
                    best_score = score
                    best_label = train_lbl

            predictions.append(best_label)

        return np.array(predictions)
