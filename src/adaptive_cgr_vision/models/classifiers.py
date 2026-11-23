"""Classifier model factory and pipeline builders.

Provides consistent, reproducible classifiers with calibration and probability support:
- Logistic Regression
- Linear SVM
- RBF Kernel SVM
- Random Forest
"""

from typing import Any, Dict, Optional, Union
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator


def create_classifier(
    classifier_name: str = "svm_rbf",
    random_state: int = 42,
    params: Optional[Dict[str, Any]] = None,
) -> BaseEstimator:
    """Create a classifier instance configured according to standard experiment parameters.

    Parameters
    ----------
    classifier_name : str
        One of 'logistic_regression', 'svm_linear', 'svm_rbf', 'random_forest'.
    random_state : int
        Seed for reproducibility.
    params : Optional[Dict[str, Any]]
        Optional hyperparameter overrides.

    Returns
    -------
    BaseEstimator
        Configured scikit-learn classifier with probability estimation support.
    """
    user_params = params or {}

    if classifier_name == "logistic_regression":
        base_params = {
            "max_iter": 1000,
            "random_state": random_state,
            "class_weight": "balanced",
            "C": 1.0,
        }
        base_params.update(user_params)
        return LogisticRegression(**base_params)

    elif classifier_name == "svm_linear":
        base_params = {
            "kernel": "linear",
            "probability": True,
            "random_state": random_state,
            "class_weight": "balanced",
            "C": 1.0,
        }
        base_params.update(user_params)
        return SVC(**base_params)

    elif classifier_name == "svm_rbf":
        base_params = {
            "kernel": "rbf",
            "probability": True,
            "random_state": random_state,
            "class_weight": "balanced",
            "C": 1.0,
            "gamma": "scale",
        }
        base_params.update(user_params)
        return SVC(**base_params)

    elif classifier_name == "random_forest":
        base_params = {
            "n_estimators": 100,
            "random_state": random_state,
            "class_weight": "balanced",
            "n_jobs": -1,
        }
        base_params.update(user_params)
        return RandomForestClassifier(**base_params)

    else:
        raise ValueError(f"Unknown classifier name: '{classifier_name}'")
