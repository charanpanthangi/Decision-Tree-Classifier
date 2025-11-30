"""Evaluation helpers for the Decision Tree classifier example."""

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


Metrics = Dict[str, float]


def evaluate_model(
    y_true, y_pred
) -> Tuple[Metrics, np.ndarray, str]:
    """
    Compute evaluation metrics for classification.

    Parameters
    ----------
    y_true : Iterable
        Ground-truth labels.
    y_pred : Iterable
        Predicted labels.

    Returns
    -------
    Tuple[Metrics, np.ndarray, str]
        Metrics dictionary, confusion matrix array, and classification report string.
    """

    metrics: Metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    return metrics, cm, report


__all__ = ["evaluate_model", "Metrics"]
