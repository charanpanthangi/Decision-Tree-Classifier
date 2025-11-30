"""Data loading utilities for the Decision Tree classifier example."""

from typing import Tuple

import pandas as pd
from sklearn.datasets import load_iris


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the Iris dataset from scikit-learn.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Feature matrix ``X`` and target vector ``y`` as pandas objects.
    """

    iris = load_iris(as_frame=True)
    X: pd.DataFrame = iris.data
    y: pd.Series = iris.target
    return X, y


__all__ = ["load_data"]
