"""Preprocessing utilities for the Decision Tree classifier example."""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


# Decision trees are scale-invariant, so we deliberately skip feature scaling here.

def train_test_split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into train and test sets.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels.
    test_size : float, optional
        Fraction of data to reserve for testing, by default 0.2.
    random_state : int, optional
        Seed for reproducibility, by default 42.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        Training features, testing features, training labels, testing labels.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


__all__ = ["train_test_split_data"]
