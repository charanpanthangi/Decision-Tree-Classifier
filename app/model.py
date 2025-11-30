"""Model definition and training utilities for the Decision Tree classifier example."""

from typing import Dict, Optional

from sklearn.tree import DecisionTreeClassifier


DEFAULT_PARAMS: Dict[str, Optional[int | float | str]] = {
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42,
}


def build_model(**kwargs) -> DecisionTreeClassifier:
    """
    Build a ``DecisionTreeClassifier`` with provided hyperparameters.

    Parameters
    ----------
    **kwargs : Any
        Hyperparameters to override defaults.

    Returns
    -------
    DecisionTreeClassifier
        Configured classifier instance.
    """

    params = {**DEFAULT_PARAMS, **kwargs}
    return DecisionTreeClassifier(**params)


def train_model(
    model: DecisionTreeClassifier, X_train, y_train
) -> DecisionTreeClassifier:
    """
    Fit the classifier on training data.

    Parameters
    ----------
    model : DecisionTreeClassifier
        Unfitted decision tree model.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.

    Returns
    -------
    DecisionTreeClassifier
        Fitted model.
    """

    model.fit(X_train, y_train)
    return model


__all__ = ["build_model", "train_model", "DEFAULT_PARAMS"]
