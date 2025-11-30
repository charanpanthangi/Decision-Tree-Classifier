"""Visualization utilities for the Decision Tree classifier example."""

from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

sns.set_theme(style="whitegrid")


FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


def plot_confusion_matrix(cm: np.ndarray, labels: Iterable[str], filename: str = "confusion_matrix.svg") -> Path:
    """Plot and save a confusion matrix heatmap."""

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    output_path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()
    return output_path


def plot_feature_importance(
    model, feature_names: List[str], filename: str = "feature_importance.svg"
) -> Path:
    """Plot and save a bar chart of feature importances."""

    importances = model.feature_importances_
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances, y=feature_names, orient="h", palette="viridis")
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    output_path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()
    return output_path


def plot_pca_scatter(
    X: pd.DataFrame,
    y: Iterable,
    filename: str = "pca_scatter.svg",
    random_state: int = 42,
) -> Path:
    """Plot and save a 2D PCA scatter plot with class colors."""

    pca = PCA(n_components=2, random_state=random_state)
    components: np.ndarray = pca.fit_transform(X)
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        components[:, 0],
        components[:, 1],
        c=y,
        cmap="viridis",
        edgecolor="k",
        alpha=0.8,
    )
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("PCA 2D Scatter")
    plt.legend(*scatter.legend_elements(), title="Classes", loc="best")
    output_path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()
    return output_path


__all__ = [
    "plot_confusion_matrix",
    "plot_feature_importance",
    "plot_pca_scatter",
    "FIGURES_DIR",
]
