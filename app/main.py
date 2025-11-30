"""Entry point to run the Decision Tree classifier pipeline."""

from typing import Dict

from app.data import load_data
from app.evaluate import evaluate_model
from app.model import build_model, train_model
from app.preprocess import train_test_split_data
from app.visualize import plot_confusion_matrix, plot_feature_importance, plot_pca_scatter


def run_pipeline() -> Dict[str, float]:
    """
    Execute the end-to-end pipeline: load, split, train, evaluate, and visualize.

    Returns
    -------
    Dict[str, float]
        Dictionary of key evaluation metrics.
    """

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    model = build_model()
    model = train_model(model, X_train, y_train)

    y_pred = model.predict(X_test)
    metrics, cm, report = evaluate_model(y_test, y_pred)

    # Visualizations
    plot_confusion_matrix(cm, labels=y.unique())
    plot_feature_importance(model, feature_names=list(X.columns))
    plot_pca_scatter(X, y)

    print("\nDecision Tree Classifier Results")
    print("=" * 35)
    print(report)
    print("Feature importances:")
    for name, importance in zip(X.columns, model.feature_importances_):
        print(f"  {name}: {importance:.3f}")
    print("\nSaved figures in the 'figures' directory.")

    return metrics


if __name__ == "__main__":
    run_pipeline()
