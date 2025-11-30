import numpy as np

from app.evaluate import evaluate_model


def test_evaluate_model_metrics():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    metrics, cm, report = evaluate_model(y_true, y_pred)

    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1_score"}
    assert cm.shape == (2, 2)
    assert isinstance(report, str)
