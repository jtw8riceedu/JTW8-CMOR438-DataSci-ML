import numpy as np

from src.ml_package.utils import classification_metrics as metrics


def test_binary_classification_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])

    np.testing.assert_array_equal(
        metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]),
        np.array([[1, 1], [1, 1]]),
    )
    assert metrics.accuracy(y_true, y_pred) == 0.5
    assert metrics.precision(y_true, y_pred, average="binary") == 0.5
    assert metrics.recall(y_true, y_pred, average="binary") == 0.5
    assert metrics.f1_score(y_true, y_pred, average="binary") == 0.5


def test_multiclass_classification_metrics():
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 2, 2, 1])

    np.testing.assert_array_equal(
        metrics.confusion_matrix(y_true, y_pred),
        np.array([[1, 0, 0], [0, 0, 1], [0, 1, 1]]),
    )
    assert metrics.accuracy(y_true, y_pred) == 0.5
    assert metrics.precision(y_true, y_pred, average="micro") == 0.5
    assert metrics.recall(y_true, y_pred, average="micro") == 0.5
    assert metrics.f1_score(y_true, y_pred, average="micro") == 0.5
