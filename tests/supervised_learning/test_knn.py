"""Tests for KNN (classification and regression)."""

import numpy as np
import pytest

from src.ml_package.supervised_learning.knn import KNN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clf_data():
    """Simple 2-class dataset that is perfectly separable at x[0]=0."""
    X = np.array([[-2.0, 0], [-1.5, 0], [-1.0, 0], [1.0, 0], [1.5, 0], [2.0, 0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


@pytest.fixture
def reg_data():
    """Simple regression dataset: y = x."""
    X = np.arange(1, 11, dtype=float).reshape(-1, 1)
    y = X.ravel().astype(float)
    return X, y


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_invalid_k_raises_value_error():
    with pytest.raises(ValueError):
        KNN(k=0)


def test_k_too_large_raises_after_fit(clf_data):
    X, y = clf_data
    knn = KNN(k=100)
    with pytest.raises(ValueError):
        knn.fit(X, y)


def test_fit_returns_self(clf_data):
    X, y = clf_data
    knn = KNN(k=1)
    assert knn.fit(X, y) is knn


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def test_clf_predict_shape(clf_data):
    X, y = clf_data
    knn = KNN(k=1).fit(X, y)
    preds = knn.predict(X)
    assert preds.shape == (len(y),)


def test_clf_predict_correct_on_train(clf_data):
    X, y = clf_data
    knn = KNN(k=1).fit(X, y)
    np.testing.assert_array_equal(knn.predict(X), y)


def test_clf_score_is_accuracy(clf_data):
    X, y = clf_data
    knn = KNN(k=1).fit(X, y)
    assert knn.score(X, y) == 1.0


def test_clf_accuracy(clf_data):
    X, y = clf_data
    knn = KNN(k=1).fit(X, y)
    assert knn.accuracy(X, y) == 1.0


def test_clf_confusion_matrix_shape(clf_data):
    X, y = clf_data
    knn = KNN(k=1).fit(X, y)
    cm = knn.confusion_matrix(X, y)
    assert cm.shape == (2, 2)


def test_clf_precision_in_range(clf_data):
    X, y = clf_data
    knn = KNN(k=1).fit(X, y)
    p = knn.precision(X, y)
    assert 0.0 <= p <= 1.0


def test_clf_recall_in_range(clf_data):
    X, y = clf_data
    knn = KNN(k=1).fit(X, y)
    r = knn.recall(X, y)
    assert 0.0 <= r <= 1.0


def test_clf_f1_in_range(clf_data):
    X, y = clf_data
    knn = KNN(k=1).fit(X, y)
    f1 = knn.f1_score(X, y)
    assert 0.0 <= f1 <= 1.0


# ---------------------------------------------------------------------------
# Classification metrics blocked for regression mode
# ---------------------------------------------------------------------------

def test_regression_mode_blocks_accuracy(reg_data):
    X, y = reg_data
    knn = KNN(k=1, regression=True).fit(X, y)
    with pytest.raises(ValueError):
        knn.accuracy(X, y)


def test_regression_mode_blocks_confusion_matrix(reg_data):
    X, y = reg_data
    knn = KNN(k=1, regression=True).fit(X, y)
    with pytest.raises(ValueError):
        knn.confusion_matrix(X, y)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def test_reg_predict_shape(reg_data):
    X, y = reg_data
    knn = KNN(k=1, regression=True).fit(X, y)
    preds = knn.predict(X)
    assert preds.shape == (len(y),)


def test_reg_predict_correct_with_k1(reg_data):
    X, y = reg_data
    knn = KNN(k=1, regression=True).fit(X, y)
    np.testing.assert_allclose(knn.predict(X), y)


def test_reg_score_is_mse(reg_data):
    X, y = reg_data
    knn = KNN(k=1, regression=True).fit(X, y)
    # Perfect fit → MSE should be 0
    assert knn.score(X, y) == pytest.approx(0.0, abs=1e-10)


def test_reg_rmse(reg_data):
    X, y = reg_data
    knn = KNN(k=1, regression=True).fit(X, y)
    assert knn.rmse(X, y) == pytest.approx(0.0, abs=1e-10)


def test_reg_mae(reg_data):
    X, y = reg_data
    knn = KNN(k=1, regression=True).fit(X, y)
    assert knn.mae(X, y) == pytest.approx(0.0, abs=1e-10)


def test_reg_r_squared_perfect_fit(reg_data):
    X, y = reg_data
    knn = KNN(k=1, regression=True).fit(X, y)
    assert knn.r_squared(X, y) == pytest.approx(1.0, abs=1e-10)


def test_reg_metrics_blocked_for_classification(clf_data):
    X, y = clf_data
    knn = KNN(k=1).fit(X, y)
    with pytest.raises(ValueError):
        knn.rmse(X, y)
    with pytest.raises(ValueError):
        knn.mae(X, y)
    with pytest.raises(ValueError):
        knn.r_squared(X, y)
