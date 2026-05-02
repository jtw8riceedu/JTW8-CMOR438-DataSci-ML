"""Tests for RandomForestClassifier."""

import numpy as np
import pytest

from src.ml_package.supervised_learning.random_forest import RandomForestClassifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_data():
    rng = np.random.default_rng(7)
    X0 = rng.standard_normal((50, 2)) + np.array([-2, -2])
    X1 = rng.standard_normal((50, 2)) + np.array([2, 2])
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)
    return X, y


@pytest.fixture
def multiclass_data():
    rng = np.random.default_rng(8)
    X0 = rng.standard_normal((40, 2)) + np.array([-3, 0])
    X1 = rng.standard_normal((40, 2)) + np.array([0, 3])
    X2 = rng.standard_normal((40, 2)) + np.array([3, 0])
    X = np.vstack([X0, X1, X2])
    y = np.array([0] * 40 + [1] * 40 + [2] * 40)
    return X, y


@pytest.fixture
def trained_rf(binary_data):
    X, y = binary_data
    rf = RandomForestClassifier(
        n_estimators=20, max_depth=4, random_state=0
    ).fit(X, y)
    return rf, X, y


# ---------------------------------------------------------------------------
# Initialization / fit
# ---------------------------------------------------------------------------

def test_fit_returns_self(binary_data):
    X, y = binary_data
    rf = RandomForestClassifier(n_estimators=5, max_depth=3)
    assert rf.fit(X, y) is rf


def test_number_of_trees(binary_data):
    X, y = binary_data
    n = 15
    rf = RandomForestClassifier(n_estimators=n, max_depth=3).fit(X, y)
    assert len(rf.estimators_) == n


def test_classes_set_after_fit(binary_data):
    X, y = binary_data
    rf = RandomForestClassifier(n_estimators=5, max_depth=3).fit(X, y)
    assert set(rf.classes_) == {0, 1}


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def test_predict_shape(trained_rf):
    rf, X, y = trained_rf
    preds = rf.predict(X)
    assert preds.shape == (len(y),)


def test_predict_values_are_valid_classes(trained_rf):
    rf, X, y = trained_rf
    preds = rf.predict(X)
    assert set(preds).issubset(set(rf.classes_))


def test_accuracy_above_threshold(trained_rf):
    rf, X, y = trained_rf
    assert rf.accuracy(X, y) > 0.90


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test_score_equals_accuracy(trained_rf):
    rf, X, y = trained_rf
    assert rf.score(X, y) == rf.accuracy(X, y)


def test_confusion_matrix_shape(trained_rf):
    rf, X, y = trained_rf
    cm = rf.confusion_matrix(X, y)
    assert cm.shape == (2, 2)


def test_confusion_matrix_sum(trained_rf):
    rf, X, y = trained_rf
    cm = rf.confusion_matrix(X, y)
    assert cm.sum() == len(y)


def test_precision_in_range(trained_rf):
    rf, X, y = trained_rf
    assert 0.0 <= rf.precision(X, y) <= 1.0


def test_recall_in_range(trained_rf):
    rf, X, y = trained_rf
    assert 0.0 <= rf.recall(X, y) <= 1.0


def test_f1_in_range(trained_rf):
    rf, X, y = trained_rf
    assert 0.0 <= rf.f1_score(X, y) <= 1.0


# ---------------------------------------------------------------------------
# Reproducibility & multiclass
# ---------------------------------------------------------------------------

def test_reproducible_with_random_state(binary_data):
    X, y = binary_data
    rf1 = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42).fit(X, y)
    rf2 = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42).fit(X, y)
    np.testing.assert_array_equal(rf1.predict(X), rf2.predict(X))


def test_multiclass_accuracy(multiclass_data):
    X, y = multiclass_data
    rf = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=0).fit(X, y)
    assert rf.accuracy(X, y) > 0.80
