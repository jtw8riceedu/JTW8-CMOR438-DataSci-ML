"""Tests for DecisionTreeClassifier and DecisionTreeRegressor."""

import numpy as np
import pytest

from src.ml_package.supervised_learning.decision_tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_clf_data():
    X = np.array([
        [1.0, 2.0], [2.0, 3.0], [3.0, 1.0],
        [6.0, 7.0], [7.0, 6.0], [8.0, 8.0],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


@pytest.fixture
def multiclass_clf_data():
    rng = np.random.default_rng(42)
    X0 = rng.standard_normal((30, 2)) + np.array([-3, 0])
    X1 = rng.standard_normal((30, 2)) + np.array([0, 3])
    X2 = rng.standard_normal((30, 2)) + np.array([3, 0])
    X = np.vstack([X0, X1, X2])
    y = np.array([0] * 30 + [1] * 30 + [2] * 30)
    return X, y


@pytest.fixture
def reg_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 2))
    y = X[:, 0] * 2 + X[:, 1] + rng.standard_normal(80) * 0.1
    return X, y


# ---------------------------------------------------------------------------
# DecisionTreeClassifier — initialization
# ---------------------------------------------------------------------------

def test_classifier_default_params():
    tree = DecisionTreeClassifier()
    assert tree.max_depth == 3
    assert tree.criterion == "entropy"
    assert tree.min_samples_split == 2


def test_classifier_invalid_criterion():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(criterion="bad").fit(
            np.ones((4, 2)), np.array([0, 0, 1, 1])
        )


# ---------------------------------------------------------------------------
# DecisionTreeClassifier — training
# ---------------------------------------------------------------------------

def test_classifier_fit_returns_self(binary_clf_data):
    X, y = binary_clf_data
    tree = DecisionTreeClassifier(max_depth=3)
    assert tree.fit(X, y) is tree


def test_classifier_predict_shape(binary_clf_data):
    X, y = binary_clf_data
    tree = DecisionTreeClassifier(max_depth=3).fit(X, y)
    preds = tree.predict(X)
    assert preds.shape == (len(y),)


def test_classifier_perfect_fit_binary(binary_clf_data):
    X, y = binary_clf_data
    tree = DecisionTreeClassifier(max_depth=5).fit(X, y)
    np.testing.assert_array_equal(tree.predict(X), y)


def test_classifier_gini_criterion(binary_clf_data):
    X, y = binary_clf_data
    tree = DecisionTreeClassifier(max_depth=3, criterion="gini").fit(X, y)
    preds = tree.predict(X)
    assert preds.shape == (len(y),)


def test_classifier_multiclass(multiclass_clf_data):
    X, y = multiclass_clf_data
    tree = DecisionTreeClassifier(max_depth=5).fit(X, y)
    assert tree.accuracy(X, y) > 0.85


# ---------------------------------------------------------------------------
# DecisionTreeClassifier — evaluation
# ---------------------------------------------------------------------------

def test_classifier_score_equals_accuracy(binary_clf_data):
    X, y = binary_clf_data
    tree = DecisionTreeClassifier(max_depth=5).fit(X, y)
    assert tree.score(X, y) == tree.accuracy(X, y)


def test_classifier_confusion_matrix_shape(binary_clf_data):
    X, y = binary_clf_data
    tree = DecisionTreeClassifier(max_depth=3).fit(X, y)
    cm = tree.confusion_matrix(X, y)
    assert cm.shape == (2, 2)


def test_classifier_confusion_matrix_sum(binary_clf_data):
    X, y = binary_clf_data
    tree = DecisionTreeClassifier(max_depth=3).fit(X, y)
    cm = tree.confusion_matrix(X, y)
    assert cm.sum() == len(y)


def test_classifier_precision_in_range(binary_clf_data):
    X, y = binary_clf_data
    tree = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert 0.0 <= tree.precision(X, y) <= 1.0


def test_classifier_recall_in_range(binary_clf_data):
    X, y = binary_clf_data
    tree = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert 0.0 <= tree.recall(X, y) <= 1.0


def test_classifier_f1_in_range(binary_clf_data):
    X, y = binary_clf_data
    tree = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert 0.0 <= tree.f1_score(X, y) <= 1.0


# ---------------------------------------------------------------------------
# DecisionTreeRegressor
# ---------------------------------------------------------------------------

def test_regressor_fit_returns_self(reg_data):
    X, y = reg_data
    tree = DecisionTreeRegressor(max_depth=3)
    assert tree.fit(X, y) is tree


def test_regressor_predict_shape(reg_data):
    X, y = reg_data
    tree = DecisionTreeRegressor(max_depth=4).fit(X, y)
    preds = tree.predict(X)
    assert preds.shape == (len(y),)


def test_regressor_deep_tree_low_mse(reg_data):
    X, y = reg_data
    tree = DecisionTreeRegressor(max_depth=10).fit(X, y)
    preds = tree.predict(X)
    mse = np.mean((preds - y) ** 2)
    assert mse < 1.0


def test_regressor_shallow_vs_deep(reg_data):
    """Deeper tree should have lower training MSE."""
    X, y = reg_data
    shallow = DecisionTreeRegressor(max_depth=2).fit(X, y)
    deep = DecisionTreeRegressor(max_depth=8).fit(X, y)
    mse_shallow = np.mean((shallow.predict(X) - y) ** 2)
    mse_deep = np.mean((deep.predict(X) - y) ** 2)
    assert mse_deep <= mse_shallow
