"""Tests for LogisticRegression."""

import numpy as np
import pytest

from src.ml_package.supervised_learning.logistic_regression import LogisticRegression


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_data():
    rng = np.random.default_rng(1)
    X0 = rng.standard_normal((60, 2)) + np.array([-2, -2])
    X1 = rng.standard_normal((60, 2)) + np.array([2, 2])
    X = np.vstack([X0, X1])
    y = np.array([0] * 60 + [1] * 60)
    return X, y


@pytest.fixture
def multinomial_data():
    rng = np.random.default_rng(2)
    X0 = rng.standard_normal((40, 2)) + np.array([-3, 0])
    X1 = rng.standard_normal((40, 2)) + np.array([0, 3])
    X2 = rng.standard_normal((40, 2)) + np.array([3, 0])
    X = np.vstack([X0, X1, X2])
    labels = np.array([0] * 40 + [1] * 40 + [2] * 40)
    # One-hot encode
    y_onehot = np.zeros((120, 3))
    y_onehot[np.arange(120), labels] = 1
    return X, y_onehot, labels


@pytest.fixture
def trained_binary(binary_data):
    X, y = binary_data
    model = LogisticRegression()
    model.train(X, y, eta=0.1, epochs=500, task="binary")
    return model, X, y


@pytest.fixture
def trained_multinomial(multinomial_data):
    X, y_onehot, labels = multinomial_data
    model = LogisticRegression()
    model.train(X, y_onehot, eta=0.1, epochs=500, task="multinomial")
    return model, X, y_onehot, labels


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_initial_state():
    model = LogisticRegression()
    assert model.weights is None
    assert model.bias is None
    assert model.task is None
    assert model.losses == []


# ---------------------------------------------------------------------------
# Training validation
# ---------------------------------------------------------------------------

def test_invalid_task_raises_value_error():
    model = LogisticRegression()
    X = np.ones((10, 2))
    y = np.zeros(10)
    with pytest.raises(ValueError, match="Task must be one of"):
        model.train(X, y, task="clustering")


def test_feedforward_before_train_raises_runtime_error():
    model = LogisticRegression()
    with pytest.raises(RuntimeError):
        model.feedforward(np.ones((5, 2)))


# ---------------------------------------------------------------------------
# Binary classification
# ---------------------------------------------------------------------------

def test_binary_train_records_losses(binary_data):
    X, y = binary_data
    model = LogisticRegression()
    model.train(X, y, eta=0.01, epochs=20, task="binary")
    assert len(model.losses) == 20


def test_binary_predict_shape(trained_binary):
    model, X, y = trained_binary
    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)


def test_binary_predict_values_in_zero_one(trained_binary):
    model, X, y = trained_binary
    preds = model.predict(X)
    assert set(preds).issubset({0, 1})


def test_binary_accuracy_above_threshold(trained_binary):
    model, X, y = trained_binary
    assert model.accuracy(X, y) > 0.90


def test_binary_confusion_matrix_shape(trained_binary):
    model, X, y = trained_binary
    cm = model.confusion_matrix(X, y)
    assert cm.shape == (2, 2)


def test_binary_confusion_matrix_sum(trained_binary):
    model, X, y = trained_binary
    cm = model.confusion_matrix(X, y)
    assert cm.sum() == len(y)


def test_binary_precision_in_range(trained_binary):
    model, X, y = trained_binary
    p = model.precision(X, y)
    assert 0.0 <= p <= 1.0


def test_binary_recall_in_range(trained_binary):
    model, X, y = trained_binary
    r = model.recall(X, y)
    assert 0.0 <= r <= 1.0


def test_binary_f1_score_in_range(trained_binary):
    model, X, y = trained_binary
    f1 = model.f1_score(X, y)
    assert 0.0 <= f1 <= 1.0


def test_binary_loss_decreases(binary_data):
    X, y = binary_data
    model = LogisticRegression()
    model.train(X, y, eta=0.1, epochs=300, task="binary")
    assert model.losses[-1] < model.losses[0]


# ---------------------------------------------------------------------------
# Multinomial classification
# ---------------------------------------------------------------------------

def test_multinomial_train_records_losses(multinomial_data):
    X, y_onehot, _ = multinomial_data
    model = LogisticRegression()
    model.train(X, y_onehot, eta=0.1, epochs=20, task="multinomial")
    assert len(model.losses) == 20


def test_multinomial_weights_shape(multinomial_data):
    X, y_onehot, _ = multinomial_data
    model = LogisticRegression()
    model.train(X, y_onehot, eta=0.1, epochs=10, task="multinomial")
    assert model.weights.shape == (X.shape[1], 3)


def test_multinomial_predict_shape(trained_multinomial):
    model, X, y_onehot, labels = trained_multinomial
    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)


def test_multinomial_accuracy_above_threshold(trained_multinomial):
    model, X, y_onehot, labels = trained_multinomial
    assert model.accuracy(X, y_onehot) > 0.85


def test_multinomial_precision_in_range(trained_multinomial):
    model, X, y_onehot, labels = trained_multinomial
    p = model.precision(X, labels)
    assert 0.0 <= p <= 1.0


def test_multinomial_recall_in_range(trained_multinomial):
    model, X, y_onehot, labels = trained_multinomial
    r = model.recall(X, labels)
    assert 0.0 <= r <= 1.0


def test_multinomial_f1_in_range(trained_multinomial):
    model, X, y_onehot, labels = trained_multinomial
    f1 = model.f1_score(X, labels)
    assert 0.0 <= f1 <= 1.0
