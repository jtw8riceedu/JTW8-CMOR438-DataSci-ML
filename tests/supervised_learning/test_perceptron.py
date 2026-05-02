"""Tests for Perceptron."""

import numpy as np
import pytest

from src.ml_package.supervised_learning.perceptron import Perceptron


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def linearly_separable():
    """Labels are +1 / -1."""
    X = np.array([[-2.0, -2], [-1.5, -1], [1.0, 1], [2.0, 2]])
    y = np.array([-1, -1, 1, 1], dtype=float)
    return X, y


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_invalid_activation_raises():
    with pytest.raises(ValueError):
        Perceptron(activation="sigmoid")


def test_valid_activations_do_not_raise():
    for act in ("relu", "leakyrelu", "tanh", "unitstep"):
        p = Perceptron(activation=act)
        assert p.activation == act


def test_initial_weights_and_bias_none():
    p = Perceptron(activation="relu")
    assert p.weights is None
    assert p.bias is None
    assert p.losses == []


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("activation", ["relu", "leakyrelu", "tanh", "unitstep"])
def test_train_sets_weights(activation, linearly_separable):
    X, y = linearly_separable
    p = Perceptron(activation=activation)
    p.train(X, y, eta=0.01, epochs=10)
    assert p.weights is not None
    assert p.bias is not None


def test_train_records_losses(linearly_separable):
    X, y = linearly_separable
    p = Perceptron(activation="tanh")
    p.train(X, y, eta=0.01, epochs=25)
    assert len(p.losses) == 25


def test_train_loss_decreases_tanh(linearly_separable):
    X, y = linearly_separable
    p = Perceptron(activation="tanh")
    p.train(X, y, eta=0.05, epochs=500)
    assert p.losses[-1] <= p.losses[0]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def test_predict_shape(linearly_separable):
    X, y = linearly_separable
    p = Perceptron(activation="tanh")
    p.train(X, y, eta=0.1, epochs=500)
    preds = p.predict(X)
    assert preds.shape == (len(y),)


def test_predict_values_are_plus_minus_one(linearly_separable):
    X, y = linearly_separable
    p = Perceptron(activation="unitstep")
    p.train(X, y, eta=0.1, epochs=500)
    preds = p.predict(X)
    assert set(preds).issubset({-1, 1})


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test_accuracy_in_range(linearly_separable):
    X, y = linearly_separable
    p = Perceptron(activation="tanh")
    p.train(X, y, eta=0.1, epochs=1000)
    acc = p.accuracy(X, y)
    assert 0.0 <= acc <= 1.0


def test_confusion_matrix_shape(linearly_separable):
    X, y = linearly_separable
    p = Perceptron(activation="tanh")
    p.train(X, y, eta=0.1, epochs=1000)
    cm = p.confusion_matrix(X, y)
    assert cm.shape == (2, 2)


def test_precision_in_range(linearly_separable):
    X, y = linearly_separable
    p = Perceptron(activation="tanh")
    p.train(X, y, eta=0.1, epochs=1000)
    assert 0.0 <= p.precision(X, y) <= 1.0


def test_recall_in_range(linearly_separable):
    X, y = linearly_separable
    p = Perceptron(activation="tanh")
    p.train(X, y, eta=0.1, epochs=1000)
    assert 0.0 <= p.recall(X, y) <= 1.0


def test_f1_score_in_range(linearly_separable):
    X, y = linearly_separable
    p = Perceptron(activation="tanh")
    p.train(X, y, eta=0.1, epochs=1000)
    assert 0.0 <= p.f1_score(X, y) <= 1.0
