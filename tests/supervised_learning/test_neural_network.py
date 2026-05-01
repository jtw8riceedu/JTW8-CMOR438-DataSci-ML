import numpy as np
import pytest

from src.ml_package.supervised_learning import CrossEntropy, MSE, NeuralNetwork


def test_regression_network_feedforward_returns_expected_shape():
    np.random.seed(0)
    network = NeuralNetwork([2, 3, 1], task="regression")
    x = np.array([[1.0], [2.0]])

    output = network.feedforward(x)

    assert output.shape == (1, 1)


def test_default_cost_matches_task_type():
    regression_network = NeuralNetwork([2, 3, 1], task="regression")
    binary_network = NeuralNetwork([2, 3, 1], task="binary")
    multiclass_network = NeuralNetwork([2, 3, 2], task="multiclass")

    assert isinstance(regression_network.cost, MSE)
    assert isinstance(binary_network.cost, CrossEntropy)
    assert isinstance(multiclass_network.cost, CrossEntropy)


def test_invalid_task_raises_value_error():
    with pytest.raises(ValueError, match="Task must be one of"):
        NeuralNetwork([2, 3, 1], task="clustering")


def test_mini_batch_update_changes_weights():
    np.random.seed(0)
    network = NeuralNetwork([2, 3, 1], task="regression")
    original_weights = [weight.copy() for weight in network.weights]
    mini_batch = [
        (np.array([[0.0], [1.0]]), np.array([[1.0]])),
        (np.array([[1.0], [0.0]]), np.array([[0.0]])),
    ]

    network.update_mini_batch(mini_batch, eta=0.1)

    assert any(
        not np.allclose(original, updated)
        for original, updated in zip(original_weights, network.weights)
    )


def test_binary_classification_evaluation_counts_correct_predictions():
    np.random.seed(0)
    network = NeuralNetwork([1, 1], task="binary")
    network.weights = [np.array([[10.0]])]
    network.biases = [np.array([[-5.0]])]
    test_data = [
        (np.array([[0.0]]), np.array([[0.0]])),
        (np.array([[1.0]]), np.array([[1.0]])),
    ]

    assert network.evaluate_classification(test_data) == 2
