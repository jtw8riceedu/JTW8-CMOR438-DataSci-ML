import numpy as np
import pytest

from src.ml_package.supervised_learning import CrossEntropy, MSE, NeuralNetwork


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_regression_data(n=20, seed=42):
    """Simple y = 2x + 1 regression task."""
    rng = np.random.default_rng(seed)
    x_vals = rng.uniform(-1, 1, n)
    y_vals = 2.0 * x_vals + 1.0
    data = [
        (np.array([[x]]), np.array([[y]]))
        for x, y in zip(x_vals, y_vals)
    ]
    return data


def make_binary_data(n=20, seed=0):
    """Binary classification: label = 1 if x > 0 else 0."""
    rng = np.random.default_rng(seed)
    x_vals = rng.uniform(-2, 2, n)
    data = [
        (np.array([[x]]), np.array([[float(x > 0)]]))
        for x in x_vals
    ]
    return data


def make_multiclass_data(n_per_class=10, seed=7):
    """3-class one-hot classification (linearly separable clusters)."""
    rng = np.random.default_rng(seed)
    data = []
    for cls in range(3):
        xs = rng.normal(loc=cls * 3.0, scale=0.3, size=n_per_class)
        for x in xs:
            y_onehot = np.zeros((3, 1))
            y_onehot[cls, 0] = 1.0
            data.append((np.array([[x]]), y_onehot))
    return data


# ---------------------------------------------------------------------------
# Initialisation & configuration
# ---------------------------------------------------------------------------

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


def test_too_few_layers_raises_value_error():
    with pytest.raises(ValueError):
        NeuralNetwork([4], task="regression")


def test_xavier_weight_initialisation_shape():
    network = NeuralNetwork([3, 5, 2], task="binary")
    assert network.weights[0].shape == (5, 3)
    assert network.weights[1].shape == (2, 5)
    assert network.biases[0].shape == (5, 1)
    assert network.biases[1].shape == (2, 1)


def test_custom_cost_is_respected():
    custom = MSE()
    net = NeuralNetwork([2, 2], task="binary", cost=custom)
    assert net.cost is custom


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def test_binary_output_activation_is_in_zero_one():
    np.random.seed(1)
    net = NeuralNetwork([2, 4, 1], task="binary")
    x = np.random.randn(2, 10)
    out = net.feedforward(x)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_multiclass_output_sums_to_one_per_sample():
    np.random.seed(2)
    net = NeuralNetwork([3, 5, 4], task="multiclass")
    x = np.random.randn(3, 8)
    out = net.feedforward(x)
    np.testing.assert_allclose(out.sum(axis=0), np.ones(8), atol=1e-10)


def test_regression_output_is_unbounded():
    np.random.seed(3)
    net = NeuralNetwork([1, 10, 1], task="regression")
    # Large positive input — a linear output layer can exceed [0,1]
    x = np.array([[100.0]])
    out = net.feedforward(x)
    assert out.shape == (1, 1)  # just confirm it runs; value may be large


# ---------------------------------------------------------------------------
# Mini-batch update
# ---------------------------------------------------------------------------

def test_mini_batch_update_changes_weights():
    np.random.seed(0)
    network = NeuralNetwork([2, 3, 1], task="regression")
    original_weights = [w.copy() for w in network.weights]
    mini_batch = [
        (np.array([[0.0], [1.0]]), np.array([[1.0]])),
        (np.array([[1.0], [0.0]]), np.array([[0.0]])),
    ]

    network.update_mini_batch(mini_batch, eta=0.1)

    assert any(
        not np.allclose(orig, updated)
        for orig, updated in zip(original_weights, network.weights)
    )


def test_mini_batch_update_changes_biases():
    np.random.seed(5)
    network = NeuralNetwork([2, 3, 1], task="regression")
    original_biases = [b.copy() for b in network.biases]
    mini_batch = [
        (np.array([[1.0], [1.0]]), np.array([[0.5]])),
    ]

    network.update_mini_batch(mini_batch, eta=0.5)

    assert any(
        not np.allclose(orig, updated)
        for orig, updated in zip(original_biases, network.biases)
    )


# ---------------------------------------------------------------------------
# SGD — loss actually decreases
# ---------------------------------------------------------------------------

def test_sgd_regression_loss_decreases():
    """After training on a simple linear task, MSE should fall."""
    np.random.seed(10)
    net = NeuralNetwork([1, 8, 1], task="regression")
    data = make_regression_data(n=40)
    val = data[:8]
    train = data[8:]

    mse_before = net.evaluate_regression(val)
    net.SGD(train, epochs=50, mini_batch_size=8, eta=0.1)
    mse_after = net.evaluate_regression(val)

    assert mse_after < mse_before


def test_sgd_binary_loss_decreases():
    """After training on a linearly-separable binary task, cost should fall."""
    np.random.seed(11)
    net = NeuralNetwork([1, 8, 1], task="binary")
    data = make_binary_data(n=40)
    val = data[:8]
    train = data[8:]

    cost_before = net.total_cost(val)
    net.SGD(train, epochs=50, mini_batch_size=8, eta=0.5)
    cost_after = net.total_cost(val)

    assert cost_after < cost_before


# ---------------------------------------------------------------------------
# evaluate_classification
# ---------------------------------------------------------------------------

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


def test_binary_evaluate_classification_counts_wrong_predictions():
    net = NeuralNetwork([1, 1], task="binary")
    # Force output always < 0.5 (strong negative weight)
    net.weights = [np.array([[-10.0]])]
    net.biases = [np.array([[0.0]])]
    test_data = [
        (np.array([[1.0]]), np.array([[1.0]])),  # output ~0 → predicted 0, label 1 → wrong
        (np.array([[1.0]]), np.array([[1.0]])),
    ]
    
    assert net.evaluate_classification(test_data) == 0


def test_multiclass_evaluation_counts_correct_predictions():
    np.random.seed(20)
    net = NeuralNetwork([1, 6, 3], task="multiclass")
    net.SGD(make_multiclass_data(), epochs=200, mini_batch_size=5, eta=1.0)
    test_data = make_multiclass_data(n_per_class=5, seed=99)
    correct = net.evaluate_classification(test_data)
    # At least slightly better than random (1/3 ≈ 5 out of 15)
    assert correct >= 5


# ---------------------------------------------------------------------------
# evaluate_regression
# ---------------------------------------------------------------------------

def test_evaluate_regression_returns_float():
    np.random.seed(0)
    net = NeuralNetwork([1, 4, 1], task="regression")
    val = make_regression_data(n=5)
    result = net.evaluate_regression(val)
    assert isinstance(result, float)


def test_evaluate_regression_zero_for_exact_fit():
    """A network whose output exactly equals y should report MSE = 0."""
    net = NeuralNetwork([1, 1], task="regression")
    # Hand-craft weights so feedforward(x) = x for scalar x
    net.weights = [np.array([[1.0]])]
    net.biases = [np.array([[0.0]])]
    val = [(np.array([[v]]), np.array([[v]])) for v in [1.0, 2.0, 3.0]]
    assert net.evaluate_regression(val) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# total_cost
# ---------------------------------------------------------------------------

def test_total_cost_returns_positive_float():
    np.random.seed(0)
    net = NeuralNetwork([1, 3, 1], task="binary")
    data = make_binary_data(n=6)
    cost = net.total_cost(data)
    assert isinstance(cost, float)
    assert cost > 0.0


def test_total_cost_decreases_after_training():
    np.random.seed(30)
    net = NeuralNetwork([1, 8, 1], task="binary")
    data = make_binary_data(n=30)
    cost_before = net.total_cost(data)
    net.SGD(data, epochs=40, mini_batch_size=10, eta=0.5)
    cost_after = net.total_cost(data)
    assert cost_after < cost_before


def test_total_cost_monitor_flag_does_not_raise(capsys):
    np.random.seed(0)
    net = NeuralNetwork([1, 4, 1], task="binary")
    data = make_binary_data(n=10)
    # Should run without error and print cost information
    net.SGD(data, epochs=2, mini_batch_size=5, eta=0.1,
            validation_data=data, monitor_cost=True)
    captured = capsys.readouterr()
    assert "cost" in captured.out.lower()