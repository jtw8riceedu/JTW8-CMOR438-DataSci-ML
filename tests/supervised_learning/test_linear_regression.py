"""Tests for LinearRegression."""

import numpy as np
import pytest

from src.ml_package.supervised_learning.linear_regression import LinearRegression


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_regression_data():
    """Simple linearly separable regression dataset: y = 2*x1 + 3*x2 + 1."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 2))
    y = 2 * X[:, 0] + 3 * X[:, 1] + 1
    return X, y


@pytest.fixture
def trained_model(simple_regression_data):
    X, y = simple_regression_data
    model = LinearRegression()
    model.train(X, y, eta=0.1, epochs=3000)
    return model, X, y


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_initial_weights_and_bias_are_none():
    model = LinearRegression()
    assert model.weights is None
    assert model.bias is None
    assert model.losses == []


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def test_train_sets_weights_and_bias(simple_regression_data):
    X, y = simple_regression_data
    model = LinearRegression()
    model.train(X, y, eta=0.01, epochs=10)
    assert model.weights is not None
    assert model.bias is not None


def test_train_weights_shape(simple_regression_data):
    X, y = simple_regression_data
    model = LinearRegression()
    model.train(X, y, eta=0.01, epochs=10)
    assert model.weights.shape == (X.shape[1],)


def test_train_records_losses(simple_regression_data):
    X, y = simple_regression_data
    model = LinearRegression()
    model.train(X, y, eta=0.01, epochs=50)
    assert len(model.losses) == 50


def test_train_loss_decreases(simple_regression_data):
    X, y = simple_regression_data
    model = LinearRegression()
    model.train(X, y, eta=0.05, epochs=500)
    assert model.losses[-1] < model.losses[0]


def test_train_returns_self(simple_regression_data):
    X, y = simple_regression_data
    model = LinearRegression()
    result = model.train(X, y, eta=0.01, epochs=10)
    assert result is model


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def test_predict_shape(trained_model):
    model, X, y = trained_model
    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)


def test_predict_close_to_truth(trained_model):
    model, X, y = trained_model
    preds = model.predict(X)
    assert np.mean((preds - y) ** 2) < 0.1


def test_feedforward_matches_predict(trained_model):
    model, X, y = trained_model
    np.testing.assert_array_equal(model.feedforward(X), model.predict(X))


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def test_rmse_nonnegative(trained_model):
    model, X, y = trained_model
    assert model.rmse(X, y) >= 0


def test_mae_nonnegative(trained_model):
    model, X, y = trained_model
    assert model.mae(X, y) >= 0


def test_mape_nonnegative(trained_model):
    model, X, y = trained_model
    assert model.mape(X, y) >= 0


def test_smape_nonnegative(trained_model):
    model, X, y = trained_model
    assert model.smape(X, y) >= 0


def test_mase_nonnegative(trained_model):
    model, X, y = trained_model
    assert model.mase(X, y) >= 0


def test_r_squared_near_one_for_good_fit(trained_model):
    model, X, y = trained_model
    assert model.r_squared(X, y) > 0.98


def test_r_squared_adj_near_one_for_good_fit(trained_model):
    model, X, y = trained_model
    assert model.r_squared_adj(X, y) > 0.98


def test_r_sqaured_adj_alias(trained_model):
    """Backward-compatible alias returns same value."""
    model, X, y = trained_model
    assert model.r_sqaured_adj(X, y) == model.r_squared_adj(X, y)


def test_aic_returns_finite(trained_model):
    model, X, y = trained_model
    assert np.isfinite(model.aic(X, y))


def test_aicc_returns_finite(trained_model):
    model, X, y = trained_model
    assert np.isfinite(model.aicc(X, y))


def test_bic_returns_finite(trained_model):
    model, X, y = trained_model
    assert np.isfinite(model.bic(X, y))


def test_rmse_less_than_or_equal_to_unscaled_mse(trained_model):
    """RMSE <= MSE only when MSE >= 1; for small MSE RMSE >= MSE.
    Simply verify RMSE == sqrt(MSE) implicitly via formula."""
    model, X, y = trained_model
    preds = model.predict(X)
    expected_rmse = np.sqrt(np.mean((preds - y) ** 2))
    assert np.isclose(model.rmse(X, y), expected_rmse)
