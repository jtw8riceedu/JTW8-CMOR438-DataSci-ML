import numpy as np
import pytest

from src.ml_package.supervised_learning.decision_tree import DecisionTreeRegressor
from src.ml_package.supervised_learning.knn import KNN
from src.ml_package.supervised_learning.random_forest import RandomForestRegressor
from src.ml_package.supervised_learning.voting import VotingRegressor
from src.ml_package.utils import regression_metrics as metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FixedRegressor:
    def __init__(self, predictions):
        self.predictions = np.array(predictions, dtype=float)

    def predict(self, X):
        return self.predictions[: len(X)]


def assert_shared_regression_methods(model, X, y):
    y_pred = model.predict(X)

    assert model.rmse(X, y) == metrics.rmse(y, y_pred)
    assert model.mae(X, y) == metrics.mae(y, y_pred)
    assert model.mape(X, y) == metrics.mape(y, y_pred)
    assert model.smape(X, y) == metrics.smape(y, y_pred)
    assert model.mase(X, y) == metrics.mase(y, y_pred)
    assert model.r_squared(X, y) == metrics.r_squared(y, y_pred)


# ---------------------------------------------------------------------------
# Model integration tests (unchanged)
# ---------------------------------------------------------------------------

def test_decision_tree_regressor_uses_shared_metrics():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    model = DecisionTreeRegressor(max_depth=2).fit(X, y)

    assert_shared_regression_methods(model, X, y)
    assert model.score(X, y) == model.r_squared(X, y)


def test_knn_regression_uses_shared_metrics():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    model = KNN(k=1, regression=True).fit(X, y)

    assert_shared_regression_methods(model, X, y)
    assert model.score(X, y) == metrics.mean_squared_error(y, model.predict(X))


def test_random_forest_regressor_uses_shared_metrics():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    model = RandomForestRegressor(
        n_estimators=3,
        max_depth=2,
        random_state=0,
    ).fit(X, y)

    assert_shared_regression_methods(model, X, y)
    assert model.score(X, y) == model.r_squared(X, y)


def test_voting_regressor_uses_shared_metrics():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    model = VotingRegressor(
        [
            ("low", FixedRegressor([1.0, 2.0, 3.0, 4.0])),
            ("high", FixedRegressor([2.0, 3.0, 4.0, 5.0])),
        ]
    )

    assert_shared_regression_methods(model, X, y)
    assert model.score(X, y) == model.r_squared(X, y)
    assert model.individual_scores(X, y)["ensemble"] == model.score(X, y)


# ---------------------------------------------------------------------------
# Unit tests for regression_metrics — mase
# ---------------------------------------------------------------------------

def test_mase_perfect_predictions_returns_zero():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = y_true.copy()
    assert metrics.mase(y_true, y_pred) == pytest.approx(0.0)


def test_mase_known_value():
    # naive baseline MAE = mean(|diff|) = 1.0; prediction MAE = 0.5 → MASE = 0.5
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.5, 2.5, 3.5, 4.5])
    expected = metrics.mae(y_true, y_pred) / np.mean(np.abs(np.diff(y_true)))
    assert metrics.mase(y_true, y_pred) == pytest.approx(expected)


def test_mase_worse_than_naive_is_above_one():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([10.0, 20.0, 30.0, 40.0])
    assert metrics.mase(y_true, y_pred) > 1.0


# ---------------------------------------------------------------------------
# Unit tests for regression_metrics — adjusted_r_squared
# ---------------------------------------------------------------------------

def test_adjusted_r_squared_perfect_fit_is_one():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = y_true.copy()
    assert metrics.adjusted_r_squared(y_true, y_pred, n_features=1) == pytest.approx(1.0)


def test_adjusted_r_squared_penalises_extra_features():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1, 5.9])
    adj1 = metrics.adjusted_r_squared(y_true, y_pred, n_features=1)
    adj3 = metrics.adjusted_r_squared(y_true, y_pred, n_features=3)
    assert adj1 > adj3


def test_adjusted_r_squared_consistent_with_formula():
    y_true = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    y_pred = np.array([2.1, 3.9, 6.1, 7.9, 10.1])
    n, p = len(y_true), 2
    r2 = metrics.r_squared(y_true, y_pred)
    expected = 1.0 - ((n - 1) / (n - p - 1)) * (1.0 - r2)
    assert metrics.adjusted_r_squared(y_true, y_pred, n_features=p) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Unit tests for regression_metrics — aicc
# ---------------------------------------------------------------------------

def test_aicc_greater_than_or_equal_aic():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1, 5.9, 7.1, 7.9])
    k = 2
    assert metrics.aicc(y_true, y_pred, k) >= metrics.aic(y_true, y_pred, k)


def test_aicc_converges_to_aic_for_large_n():
    rng = np.random.default_rng(42)
    n = 10_000
    y_true = rng.standard_normal(n)
    y_pred = y_true + rng.standard_normal(n) * 0.01
    k = 2
    assert metrics.aicc(y_true, y_pred, k) == pytest.approx(
        metrics.aic(y_true, y_pred, k), rel=1e-3
    )


def test_aicc_matches_known_formula():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1, 5.9])
    k = 1
    n = len(y_true)
    expected = metrics.aic(y_true, y_pred, k) + (2 * k * (k + 1)) / (n - k - 1)
    assert metrics.aicc(y_true, y_pred, k) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Edge cases — perfect predictions and degenerate inputs
# ---------------------------------------------------------------------------

def test_mse_zero_for_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0])
    assert metrics.mean_squared_error(y, y) == pytest.approx(0.0)


def test_rmse_zero_for_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0])
    assert metrics.rmse(y, y) == pytest.approx(0.0)


def test_mae_zero_for_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0])
    assert metrics.mae(y, y) == pytest.approx(0.0)


def test_r_squared_is_one_for_perfect_predictions():
    y = np.array([1.0, 3.0, 5.0, 7.0])
    assert metrics.r_squared(y, y) == pytest.approx(1.0)


def test_r_squared_zero_for_constant_predictor():
    # Predicting the mean always gives R² = 0
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.full_like(y_true, y_true.mean())
    assert metrics.r_squared(y_true, y_pred) == pytest.approx(0.0)


def test_smape_is_zero_for_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0])
    assert metrics.smape(y, y) == pytest.approx(0.0)


def test_smape_bounded_between_zero_and_200():
    rng = np.random.default_rng(0)
    y_true = rng.uniform(0.1, 10.0, 50)
    y_pred = rng.uniform(0.1, 10.0, 50)
    val = metrics.smape(y_true, y_pred)
    assert 0.0 <= val <= 200.0


def test_zero_variance_target_r_squared_does_not_raise():
    # All targets identical → SSTo = 0 → R² undefined.
    # The implementation should not raise; it may return ±inf or nan.
    y_true = np.array([3.0, 3.0, 3.0, 3.0])
    y_pred = np.array([3.1, 2.9, 3.0, 3.0])
    result = metrics.r_squared(y_true, y_pred)
    # Any non-exception result (including nan / -inf) is acceptable
    assert isinstance(float(result), float)