import numpy as np

from src.ml_package.utils import regression_metrics as metrics


def test_regression_error_metrics():
    y_true = np.array([2.0, 4.0, 6.0])
    y_pred = np.array([1.0, 5.0, 6.0])

    assert metrics.mean_squared_error(y_true, y_pred) == np.mean([1.0, 1.0, 0.0])
    assert metrics.rmse(y_true, y_pred) == np.sqrt(np.mean([1.0, 1.0, 0.0]))
    assert metrics.mae(y_true, y_pred) == np.mean([1.0, 1.0, 0.0])
    assert metrics.mape(y_true, y_pred) == np.mean([0.5, 0.25, 0.0]) * 100
    assert metrics.smape(y_true, y_pred) == np.mean([2 / 3, 2 / 9, 0.0]) * 100


def test_regression_model_selection_metrics():
    y_true = np.array([1.0, 2.0, 4.0, 8.0])
    y_pred = np.array([1.0, 3.0, 5.0, 7.0])
    sse = 3.0
    ssto = np.sum((y_true - np.mean(y_true)) ** 2)

    assert metrics.r_squared(y_true, y_pred) == 1.0 - sse / ssto
    assert metrics.adjusted_r_squared(y_true, y_pred, n_features=1) == (
        1.0 - ((4 - 1) / (4 - 1 - 1)) * (sse / ssto)
    )
    assert metrics.aic(y_true, y_pred, n_parameters=2) == 4 * np.log(sse / 4) + 4
    assert metrics.bic(y_true, y_pred, n_parameters=2) == 4 * np.log(sse / 4) + 2 * np.log(4)
