"""Regression metric helpers."""

import numpy as np


def mean_squared_error(y_true, y_pred):
    """Compute mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean((y_pred - y_true) ** 2)


def rmse(y_true, y_pred):
    """Compute root mean squared error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    """Compute mean absolute error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs(y_pred - y_true))


def mape(y_true, y_pred):
    """Compute mean absolute percentage error as a percentage."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def smape(y_true, y_pred):
    """Compute symmetric mean absolute percentage error as a percentage."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true) + np.abs(y_pred)
    return np.mean(np.nan_to_num(2.0 * np.abs(y_pred - y_true) / denominator)) * 100


def mase(y_true, y_pred):
    """Compute mean absolute scaled error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    naive_base = np.mean(np.abs(np.diff(y_true)))
    return mae(y_true, y_pred) / naive_base


def r_squared(y_true, y_pred):
    """Compute the coefficient of determination."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sse = np.sum((y_true - y_pred) ** 2)
    ssto = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - sse / ssto


def adjusted_r_squared(y_true, y_pred, n_features):
    """Compute adjusted R-squared."""
    y_true = np.asarray(y_true, dtype=float)
    n = y_true.shape[0]
    return 1.0 - ((n - 1) / (n - n_features - 1)) * (1.0 - r_squared(y_true, y_pred))


def aic(y_true, y_pred, n_parameters):
    """Compute Akaike's Information Criterion."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = y_true.shape[0]
    sse = np.sum((y_true - y_pred) ** 2)
    return n * np.log(sse / n) + 2 * n_parameters


def aicc(y_true, y_pred, n_parameters):
    """Compute corrected Akaike's Information Criterion."""
    y_true = np.asarray(y_true, dtype=float)
    n = y_true.shape[0]
    return (
        aic(y_true, y_pred, n_parameters)
        + (2 * n_parameters * (n_parameters + 1)) / (n - n_parameters - 1)
    )


def bic(y_true, y_pred, n_parameters):
    """Compute Bayesian Information Criterion."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = y_true.shape[0]
    sse = np.sum((y_true - y_pred) ** 2)
    return n * np.log(sse / n) + n_parameters * np.log(n)
