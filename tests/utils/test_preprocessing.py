import numpy as np
import pytest

from src.ml_package.utils.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    k_fold_split,
    randomized_search_cv,
    train_test_split,
    train_val_test_split,
)


class ThresholdClassifier:
    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y):
        return self

    def score(self, X, y):
        predictions = (np.asarray(X)[:, 0] >= self.threshold).astype(int)
        return np.mean(predictions == y)


class TrainParamClassifier:
    def __init__(self):
        self.threshold = None

    def train(self, X, y, threshold=0.0):
        self.threshold = threshold
        return self

    def accuracy(self, X, y):
        predictions = (np.asarray(X)[:, 0] >= self.threshold).astype(int)
        return np.mean(predictions == y)


def test_train_test_split_is_reproducible():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    split_one = train_test_split(X, y, test_size=0.3, random_state=42)
    split_two = train_test_split(X, y, test_size=0.3, random_state=42)

    for first, second in zip(split_one, split_two):
        np.testing.assert_array_equal(first, second)

    X_train, X_test, y_train, y_test = split_one
    assert X_train.shape == (7, 2)
    assert X_test.shape == (3, 2)
    assert y_train.shape == (7,)
    assert y_test.shape == (3,)


def test_train_test_split_preserves_order_without_shuffle():
    X = np.arange(10).reshape(5, 2)
    y = np.arange(5)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=2,
        shuffle=False,
    )

    np.testing.assert_array_equal(X_test, X[:2])
    np.testing.assert_array_equal(y_test, y[:2])
    np.testing.assert_array_equal(X_train, X[2:])
    np.testing.assert_array_equal(y_train, y[2:])


def test_train_test_split_supports_stratify():
    X = np.arange(20).reshape(10, 2)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    _, _, _, y_test = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=0,
        stratify=y,
    )

    values, counts = np.unique(y_test, return_counts=True)
    assert dict(zip(values, counts)) == {0: 2, 1: 2}


def test_train_val_test_split_returns_expected_shapes():
    X = np.arange(40).reshape(20, 2)
    y = np.arange(20)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X,
        y,
        val_size=0.2,
        test_size=0.2,
        random_state=1,
    )

    assert X_train.shape == (12, 2)
    assert X_val.shape == (4, 2)
    assert X_test.shape == (4, 2)
    assert y_train.shape == (12,)
    assert y_val.shape == (4,)
    assert y_test.shape == (4,)


def test_k_fold_split_covers_each_sample_once_as_validation():
    X = np.arange(10).reshape(5, 2)
    y = np.arange(5)

    folds = k_fold_split(X, y, n_splits=3, shuffle=False)

    assert len(folds) == 3
    validation_indices = np.concatenate([val for _, val in folds])
    np.testing.assert_array_equal(np.sort(validation_indices), np.arange(5))

    for train_indices, val_indices in folds:
        assert set(train_indices).isdisjoint(set(val_indices))


def test_randomized_search_cv_finds_best_parameter():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    results = randomized_search_cv(
        ThresholdClassifier,
        {"threshold": [1.5, 2.5, 4.5]},
        X,
        y,
        n_iter=3,
        n_splits=3,
        random_state=0,
        shuffle=False,
    )

    assert results["best_params"] == {"threshold": 2.5}
    assert results["best_score"] == 1.0
    assert isinstance(results["best_estimator"], ThresholdClassifier)
    assert len(results["cv_results"]) == 3


def test_randomized_search_cv_can_sample_fit_params():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    results = randomized_search_cv(
        TrainParamClassifier,
        {},
        X,
        y,
        n_iter=3,
        n_splits=3,
        fit_method="train",
        fit_param_distributions={"threshold": [1.5, 2.5, 4.5]},
        scoring="accuracy",
        random_state=0,
        shuffle=False,
    )

    assert results["best_params"] == {}
    assert results["best_fit_params"] == {"threshold": 2.5}
    assert results["best_score"] == 1.0
    assert results["best_estimator"].threshold == 2.5
    assert results["cv_results"][0]["fit_params"]


def test_standard_scaler_fit_transform_and_inverse_transform():
    X = np.array([[1.0, 10.0], [2.0, 10.0], [3.0, 10.0]])
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    np.testing.assert_allclose(np.mean(X_scaled, axis=0), np.array([0.0, 0.0]))
    np.testing.assert_allclose(np.std(X_scaled, axis=0), np.array([1.0, 0.0]))
    np.testing.assert_allclose(scaler.inverse_transform(X_scaled), X)


def test_min_max_scaler_fit_transform_and_inverse_transform():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    scaler = MinMaxScaler(feature_range=(-1, 1))

    X_scaled = scaler.fit_transform(X)

    np.testing.assert_allclose(np.min(X_scaled, axis=0), np.array([-1.0, -1.0]))
    np.testing.assert_allclose(np.max(X_scaled, axis=0), np.array([1.0, 1.0]))
    np.testing.assert_allclose(scaler.inverse_transform(X_scaled), X)


def test_scalers_require_fit_before_transform():
    with pytest.raises(RuntimeError, match="Call fit"):
        StandardScaler().transform([[1.0]])

    with pytest.raises(RuntimeError, match="Call fit"):
        MinMaxScaler().transform([[1.0]])
