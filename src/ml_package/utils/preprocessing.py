"""Preprocessing helpers for splitting and scaling data."""

import copy

import numpy as np


def _validate_split_size(size, n_samples, name):
    if isinstance(size, float):
        if not 0.0 < size < 1.0:
            raise ValueError(f"{name} as a float must be between 0 and 1.")
        return int(np.ceil(size * n_samples))

    if isinstance(size, int):
        if not 0 < size < n_samples:
            raise ValueError(f"{name} as an int must be between 1 and n_samples - 1.")
        return size

    raise TypeError(f"{name} must be a float or int.")


def _safe_index(array, indices):
    array = np.asarray(array)
    return array[indices]


def _stratified_test_indices(y, n_test, rng):
    y = np.asarray(y)
    test_indices = []

    for label in np.unique(y):
        label_indices = np.where(y == label)[0]
        rng.shuffle(label_indices)
        label_test_size = int(round(len(label_indices) * n_test / len(y)))
        if label_test_size == 0 and len(label_indices) > 1:
            label_test_size = 1
        test_indices.extend(label_indices[:label_test_size])

    test_indices = np.array(test_indices, dtype=int)

    if len(test_indices) > n_test:
        test_indices = rng.choice(test_indices, size=n_test, replace=False)
    elif len(test_indices) < n_test:
        remaining = np.setdiff1d(np.arange(len(y)), test_indices, assume_unique=False)
        extra = rng.choice(remaining, size=n_test - len(test_indices), replace=False)
        test_indices = np.concatenate([test_indices, extra])

    rng.shuffle(test_indices)
    return test_indices


def train_test_split(X, y, test_size=0.25, random_state=None,
                     shuffle=True, stratify=None):
    """
    Split feature and target arrays into train and test sets.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        Feature matrix.
    y: array-like, shape (n_samples,)
        Target values or class labels.
    test_size: float or int, optional
        If float, proportion of samples assigned to the test set. If int,
        absolute number of test samples. Defaults to 0.25.
    random_state: int or None, optional
        Seed used when shuffling. Defaults to ``None``.
    shuffle: bool, optional
        Whether to shuffle samples before splitting. Defaults to ``True``.
    stratify: array-like or None, optional
        Class labels used to preserve class proportions in the split.
        Must be ``None`` when ``shuffle=False``. Defaults to ``None``.

    Returns
    -------
    X_train, X_test, y_train, y_test: tuple of numpy.ndarray
        Split feature and target arrays.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")
    if stratify is not None and not shuffle:
        raise ValueError("stratify must be None when shuffle=False.")

    n_samples = len(X)
    n_test = _validate_split_size(test_size, n_samples, "test_size")
    rng = np.random.default_rng(random_state)

    if stratify is not None:
        stratify = np.asarray(stratify)
        if len(stratify) != n_samples:
            raise ValueError("stratify must contain one label per sample.")
        test_indices = _stratified_test_indices(stratify, n_test, rng)
        train_indices = np.setdiff1d(np.arange(n_samples), test_indices)
        rng.shuffle(train_indices)
    else:
        indices = np.arange(n_samples)
        if shuffle:
            rng.shuffle(indices)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

    return (
        _safe_index(X, train_indices),
        _safe_index(X, test_indices),
        _safe_index(y, train_indices),
        _safe_index(y, test_indices),
    )


def train_val_test_split(X, y, val_size=0.15, test_size=0.15,
                         random_state=None, shuffle=True, stratify=None):
    """
    Split feature and target arrays into train, validation, and test sets.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        Feature matrix.
    y: array-like, shape (n_samples,)
        Target values or class labels.
    val_size: float or int, optional
        Validation set size. If float, interpreted as a proportion of the
        full dataset. Defaults to 0.15.
    test_size: float or int, optional
        Test set size. If float, interpreted as a proportion of the full
        dataset. Defaults to 0.15.
    random_state: int or None, optional
        Seed used when shuffling. Defaults to ``None``.
    shuffle: bool, optional
        Whether to shuffle samples before splitting. Defaults to ``True``.
    stratify: array-like or None, optional
        Class labels used to preserve class proportions across splits.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test: tuple of numpy.ndarray
        Split feature and target arrays.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples = len(X)
    n_test = _validate_split_size(test_size, n_samples, "test_size")
    n_val = _validate_split_size(val_size, n_samples, "val_size")

    if n_test + n_val >= n_samples:
        raise ValueError("val_size + test_size must leave at least one training sample.")

    X_remaining, X_test, y_remaining, y_test = train_test_split(
        X,
        y,
        test_size=n_test,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify,
    )

    remaining_stratify = y_remaining if stratify is not None else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_remaining,
        y_remaining,
        test_size=n_val,
        random_state=random_state,
        shuffle=shuffle,
        stratify=remaining_stratify,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def k_fold_split(X, y=None, n_splits=5, shuffle=True, random_state=None):
    """
    Generate train and validation indices for K-fold cross-validation.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        Feature matrix. Only the number of samples is used.
    y: array-like or None, optional
        Target values. Included for API consistency; not used by standard
        K-fold splitting. Defaults to ``None``.
    n_splits: int, optional
        Number of folds. Must be at least 2 and no greater than the number
        of samples. Defaults to 5.
    shuffle: bool, optional
        Whether to shuffle samples before assigning folds. Defaults to
        ``True``.
    random_state: int or None, optional
        Seed used when shuffling. Defaults to ``None``.

    Returns
    -------
    folds: list of (train_indices, val_indices) tuples
        Integer index arrays for each fold.
    """
    n_samples = len(X)
    if y is not None and len(y) != n_samples:
        raise ValueError("X and y must contain the same number of samples.")
    if not isinstance(n_splits, int):
        raise TypeError("n_splits must be an int.")
    if n_splits < 2 or n_splits > n_samples:
        raise ValueError("n_splits must be between 2 and n_samples.")

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1

    folds = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_indices, val_indices))
        current = stop

    return folds


def _sample_parameter_value(values, rng):
    if callable(values):
        return values(rng)
    if hasattr(values, "rvs"):
        return values.rvs(random_state=rng)

    values = list(values)
    if len(values) == 0:
        raise ValueError("Parameter choices cannot be empty.")
    return values[int(rng.integers(0, len(values)))]


def _make_estimator(estimator, params):
    if isinstance(estimator, type):
        return estimator(**params)

    model = copy.deepcopy(estimator)
    for name, value in params.items():
        setattr(model, name, value)
    return model


def _fit_estimator(model, X, y, fit_method, fit_params):
    if not hasattr(model, fit_method):
        raise AttributeError(f"Estimator does not have a '{fit_method}' method.")
    getattr(model, fit_method)(X, y, **fit_params)
    return model


def _score_estimator(model, X, y, scoring):
    if callable(scoring):
        return scoring(model, X, y)
    if not hasattr(model, scoring):
        raise AttributeError(f"Estimator does not have a '{scoring}' scoring method.")
    return getattr(model, scoring)(X, y)


def randomized_search_cv(estimator, param_distributions, X, y, n_iter=10,
                         n_splits=5, scoring="score", fit_method="fit",
                         fit_params=None, fit_param_distributions=None,
                         random_state=None, shuffle=True, refit=True):
    """
    Run randomized hyperparameter search with K-fold cross-validation.

    Parameters
    ----------
    estimator: class or object
        Estimator class such as ``KNN`` or an estimator instance. If a class
        is provided, sampled parameters are passed to the constructor. If an
        instance is provided, it is deep-copied and sampled parameters are set
        as attributes before fitting.
    param_distributions: dict
        Mapping of parameter names to possible values. Values can be lists,
        tuples, numpy arrays, callables accepting a random generator, or
        objects with an ``rvs`` method.
    X: array-like, shape (n_samples, n_features)
        Feature matrix.
    y: array-like, shape (n_samples,)
        Target values or class labels.
    n_iter: int, optional
        Number of random parameter configurations to evaluate. Defaults to 10.
    n_splits: int, optional
        Number of cross-validation folds. Defaults to 5.
    scoring: str or callable, optional
        Scoring method name on the estimator, or callable with signature
        ``scoring(model, X_val, y_val)``. Higher scores are considered better.
        Defaults to ``"score"``.
    fit_method: str, optional
        Name of the estimator method used for fitting. Defaults to ``"fit"``.
    fit_params: dict or None, optional
        Extra keyword arguments passed to the fitting method. Defaults to
        ``None``.
    fit_param_distributions: dict or None, optional
        Mapping of fitting-method parameter names to possible values. These
        are sampled for each random-search trial and merged with
        ``fit_params``. This is useful for estimators where hyperparameters
        such as learning rate or epochs are passed to ``train()`` rather than
        to ``__init__``. Defaults to ``None``.
    random_state: int or None, optional
        Seed used for parameter sampling and fold shuffling. Defaults to
        ``None``.
    shuffle: bool, optional
        Whether to shuffle before creating folds. Defaults to ``True``.
    refit: bool, optional
        Whether to fit the best estimator on the full dataset. Defaults to
        ``True``.

    Returns
    -------
    results: dict
        Dictionary containing ``best_estimator``, ``best_params``,
        ``best_fit_params``, ``best_score``, and ``cv_results``.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")
    if not isinstance(n_iter, int) or n_iter < 1:
        raise ValueError("n_iter must be a positive int.")

    fit_params = {} if fit_params is None else dict(fit_params)
    rng = np.random.default_rng(random_state)
    folds = k_fold_split(
        X,
        y,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    cv_results = []
    best_score = -np.inf
    best_params = None
    best_fit_params = None

    for _ in range(n_iter):
        params = {
            name: _sample_parameter_value(values, rng)
            for name, values in param_distributions.items()
        }
        sampled_fit_params = {
            name: _sample_parameter_value(values, rng)
            for name, values in (fit_param_distributions or {}).items()
        }
        trial_fit_params = {**fit_params, **sampled_fit_params}

        fold_scores = []
        for train_indices, val_indices in folds:
            model = _make_estimator(estimator, params)
            _fit_estimator(
                model,
                _safe_index(X, train_indices),
                _safe_index(y, train_indices),
                fit_method,
                trial_fit_params,
            )
            score = _score_estimator(
                model,
                _safe_index(X, val_indices),
                _safe_index(y, val_indices),
                scoring,
            )
            fold_scores.append(score)

        mean_score = float(np.mean(fold_scores))
        result = {
            "params": params,
            "fit_params": sampled_fit_params,
            "mean_score": mean_score,
            "fold_scores": fold_scores,
        }
        cv_results.append(result)

        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            best_fit_params = trial_fit_params

    best_estimator = None
    if refit:
        best_estimator = _make_estimator(estimator, best_params)
        _fit_estimator(best_estimator, X, y, fit_method, best_fit_params)

    return {
        "best_estimator": best_estimator,
        "best_params": best_params,
        "best_fit_params": best_fit_params,
        "best_score": best_score,
        "cv_results": cv_results,
    }


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.

    Methods
    -------
    fit(X)
        Learn feature means and standard deviations.
    transform(X)
        Apply the learned standardization.
    fit_transform(X)
        Learn scaling parameters and transform ``X``.
    inverse_transform(X)
        Undo the learned standardization.
    """

    def __init__(self):
        """Initialize an unfitted standard scaler."""
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """
        Learn feature means and standard deviations.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        self: StandardScaler
            The fitted scaler.
        """
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        return self

    def transform(self, X):
        """
        Apply the learned standardization.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        numpy.ndarray, shape (n_samples, n_features)
            Standardized feature matrix.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Call fit() before transform().")
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X):
        """
        Learn scaling parameters and transform ``X``.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        numpy.ndarray, shape (n_samples, n_features)
            Standardized feature matrix.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """
        Undo the learned standardization.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Standardized feature matrix.

        Returns
        -------
        numpy.ndarray, shape (n_samples, n_features)
            Feature matrix in the original scale.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Call fit() before inverse_transform().")
        return np.asarray(X, dtype=float) * self.scale_ + self.mean_


class MinMaxScaler:
    """
    Scale each feature to a target range.

    Parameters
    ----------
    feature_range: tuple of float, optional
        Desired output range. Defaults to ``(0, 1)``.
    """

    def __init__(self, feature_range=(0, 1)):
        """Initialize an unfitted min-max scaler."""
        self.feature_range = feature_range
        self.data_min_ = None
        self.data_max_ = None
        self.scale_ = None

    def fit(self, X):
        """
        Learn feature minimums and maximums.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        self: MinMaxScaler
            The fitted scaler.
        """
        low, high = self.feature_range
        if low >= high:
            raise ValueError("feature_range minimum must be less than maximum.")

        X = np.asarray(X, dtype=float)
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        data_range = self.data_max_ - self.data_min_
        data_range = np.where(data_range == 0, 1.0, data_range)
        self.scale_ = (high - low) / data_range
        return self

    def transform(self, X):
        """
        Apply min-max scaling.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        numpy.ndarray, shape (n_samples, n_features)
            Scaled feature matrix.
        """
        if self.data_min_ is None or self.scale_ is None:
            raise RuntimeError("Call fit() before transform().")

        low, _ = self.feature_range
        return (np.asarray(X, dtype=float) - self.data_min_) * self.scale_ + low

    def fit_transform(self, X):
        """
        Learn scaling parameters and transform ``X``.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        numpy.ndarray, shape (n_samples, n_features)
            Scaled feature matrix.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """
        Undo min-max scaling.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Scaled feature matrix.

        Returns
        -------
        numpy.ndarray, shape (n_samples, n_features)
            Feature matrix in the original scale.
        """
        if self.data_min_ is None or self.scale_ is None:
            raise RuntimeError("Call fit() before inverse_transform().")

        low, _ = self.feature_range
        return (np.asarray(X, dtype=float) - low) / self.scale_ + self.data_min_
