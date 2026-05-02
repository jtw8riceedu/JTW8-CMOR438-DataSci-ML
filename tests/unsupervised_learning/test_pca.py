"""Tests for PCA."""

import numpy as np
import pytest

from src.ml_package.unsupervised_learning.pca import PCA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def correlated_data():
    """4-feature dataset where variance is concentrated in 2 directions."""
    rng = np.random.default_rng(0)
    n = 100
    t = rng.standard_normal(n)
    X = np.column_stack([
        t + rng.standard_normal(n) * 0.01,
        2 * t + rng.standard_normal(n) * 0.01,
        t * 0.5 + rng.standard_normal(n) * 0.5,
        rng.standard_normal(n),
    ])
    return X


# ---------------------------------------------------------------------------
# Original test (preserved)
# ---------------------------------------------------------------------------

def test_pca_centers_data_without_internal_scaling():
    X = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
    ])
    pca = PCA(n_components=1)

    transformed = pca.fit_transform(X)
    centered = X - np.mean(X, axis=0)
    manual_projection = centered @ pca.components_

    assert transformed.shape == (4, 1)
    np.testing.assert_allclose(transformed, manual_projection)
    np.testing.assert_allclose(pca.mean_, np.mean(X, axis=0))
    assert not hasattr(pca, "std_")


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def test_fit_sets_mean(correlated_data):
    pca = PCA(n_components=2)
    pca.fit(correlated_data)
    assert pca.mean_.shape == (correlated_data.shape[1],)
    np.testing.assert_allclose(pca.mean_, np.mean(correlated_data, axis=0))


def test_fit_sets_components_shape(correlated_data):
    pca = PCA(n_components=2)
    pca.fit(correlated_data)
    assert pca.components_.shape == (correlated_data.shape[1], 2)


def test_explained_variance_shape(correlated_data):
    pca = PCA(n_components=2)
    pca.fit(correlated_data)
    assert pca.explained_variance_.shape == (2,)


def test_explained_variance_ratio_shape(correlated_data):
    pca = PCA(n_components=2)
    pca.fit(correlated_data)
    assert pca.explained_variance_ratio_.shape == (2,)


def test_explained_variance_ratio_sums_le_one(correlated_data):
    pca = PCA(n_components=2)
    pca.fit(correlated_data)
    assert np.sum(pca.explained_variance_ratio_) <= 1.0 + 1e-9


def test_explained_variance_ratio_positive(correlated_data):
    pca = PCA(n_components=2)
    pca.fit(correlated_data)
    assert np.all(pca.explained_variance_ratio_ >= 0)


def test_explained_variance_sorted_descending(correlated_data):
    pca = PCA(n_components=3)
    pca.fit(correlated_data)
    diffs = np.diff(pca.explained_variance_)
    assert np.all(diffs <= 1e-10)


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def test_transform_shape(correlated_data):
    pca = PCA(n_components=2)
    pca.fit(correlated_data)
    transformed = pca.transform(correlated_data)
    assert transformed.shape == (correlated_data.shape[0], 2)


def test_fit_transform_equals_fit_then_transform(correlated_data):
    pca1 = PCA(n_components=2)
    T1 = pca1.fit_transform(correlated_data)

    pca2 = PCA(n_components=2)
    pca2.fit(correlated_data)
    T2 = pca2.transform(correlated_data)

    np.testing.assert_allclose(T1, T2)


def test_transformed_mean_near_zero(correlated_data):
    pca = PCA(n_components=2)
    T = pca.fit_transform(correlated_data)
    np.testing.assert_allclose(np.mean(T, axis=0), np.zeros(2), atol=1e-10)


def test_single_component_output(correlated_data):
    pca = PCA(n_components=1)
    T = pca.fit_transform(correlated_data)
    assert T.shape == (correlated_data.shape[0], 1)


def test_full_components_ratio_sums_to_one(correlated_data):
    n_features = correlated_data.shape[1]
    pca = PCA(n_components=n_features)
    pca.fit(correlated_data)
    np.testing.assert_allclose(
        np.sum(pca.explained_variance_ratio_), 1.0, atol=1e-10
    )
