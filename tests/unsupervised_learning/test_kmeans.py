"""Tests for KMeans."""

import numpy as np
import pytest

from src.ml_package.unsupervised_learning.kmeans import KMeans


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clustered_data():
    """Three well-separated Gaussian blobs."""
    rng = np.random.default_rng(0)
    X0 = rng.standard_normal((50, 2)) + np.array([-6, 0])
    X1 = rng.standard_normal((50, 2)) + np.array([0, 6])
    X2 = rng.standard_normal((50, 2)) + np.array([6, 0])
    return np.vstack([X0, X1, X2])


@pytest.fixture
def fitted_kmeans(clustered_data):
    km = KMeans(n_clusters=3, random_state=0)
    km.fit(clustered_data)
    return km, clustered_data


# ---------------------------------------------------------------------------
# After fit — attributes
# ---------------------------------------------------------------------------

def test_centroids_shape(fitted_kmeans):
    km, X = fitted_kmeans
    assert km.centroids_.shape == (3, X.shape[1])


def test_labels_shape(fitted_kmeans):
    km, X = fitted_kmeans
    assert km.labels_.shape == (X.shape[0],)


def test_labels_values_in_range(fitted_kmeans):
    km, X = fitted_kmeans
    assert set(km.labels_).issubset(set(range(3)))


def test_inertia_nonnegative(fitted_kmeans):
    km, _ = fitted_kmeans
    assert km.inertia_ >= 0


def test_inertia_is_finite(fitted_kmeans):
    km, _ = fitted_kmeans
    assert np.isfinite(km.inertia_)


# ---------------------------------------------------------------------------
# fit_predict
# ---------------------------------------------------------------------------

def test_fit_predict_matches_labels(clustered_data):
    km = KMeans(n_clusters=3, random_state=1)
    labels = km.fit_predict(clustered_data)
    np.testing.assert_array_equal(labels, km.labels_)


def test_fit_predict_shape(clustered_data):
    km = KMeans(n_clusters=3, random_state=0)
    labels = km.fit_predict(clustered_data)
    assert labels.shape == (clustered_data.shape[0],)


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def test_predict_assigns_to_nearest_centroid(fitted_kmeans):
    km, X = fitted_kmeans
    preds = km.predict(X)
    assert preds.shape == (X.shape[0],)
    assert set(preds).issubset(set(range(3)))


def test_predict_new_point_near_cluster(fitted_kmeans):
    km, _ = fitted_kmeans
    # Point very close to cluster 0 centroid (should be assigned to it)
    new_point = km.centroids_[0:1] + 0.001
    pred = km.predict(new_point)
    assert pred[0] == 0


# ---------------------------------------------------------------------------
# Cluster quality
# ---------------------------------------------------------------------------

def test_three_clusters_found(fitted_kmeans):
    km, _ = fitted_kmeans
    assert len(np.unique(km.labels_)) == 3


def test_inertia_decreases_with_more_clusters(clustered_data):
    km2 = KMeans(n_clusters=2, random_state=0)
    km2.fit(clustered_data)
    km3 = KMeans(n_clusters=3, random_state=0)
    km3.fit(clustered_data)
    assert km3.inertia_ < km2.inertia_


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_reproducible_with_random_state(clustered_data):
    km1 = KMeans(n_clusters=3, random_state=42)
    km1.fit(clustered_data)
    km2 = KMeans(n_clusters=3, random_state=42)
    km2.fit(clustered_data)
    np.testing.assert_allclose(km1.centroids_, km2.centroids_)
