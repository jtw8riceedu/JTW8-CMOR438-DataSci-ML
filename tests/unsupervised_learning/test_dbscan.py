"""Tests for DBSCAN."""

import numpy as np
import pytest

from src.ml_package.unsupervised_learning.dbscan import DBSCAN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_blobs():
    """Two dense blobs that DBSCAN should easily separate."""
    rng = np.random.default_rng(0)
    X0 = rng.standard_normal((30, 2)) * 0.3 + np.array([-4, 0])
    X1 = rng.standard_normal((30, 2)) * 0.3 + np.array([4, 0])
    return np.vstack([X0, X1])


@pytest.fixture
def noise_data():
    """Points spread uniformly — all noise for tight eps."""
    rng = np.random.default_rng(1)
    return rng.uniform(-10, 10, (20, 2))


# ---------------------------------------------------------------------------
# Basic attributes after fit
# ---------------------------------------------------------------------------

def test_labels_set_after_fit(two_blobs):
    db = DBSCAN(eps=1.0, min_samples=3)
    db.fit(two_blobs)
    assert db.labels_ is not None


def test_labels_shape(two_blobs):
    db = DBSCAN(eps=1.0, min_samples=3)
    db.fit(two_blobs)
    assert db.labels_.shape == (two_blobs.shape[0],)


def test_labels_are_integer(two_blobs):
    db = DBSCAN(eps=1.0, min_samples=3)
    db.fit(two_blobs)
    assert db.labels_.dtype in (np.float64, np.int64, np.int32, float, int)


# ---------------------------------------------------------------------------
# fit_predict
# ---------------------------------------------------------------------------

def test_fit_predict_returns_labels(two_blobs):
    db = DBSCAN(eps=1.0, min_samples=3)
    labels = db.fit_predict(two_blobs)
    np.testing.assert_array_equal(labels, db.labels_)


def test_fit_predict_shape(two_blobs):
    db = DBSCAN(eps=1.0, min_samples=3)
    labels = db.fit_predict(two_blobs)
    assert labels.shape == (two_blobs.shape[0],)


# ---------------------------------------------------------------------------
# Cluster detection
# ---------------------------------------------------------------------------

def test_two_blobs_finds_two_clusters(two_blobs):
    db = DBSCAN(eps=0.8, min_samples=3)
    db.fit(two_blobs)
    cluster_ids = set(db.labels_[db.labels_ >= 0])
    assert len(cluster_ids) == 2


def test_noise_points_labeled_minus_one(noise_data):
    """Very tight eps → all points should be noise."""
    db = DBSCAN(eps=0.01, min_samples=10)
    db.fit(noise_data)
    assert np.all(db.labels_ == -1)


def test_single_dense_blob_one_cluster():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((50, 2)) * 0.2
    db = DBSCAN(eps=1.0, min_samples=3)
    db.fit(X)
    cluster_ids = set(db.labels_[db.labels_ >= 0])
    assert len(cluster_ids) == 1


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

def test_larger_eps_fewer_noise_points(two_blobs):
    db_tight = DBSCAN(eps=0.3, min_samples=3)
    db_loose = DBSCAN(eps=2.0, min_samples=3)
    db_tight.fit(two_blobs)
    db_loose.fit(two_blobs)
    noise_tight = np.sum(db_tight.labels_ == -1)
    noise_loose = np.sum(db_loose.labels_ == -1)
    assert noise_loose <= noise_tight


def test_larger_min_samples_more_noise(two_blobs):
    db_few = DBSCAN(eps=0.8, min_samples=2)
    db_many = DBSCAN(eps=0.8, min_samples=30)
    db_few.fit(two_blobs)
    db_many.fit(two_blobs)
    noise_few = np.sum(db_few.labels_ == -1)
    noise_many = np.sum(db_many.labels_ == -1)
    assert noise_many >= noise_few
