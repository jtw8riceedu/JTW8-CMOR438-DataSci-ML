"""Tests for HardVotingClassifier and VotingRegressor."""

import numpy as np
import pytest

from src.ml_package.supervised_learning.voting import (
    HardVotingClassifier,
    VotingRegressor,
)
from src.ml_package.supervised_learning.decision_tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)
from src.ml_package.supervised_learning.knn import KNN


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clf_data():
    X = np.array([
        [1.0, 2.0], [2.0, 3.0], [3.0, 1.0],
        [6.0, 7.0], [7.0, 6.0], [8.0, 8.0],
        [1.5, 2.5], [7.5, 6.5],
    ])
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    return X, y


@pytest.fixture
def reg_data():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((60, 2))
    y = X[:, 0] * 2 + X[:, 1] + rng.standard_normal(60) * 0.1
    return X, y


def _make_clf_ensemble(X, y):
    t1 = DecisionTreeClassifier(max_depth=2).fit(X, y)
    t2 = DecisionTreeClassifier(max_depth=3).fit(X, y)
    t3 = KNN(k=3).fit(X, y)
    return HardVotingClassifier([("t1", t1), ("t2", t2), ("t3", t3)])


def _make_reg_ensemble(X, y):
    t1 = DecisionTreeRegressor(max_depth=2).fit(X, y)
    t2 = DecisionTreeRegressor(max_depth=4).fit(X, y)
    t3 = KNN(k=3, regression=True).fit(X, y)
    return VotingRegressor([("t1", t1), ("t2", t2), ("t3", t3)])


# ---------------------------------------------------------------------------
# HardVotingClassifier — initialization
# ---------------------------------------------------------------------------

def test_voting_clf_empty_models_raises():
    with pytest.raises((ValueError, Exception)):
        HardVotingClassifier([]).predict(np.ones((2, 2)))


# ---------------------------------------------------------------------------
# HardVotingClassifier — prediction
# ---------------------------------------------------------------------------

def test_voting_clf_predict_shape(clf_data):
    X, y = clf_data
    vc = _make_clf_ensemble(X, y)
    preds = vc.predict(X)
    assert preds.shape == (len(y),)


def test_voting_clf_predict_valid_classes(clf_data):
    X, y = clf_data
    vc = _make_clf_ensemble(X, y)
    preds = vc.predict(X)
    assert set(preds).issubset({0, 1})


def test_voting_clf_accuracy_above_threshold(clf_data):
    X, y = clf_data
    vc = _make_clf_ensemble(X, y)
    assert vc.accuracy(X, y) >= 0.75


def test_voting_clf_score_equals_accuracy(clf_data):
    X, y = clf_data
    vc = _make_clf_ensemble(X, y)
    assert vc.score(X, y) == vc.accuracy(X, y)


# ---------------------------------------------------------------------------
# HardVotingClassifier — evaluation
# ---------------------------------------------------------------------------

def test_voting_clf_confusion_matrix_shape(clf_data):
    X, y = clf_data
    vc = _make_clf_ensemble(X, y)
    cm = vc.confusion_matrix(X, y)
    assert cm.shape == (2, 2)


def test_voting_clf_confusion_matrix_sum(clf_data):
    X, y = clf_data
    vc = _make_clf_ensemble(X, y)
    cm = vc.confusion_matrix(X, y)
    assert cm.sum() == len(y)


def test_voting_clf_precision_in_range(clf_data):
    X, y = clf_data
    vc = _make_clf_ensemble(X, y)
    assert 0.0 <= vc.precision(X, y) <= 1.0


def test_voting_clf_recall_in_range(clf_data):
    X, y = clf_data
    vc = _make_clf_ensemble(X, y)
    assert 0.0 <= vc.recall(X, y) <= 1.0


def test_voting_clf_f1_in_range(clf_data):
    X, y = clf_data
    vc = _make_clf_ensemble(X, y)
    assert 0.0 <= vc.f1_score(X, y) <= 1.0


def test_voting_clf_individual_scores_returns_dict(clf_data):
    X, y = clf_data
    vc = _make_clf_ensemble(X, y)
    scores = vc.individual_scores(X, y)
    assert isinstance(scores, dict)
    assert "t1" in scores
    assert "t2" in scores
    assert "t3" in scores


# ---------------------------------------------------------------------------
# VotingRegressor — prediction
# ---------------------------------------------------------------------------

def test_voting_reg_predict_shape(reg_data):
    X, y = reg_data
    vr = _make_reg_ensemble(X, y)
    preds = vr.predict(X)
    assert preds.shape == (len(y),)


def test_voting_reg_score_returns_float(reg_data):
    X, y = reg_data
    vr = _make_reg_ensemble(X, y)
    score = vr.score(X, y)
    assert isinstance(float(score), float)


# ---------------------------------------------------------------------------
# VotingRegressor — evaluation metrics
# ---------------------------------------------------------------------------

def test_voting_reg_rmse_nonnegative(reg_data):
    X, y = reg_data
    vr = _make_reg_ensemble(X, y)
    assert vr.rmse(X, y) >= 0


def test_voting_reg_mae_nonnegative(reg_data):
    X, y = reg_data
    vr = _make_reg_ensemble(X, y)
    assert vr.mae(X, y) >= 0


def test_voting_reg_r_squared_finite(reg_data):
    X, y = reg_data
    vr = _make_reg_ensemble(X, y)
    assert np.isfinite(vr.r_squared(X, y))


def test_voting_reg_individual_scores_returns_dict(reg_data):
    X, y = reg_data
    vr = _make_reg_ensemble(X, y)
    scores = vr.individual_scores(X, y)
    assert isinstance(scores, dict)


# ---------------------------------------------------------------------------
# VotingRegressor — weighted ensemble
# ---------------------------------------------------------------------------

def test_voting_reg_weighted_predict_shape(reg_data):
    X, y = reg_data
    t1 = DecisionTreeRegressor(max_depth=3).fit(X, y)
    t2 = DecisionTreeRegressor(max_depth=3).fit(X, y)
    vr = VotingRegressor([("t1", t1), ("t2", t2)], weights=[2.0, 1.0])
    preds = vr.predict(X)
    assert preds.shape == (len(y),)
