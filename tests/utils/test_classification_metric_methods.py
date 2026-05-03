import numpy as np
import pytest

from src.ml_package.supervised_learning.decision_tree import DecisionTreeClassifier
from src.ml_package.supervised_learning.knn import KNN
from src.ml_package.supervised_learning.logistic_regression import LogisticRegression
from src.ml_package.supervised_learning.perceptron import Perceptron
from src.ml_package.supervised_learning.random_forest import RandomForestClassifier
from src.ml_package.supervised_learning.voting import HardVotingClassifier
from src.ml_package.utils import classification_metrics as metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FixedClassifier:
    def __init__(self, predictions):
        self.predictions = np.array(predictions)

    def predict(self, X):
        return self.predictions[: len(X)]


def assert_shared_classification_methods(model, X, y):
    y_pred = model.predict(X)

    assert model.accuracy(X, y) == metrics.accuracy(y, y_pred)
    assert model.precision(X, y, average="macro") == metrics.precision(
        y, y_pred, average="macro"
    )
    assert model.recall(X, y, average="macro") == metrics.recall(
        y, y_pred, average="macro"
    )
    assert model.f1_score(X, y, average="macro") == metrics.f1_score(
        y, y_pred, average="macro"
    )
    np.testing.assert_array_equal(
        model.confusion_matrix(X, y),
        metrics.confusion_matrix(y, y_pred),
    )


# ---------------------------------------------------------------------------
# Model integration tests (unchanged)
# ---------------------------------------------------------------------------

def test_decision_tree_classifier_uses_shared_metrics():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = DecisionTreeClassifier(max_depth=2).fit(X, y)

    assert_shared_classification_methods(model, X, y)
    assert model.score(X, y) == model.accuracy(X, y)


def test_knn_classifier_uses_shared_metrics():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = KNN(k=1).fit(X, y)

    assert_shared_classification_methods(model, X, y)
    assert model.score(X, y) == model.accuracy(X, y)


def test_random_forest_classifier_uses_shared_metrics():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = RandomForestClassifier(
        n_estimators=3,
        max_depth=2,
        random_state=0,
    ).fit(X, y)

    assert_shared_classification_methods(model, X, y)
    assert model.score(X, y) == model.accuracy(X, y)


def test_hard_voting_classifier_uses_shared_metrics():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = HardVotingClassifier(
        [
            ("a", FixedClassifier([0, 0, 1, 1])),
            ("b", FixedClassifier([0, 1, 1, 1])),
        ]
    )

    assert_shared_classification_methods(model, X, y)
    assert model.score(X, y) == model.accuracy(X, y)
    assert model.individual_scores(X, y)["ensemble"] == model.score(X, y)


def test_perceptron_binary_metrics_use_shared_metrics():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([-1, -1, 1, 1])
    model = Perceptron(activation="unitstep")
    model.weights = np.array([1.0])
    model.bias = -1.5

    y_pred = model.predict(X)
    assert model.accuracy(X, y) == metrics.accuracy(y, y_pred)
    assert model.precision(X, y) == metrics.precision(
        y, y_pred, average="binary", labels=[-1, 1], positive_label=1
    )
    assert model.recall(X, y) == metrics.recall(
        y, y_pred, average="binary", labels=[-1, 1], positive_label=1
    )
    assert model.f1_score(X, y) == metrics.f1_score(
        y, y_pred, average="binary", labels=[-1, 1], positive_label=1
    )


def test_logistic_regression_binary_metrics_use_shared_metrics():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = LogisticRegression()
    model.task = "binary"
    model.weights = np.array([1.0])
    model.bias = -1.5

    y_pred = model.predict(X)
    np.testing.assert_array_equal(
        model.confusion_matrix(X, y),
        metrics.confusion_matrix(y, y_pred, labels=[0, 1]),
    )
    assert model.accuracy(X, y) == metrics.accuracy(y, y_pred)
    assert model.precision(X, y) == metrics.precision(
        y, y_pred, average="binary", labels=[0, 1], positive_label=1
    )
    assert model.recall(X, y) == metrics.recall(
        y, y_pred, average="binary", labels=[0, 1], positive_label=1
    )
    assert model.f1_score(X, y) == metrics.f1_score(
        y, y_pred, average="binary", labels=[0, 1], positive_label=1
    )


# ---------------------------------------------------------------------------
# Unit tests — weighted averaging
# ---------------------------------------------------------------------------

def test_precision_weighted_differs_from_macro_on_imbalanced_data():
    # Class 0 has 3 samples, class 1 has 1 sample — weighted ≠ macro
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 0, 1, 1])
    p_macro = metrics.precision(y_true, y_pred, average="macro")
    p_weighted = metrics.precision(y_true, y_pred, average="weighted")
    assert p_macro != pytest.approx(p_weighted)


def test_recall_weighted_differs_from_macro_on_imbalanced_data():
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 0, 1, 1])
    r_macro = metrics.recall(y_true, y_pred, average="macro")
    r_weighted = metrics.recall(y_true, y_pred, average="weighted")
    assert r_macro != pytest.approx(r_weighted)


def test_f1_weighted_differs_from_macro_on_imbalanced_data():
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 0, 1, 1])
    f_macro = metrics.f1_score(y_true, y_pred, average="macro")
    f_weighted = metrics.f1_score(y_true, y_pred, average="weighted")
    assert f_macro != pytest.approx(f_weighted)


def test_precision_weighted_known_value():
    # class 0: TP=2, FP=0 → prec=1.0; class 1: TP=1, FP=1 → prec=0.5
    # weights = [3, 1] → weighted = (1.0*3 + 0.5*1) / 4 = 0.875
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 0, 1, 1])
    assert metrics.precision(y_true, y_pred, average="weighted") == pytest.approx(0.875)


def test_recall_weighted_known_value():
    # class 0: TP=2, FN=1 → recall=2/3; class 1: TP=1, FN=0 → recall=1.0
    # weights = [3, 1] → weighted = (2/3*3 + 1.0*1) / 4 = 3/4 = 0.75
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 0, 1, 1])
    assert metrics.recall(y_true, y_pred, average="weighted") == pytest.approx(0.75)


def test_precision_weighted_perfect_predictions_is_one():
    y = np.array([0, 0, 1, 1, 2, 2])
    assert metrics.precision(y, y, average="weighted") == pytest.approx(1.0)


def test_recall_weighted_perfect_predictions_is_one():
    y = np.array([0, 0, 1, 1, 2, 2])
    assert metrics.recall(y, y, average="weighted") == pytest.approx(1.0)


def test_f1_weighted_perfect_predictions_is_one():
    y = np.array([0, 0, 1, 1, 2, 2])
    assert metrics.f1_score(y, y, average="weighted") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Unit tests — micro averaging
# ---------------------------------------------------------------------------

def test_precision_micro_equals_accuracy_for_multiclass():
    # For multiclass, micro-precision equals overall accuracy
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    assert metrics.precision(y_true, y_pred, average="micro") == pytest.approx(
        metrics.accuracy(y_true, y_pred)
    )


def test_recall_micro_equals_accuracy_for_multiclass():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    assert metrics.recall(y_true, y_pred, average="micro") == pytest.approx(
        metrics.accuracy(y_true, y_pred)
    )


def test_f1_micro_perfect_predictions_is_one():
    y = np.array([0, 1, 2, 0, 1, 2])
    assert metrics.f1_score(y, y, average="micro") == pytest.approx(1.0)


def test_precision_micro_known_value():
    # TP_total = 3, FP_total = 1 → micro-precision = 3/4
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    # Both classes: TP=1 each; predicted_pos=[2,2]; total TP=2, total FP=2-2+2-2=... recount:
    # cm = [[1,1],[1,1]] → diag=[1,1], col_sum=[2,2]
    # micro = sum(TP)/(sum(TP)+sum(FP)) = 2/(2+2) = 0.5  (but let's just assert approx == acc)
    assert metrics.precision(y_true, y_pred, average="micro") == pytest.approx(
        metrics.accuracy(y_true, y_pred)
    )


# ---------------------------------------------------------------------------
# Unit tests — invalid average guard
# ---------------------------------------------------------------------------

def test_precision_invalid_average_raises_value_error():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    with pytest.raises(ValueError, match="average must be one of"):
        metrics.precision(y_true, y_pred, average="invalid")


def test_recall_invalid_average_raises_value_error():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    with pytest.raises(ValueError, match="average must be one of"):
        metrics.recall(y_true, y_pred, average="invalid")


def test_f1_score_invalid_average_raises_value_error():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    with pytest.raises(ValueError, match="average must be one of"):
        metrics.f1_score(y_true, y_pred, average="invalid")