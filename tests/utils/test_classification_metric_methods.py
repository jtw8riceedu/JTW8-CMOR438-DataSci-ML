import numpy as np

from src.ml_package.supervised_learning.decision_tree import DecisionTreeClassifier
from src.ml_package.supervised_learning.knn import KNN
from src.ml_package.supervised_learning.logistic_regression import LogisticRegression
from src.ml_package.supervised_learning.perceptron import Perceptron
from src.ml_package.supervised_learning.random_forest import RandomForestClassifier
from src.ml_package.supervised_learning.voting import HardVotingClassifier
from src.ml_package.utils import classification_metrics as metrics


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
