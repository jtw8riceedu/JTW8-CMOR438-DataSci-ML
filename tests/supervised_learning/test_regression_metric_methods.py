import numpy as np

from src.ml_package.supervised_learning.decision_tree import DecisionTreeRegressor
from src.ml_package.supervised_learning.knn import KNN
from src.ml_package.supervised_learning.random_forest import RandomForestRegressor
from src.ml_package.supervised_learning.voting import VotingRegressor
from src.ml_package.utils import regression_metrics as metrics


class FixedRegressor:
    def __init__(self, predictions):
        self.predictions = np.array(predictions, dtype=float)

    def predict(self, X):
        return self.predictions[: len(X)]


def assert_shared_regression_methods(model, X, y):
    y_pred = model.predict(X)

    assert model.rmse(X, y) == metrics.rmse(y, y_pred)
    assert model.mae(X, y) == metrics.mae(y, y_pred)
    assert model.mape(X, y) == metrics.mape(y, y_pred)
    assert model.smape(X, y) == metrics.smape(y, y_pred)
    assert model.mase(X, y) == metrics.mase(y, y_pred)
    assert model.r_squared(X, y) == metrics.r_squared(y, y_pred)


def test_decision_tree_regressor_uses_shared_metrics():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    model = DecisionTreeRegressor(max_depth=2).fit(X, y)

    assert_shared_regression_methods(model, X, y)
    assert model.score(X, y) == model.r_squared(X, y)


def test_knn_regression_uses_shared_metrics():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    model = KNN(k=1, regression=True).fit(X, y)

    assert_shared_regression_methods(model, X, y)
    assert model.score(X, y) == metrics.mean_squared_error(y, model.predict(X))


def test_random_forest_regressor_uses_shared_metrics():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    model = RandomForestRegressor(
        n_estimators=3,
        max_depth=2,
        random_state=0,
    ).fit(X, y)

    assert_shared_regression_methods(model, X, y)
    assert model.score(X, y) == model.r_squared(X, y)


def test_voting_regressor_uses_shared_metrics():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    model = VotingRegressor(
        [
            ("low", FixedRegressor([1.0, 2.0, 3.0, 4.0])),
            ("high", FixedRegressor([2.0, 3.0, 4.0, 5.0])),
        ]
    )

    assert_shared_regression_methods(model, X, y)
    assert model.score(X, y) == model.r_squared(X, y)
    assert model.individual_scores(X, y)["ensemble"] == model.score(X, y)
