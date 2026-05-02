""" voting.py """

import numpy as np

from ..utils import classification_metrics
from ..utils import regression_metrics as metrics

# ------------------------------------------------------------------
#   Class for voting classifier
# ------------------------------------------------------------------

class HardVotingClassifier:
    """
    An ensemble classifier that uses majority vote to make final predictions
    by aggregating individual predictions from multiple fitted models.

    Each model makes one prediction per sample, and the class receiving the
    most votes is returned as the final prediction. Ties are broken by
    whichever tied class appears first in the sorted order of class labels.

    This class is model-agnostic; any object with a predict() method
    can be included, regardless of algorithm (decision tree, logistic
    regression, KNN, etc.).

    Parameters
    ----------
    models: list of (str, estimator) tuples
        Named models to include in the vote. Each element should be a
        tuple of (name, fitted_model), e.g.:
            [("tree",  DecisionTreeClassifier(...).fit(X, y)),
             ("knn", KNN(...).fit(X, y))]
        Models must already be fitted before being passed in.

    Attributes
    ----------
    classes_: np.ndarray
        Unique class labels seen across all model predictions, set after
        calling predict() for the first time.

    Methods
    -------
    predict(X)
        Predict class labels by majority vote.
    score(X, y)
        Return classification accuracy.
    confusion_matrix(X, y)
        Return the confusion matrix.
    accuracy(X, y)
        Return classification accuracy.
    precision(X, y, average)
        Compute precision with configurable averaging.
    recall(X, y, average)
        Compute recall with configurable averaging.
    f1_score(X, y, average)
        Compute the F1 score with configurable averaging.
    individual_scores(X, y)
        Return each model's accuracy and the ensemble accuracy.

    Examples
    --------
    >>> from hard_voting import HardVotingClassifier
    >>> from decision_tree import DecisionTreeClassifier
    >>> import numpy as np
    >>> X = np.array([[1,2],[3,4],[5,6],[7,8]])
    >>> y = np.array([0, 0, 1, 1])
    >>> t1 = DecisionTreeClassifier(max_depth=1).fit(X, y)
    >>> t2 = DecisionTreeClassifier(max_depth=2).fit(X, y)
    >>> t3 = DecisionTreeClassifier(max_depth=3).fit(X, y)
    >>> vc = HardVotingClassifier([("t1", t1), ("t2", t2), ("t3", t3)])
    >>> vc.predict(X)
    array([0, 0, 1, 1])
    """

    def __init__(self, models):
        """
        Initialize the hard voting classifier.

        Parameters
        ----------
        models: list of (str, estimator) tuples
            Named, already-fitted models to include in the ensemble.
            Each element must be a ``(name, model)`` tuple where
            ``model`` exposes a ``predict()`` method, e.g.:

                [("tree", DecisionTreeClassifier(...).fit(X, y)),
                 ("knn",  KNN(...).fit(X, y))]

        Raises
        ------
        ValueError
            If ``models`` is an empty list.
        """
    
        if len(models) == 0:
            raise ValueError("models list must contain at least one estimator.")
        self.models = models
        self.classes_ = None

    # Helper function to validate each model.
    def _validate_models(self):
        """Check that every model has a predict() method."""
        for name, model in self.models:
            if not hasattr(model, "predict"):
                raise TypeError(
                    f"Model '{name}' does not have a predict() method."
                )

    # Prediction method
    def predict(self, X):
        """
        Predict class labels for samples in X by majority vote.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
            Majority-vote predicted class label for each sample.
        """
        self._validate_models()
        X = np.array(X)

        # Collect predictions from every model — shape (n_models, n_samples)
        all_predictions = np.array([model.predict(X) for _, model in self.models])

        # Record all the class labels seen
        self.classes_ = np.unique(all_predictions)

        # For each sample, take a majority vote across the n_models predictions
        n_samples = X.shape[0]
        y_pred = np.empty(n_samples, dtype=all_predictions.dtype)

        for i in range(n_samples):
            votes = all_predictions[:, i]
            classes, counts = np.unique(votes, return_counts=True)
            y_pred[i] = classes[np.argmax(counts)]

        return y_pred


 # Evaluation method
    def score(self, X, y):
        """
        Return the mean accuracy on the given data and labels.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
        y: array-like, shape (n_samples,)

        Returns
        -------
        accuracy: float
            Fraction of correctly classified samples.
        """
        return self.accuracy(X, y)

    def confusion_matrix(self, X, y):
        """
        Compute the confusion matrix.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Input data.
        y: array-like, shape (n_samples,)
            Observed class labels.

        Returns
        -------
        numpy.ndarray, shape (n_classes, n_classes)
            Confusion matrix with rows as true labels and columns as
            predicted labels.
        """
        return classification_metrics.confusion_matrix(y, self.predict(X))

    def accuracy(self, X, y):
        """
        Compute the proportion of correctly classified samples.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Input data.
        y: array-like, shape (n_samples,)
            Observed class labels.

        Returns
        -------
        float
            Proportion of samples correctly classified, in [0, 1].
        """
        return classification_metrics.accuracy(y, self.predict(X))

    def precision(self, X, y, average="macro"):
        """
        Compute precision with configurable class averaging.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Input data.
        y: array-like, shape (n_samples,)
            Observed class labels.
        average: {'binary', 'macro', 'weighted', 'micro'}, optional
            Averaging strategy. Defaults to 'macro'.

        Returns
        -------
        float
            Precision score.
        """
        return classification_metrics.precision(y, self.predict(X), average=average)

    def recall(self, X, y, average="macro"):
        """
        Compute recall with configurable class averaging.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Input data.
        y: array-like, shape (n_samples,)
            Observed class labels.
        average: {'binary', 'macro', 'weighted', 'micro'}, optional
            Averaging strategy. Defaults to 'macro'.

        Returns
        -------
        float
            Recall score.
        """
        return classification_metrics.recall(y, self.predict(X), average=average)

    def f1_score(self, X, y, average="macro"):
        """
        Compute the F1 score with configurable class averaging.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Input data.
        y: array-like, shape (n_samples,)
            Observed class labels.
        average: {'binary', 'macro', 'weighted', 'micro'}, optional
            Averaging strategy. Defaults to 'macro'.

        Returns
        -------
        float
            F1 score.
        """
        return classification_metrics.f1_score(y, self.predict(X), average=average)

    def individual_scores(self, X, y):
        """
        Return the accuracy of each individual model alongside the
        ensemble score. Helps determine if the ensemble model adds value.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
        y: array-like, shape (n_samples,)

        Returns
        -------
        scores: dict
            {model_name: accuracy} for each individual model, plus an
            "ensemble" key for the hard voting score.
        """
        self._validate_models()
        X = np.array(X)
        y = np.array(y)

        scores = {}
        for name, model in self.models:
            scores[name] = classification_metrics.accuracy(y, model.predict(X))
        scores["ensemble"] = self.score(X, y)

        return scores



# ------------------------------------------------------------------
#   Class for voting regressor
# ------------------------------------------------------------------

class VotingRegressor:
    """
    An ensemble regressor that aggregates predictions from multiple
    fitted models by averaging.

    Each model contributes one prediction per sample, and the final
    prediction is the mean across all models. Supports weighted
    averaging.

    This class is model-agnostic — any object with a predict() method
    can be included, regardless of algorithm (decision tree, random
    forest, linear regression, etc.).

    Parameters
    ----------
    models: list of (str, estimator) tuples
        Named models to include in the ensemble. Each element should be
        a tuple of (name, fitted_model), e.g.:
            [("tree",   DecisionTreeRegressor(...).fit(X, y)),
             ("forest", RandomForestRegressor(...).fit(X, y))]
        Models must already be fitted before being passed in.
    weights: array-like of shape (n_models,) or None, default=None
        Weight for each model's prediction. If None, all models are
        weighted equally. Weights do not need to sum to 1 — they are
        normalised internally.

    Attributes
    ----------
    weights_: np.ndarray
        Normalised weights actually used during prediction, set after
        calling predict() for the first time.

    Methods
    -------
    predict(X)
        Predict continuous target values by weighted averaging.
    score(X, y)
        Return the R-squared score.
    rmse(X, y)
        Root mean squared error.
    mae(X, y)
        Mean absolute error.
    mape(X, y)
        Mean absolute percentage error.
    smape(X, y)
        Symmetric mean absolute percentage error.
    mase(X, y)
        Mean absolute scaled error.
    r_squared(X, y)
        Coefficient of determination.
    individual_scores(X, y)
        Return each model's R-squared score and the ensemble R-squared score.

    Examples
    --------
    >>> from voting_regressor import VotingRegressor
    >>> from decision_tree import DecisionTreeRegressor
    >>> import numpy as np
    >>> X = np.array([[1.],[2.],[3.],[4.]])
    >>> y = np.array([1.5, 2.5, 3.5, 4.5])
    >>> t1 = DecisionTreeRegressor(max_depth=1).fit(X, y)
    >>> t2 = DecisionTreeRegressor(max_depth=2).fit(X, y)
    >>> t3 = DecisionTreeRegressor(max_depth=3).fit(X, y)
    >>> vr = VotingRegressor([("t1", t1), ("t2", t2), ("t3", t3)])
    >>> vr.predict(X)
    """

    def __init__(self, models, weights=None):
        """
        Initialize the voting regressor.

        Parameters
        ----------
        models: list of (str, estimator) tuples
            Named, already-fitted models to include in the ensemble.
            Each element must be a ``(name, model)`` tuple where
            ``model`` exposes a ``predict()`` method, e.g.::

                [("tree",   DecisionTreeRegressor(...).fit(X, y)),
                 ("forest", RandomForestRegressor(...).fit(X, y))]

        weights: array-like of shape (n_models,) or None, optional
            Weight assigned to each model's prediction. If ``None``,
            all models are weighted equally. Weights are normalised
            internally and do not need to sum to 1. All weights must
            be non-negative. Defaults to ``None``.

        Raises
        ------
        ValueError
            If ``models`` is an empty list, if the length of ``weights``
            does not match the number of models, or if any weight is
            negative.
        """
    
        if len(models) == 0:
            raise ValueError("models list must contain at least one estimator.")

        if weights is not None:
            weights = np.array(weights, dtype=float)
            if len(weights) != len(models):
                raise ValueError(
                    f"weights length ({len(weights)}) must match "
                    f"number of models ({len(models)})."
                )
            if np.any(weights < 0):
                raise ValueError("All weights must be non-negative.")

        self.models = models
        self.weights = weights
        self.weights_ = None

    # Helper function to validate each model.
    def _validate_models(self):
        """Check that every model has a predict() method."""
        for name, model in self.models:
            if not hasattr(model, "predict"):
                raise TypeError(
                    f"Model '{name}' does not have a predict() method."
                )

    def _normalised_weights(self):
        """
        Return normalised weights as a 1-D array of length n_models.
        If no weights were provided, returns uniform weights.
        """
        if self.weights is None:
            n = len(self.models)
            return np.ones(n) / n
        return self.weights / self.weights.sum()

    # Prediction method
    def predict(self, X):
        """
        Predict target values for samples in X by weighted averaging.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)

        Returns
        -------
        y_pred: np.ndarray, shape (n_samples,)
            Weighted average prediction across all models.
        """
        self._validate_models()
        X = np.array(X)

        self.weights_ = self._normalised_weights()

        # Shape: (n_models, n_samples)
        all_predictions = np.array(
            [model.predict(X) for _, model in self.models]
        )

        # Weighted average across models — equivalent to a dot product
        # between the weight vector and each column of all_predictions
        return self.weights_ @ all_predictions

    # Evaluation method
    def score(self, X, y):
        """
        Return the R-sqaured score on the given data.

        R-squared = 1 - SSE / SSTO. A score of 1.0 is perfect; 0.0 means
        the ensemble does no better than predicting the mean of y.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
        y: array-like, shape (n_samples,)

        Returns
        -------
        r2: float
        """
        return metrics.r_squared(y, self.predict(X))

    def rmse(self, X, y):
        """
        Compute the root mean squared error.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Input data.
        y: array-like, shape (n_samples,)
            Observed target values.

        Returns
        -------
        float
            Square root of the mean squared error.
        """
        return metrics.rmse(y, self.predict(X))

    def mae(self, X, y):
        """
        Compute the mean absolute error.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Input data.
        y: array-like, shape (n_samples,)
            Observed target values.

        Returns
        -------
        float
            Average absolute difference between predictions and targets.
        """
        return metrics.mae(y, self.predict(X))

    def mape(self, X, y):
        """
        Compute the mean absolute percentage error.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Input data.
        y: array-like, shape (n_samples,)
            Observed target values.

        Returns
        -------
        float
            MAPE as a percentage.
        """
        return metrics.mape(y, self.predict(X))

    def smape(self, X, y):
        """
        Compute the symmetric mean absolute percentage error.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Input data.
        y: array-like, shape (n_samples,)
            Observed target values.

        Returns
        -------
        float
            SMAPE as a percentage.
        """
        return metrics.smape(y, self.predict(X))

    def mase(self, X, y):
        """
        Compute the mean absolute scaled error.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Input data.
        y: array-like, shape (n_samples,)
            Observed target values.

        Returns
        -------
        float
            MASE score.
        """
        return metrics.mase(y, self.predict(X))

    def r_squared(self, X, y):
        """
        Compute the coefficient of determination.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Input data.
        y: array-like, shape (n_samples,)
            Observed target values.

        Returns
        -------
        float
            R-squared score.
        """
        return self.score(X, y)

    def individual_scores(self, X, y):
        """
        Return the R-sqaured of each individual model alongside the ensemble
        score. Helps determine if the ensemble model adds value.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
        y: array-like, shape (n_samples,)

        Returns
        -------
        scores: dict
            {model_name: R-sqaured} for each individual model, plus an
            "ensemble" key for the voting regressor score.
        """
        self._validate_models()
        X = np.array(X)
        y = np.array(y, dtype=float)

        scores = {}
        for name, model in self.models:
            scores[name] = metrics.r_squared(y, model.predict(X))
        scores["ensemble"] = self.score(X, y)

        return scores
