""" knn.py """

import numpy as np

from ..utils import classification_metrics
from ..utils import regression_metrics as metrics

class KNN(object):
    """
    K-Nearest Neighbors for classification and regression.

    Stores the training data at fit time and makes predictions by
    finding the ``k`` closest training samples (using Euclidean distance)
    for each query point. For classification, the majority class among
    the neighbors is returned; for regression, their mean target value
    is returned.

    KNN uses Euclidean distance and is sensitive to feature scale. Scale
    features before fitting when columns use different units or ranges
    (for example with ``StandardScaler`` from ``ml_package.utils``).

    Parameters
    ----------
    k: int
        Number of nearest neighbors to consider. Must be positive.
    regression: bool, optional
        If ``True``, performs regression (returns mean of neighbor
        targets). If ``False`` (default), performs classification
        (returns majority class label).

    Attributes
    ----------
    X_train: numpy.ndarray
        Training input data, stored at fit time.
    y_train: numpy.ndarray
        Training target values, stored at fit time.

    Methods
    -------
    fit(X, y)
        Store training data.
    predict(X)
        Predict labels or values for input samples.
    score(X, y)
        Compute accuracy (classification) or MSE (regression).
    confusion_matrix(X, y)
        Return the confusion matrix for classification.
    accuracy(X, y)
        Return accuracy for classification.
    precision(X, y, average)
        Compute precision with configurable averaging for classification.
    recall(X, y, average)
        Compute recall with configurable averaging for classification.
    f1_score(X, y, average)
        Compute the F1 score with configurable averaging for classification.
    rmse(X, y)
        Root mean squared error for regression.
    mae(X, y)
        Mean absolute error for regression.
    mape(X, y)
        Mean absolute percentage error for regression.
    smape(X, y)
        Symmetric mean absolute percentage error for regression.
    mase(X, y)
        Mean absolute scaled error for regression.
    r_squared(X, y)
        Coefficient of determination for regression.
    """


# ------------------------------------------------------------------
#   Initialize network
# ------------------------------------------------------------------

    def __init__(self, k, regression = False):
        """
        Initialize the KNN model.

        Parameters
        ----------
        k: int
            Number of nearest neighbors. Must be a positive integer.
        regression: bool, optional
            Set to ``True`` for regression tasks, ``False`` for
            classification. Defaults to ``False``.

        Raises
        ------
        ValueError
            If ``k`` is not a positive integer.
        """      
        
        if k <= 0:
            raise ValueError("k must be positive")

        self.k = k
        self.regression = regression


# ------------------------------------------------------------------
#   Fit the model
# ------------------------------------------------------------------
    
    # Fit function used to store the training data
    def fit(self, X, y):
        """
        Store the training data for use at prediction time.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training input data.
        y: array-like, shape (n_samples,)
            Training target values or class labels.

        Returns
        -------
        self: KNN
            The fitted KNN instance.

        Raises
        ------
        ValueError
            If ``k`` exceeds the number of training samples.
        """    
        
        self.X_train = np.array(X)
        self.y_train = np.array(y)

        if self.k > len(self.X_train):
            raise ValueError("k cannot be greater than the number of training samples")

        return self


    # Euclidean distance function
    def _distance(self, p, q):
        """
        Compute the Euclidean distance between two points.

        Parameters
        ----------
        p: numpy.ndarray, shape (n_features,)
            First point.
        q: numpy.ndarray, shape (n_features,)
            Second point.

        Returns
        -------
        float
            Euclidean distance between ``p`` and ``q``.
        """   
        
        return np.sqrt((p - q) @ (p - q))


# ------------------------------------------------------------------
#   Predict using the model
# ------------------------------------------------------------------

    # Predict function
    def predict(self, X):
        """
        Predict labels or values for input samples.

        Applies ``_predict`` to each row of ``X`` independently.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Input data to predict.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Predicted class labels (classification) or continuous values
            (regression).
        """    
        
        X = np.array(X)
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)


    # Hidden predict function
    def _predict(self, x):
        """
        Predict the label or value for a single sample.

        Computes the distance from ``x`` to every training point, finds
        the ``k`` nearest neighbors, and returns either the majority class
        (classification) or the mean target value (regression).

        Parameters
        ----------
        x: numpy.ndarray, shape (n_features,)
            A single input sample.

        Returns
        -------
        scalar
            Predicted class label or continuous value.
        """     
        
        # Compute distances between x and all examples in the training set
        distances = np.array([self._distance(x, x_train) for x_train in self.X_train])

        # Sort distances in ascending order and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Extract the target variable values of the k nearest training samples
        k_nearest_values = self.y_train[k_indices]

        # For classification, return the most common class label
        if not self.regression:
            values, counts = np.unique(k_nearest_values, return_counts = True)
            return values[np.argmax(counts)]

        # For regression, return the mean of the y values
        else:
            return np.mean(k_nearest_values)


# ------------------------------------------------------------------
#   Evaluation
# ------------------------------------------------------------------

    # Evaluation function
    def score(self, X, y):
        """
        Evaluate the model on labeled data.

        For classification, returns accuracy (fraction of correct
        predictions). For regression, returns mean squared error.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Input data.
        y: array-like, shape (n_samples,)
            Observed labels or target values.

        Returns
        -------
        float
            Accuracy in [0, 1] for classification, or MSE for regression.
        """
    
        y = np.array(y)
        y_pred = self.predict(X)

        if not self.regression:
            # Accuracy for classifications
            return classification_metrics.accuracy(y, y_pred)
        else:
            # MSE for regression
            return metrics.mean_squared_error(y, y_pred)

    def _check_classification(self):
        """Raise an error when a classification-only metric is used for regression."""
        if self.regression:
            raise ValueError("This metric is only available when regression=False.")

    def _check_regression(self):
        """Raise an error when a regression-only metric is used for classification."""
        if not self.regression:
            raise ValueError("This metric is only available when regression=True.")

    def confusion_matrix(self, X, y):
        """
        Compute the confusion matrix for classification.

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

        Raises
        ------
        ValueError
            If the model was initialized with ``regression=True``.
        """
        self._check_classification()
        return classification_metrics.confusion_matrix(y, self.predict(X))

    def accuracy(self, X, y):
        """
        Compute classification accuracy.

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

        Raises
        ------
        ValueError
            If the model was initialized with ``regression=True``.
        """
        self._check_classification()
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

        Raises
        ------
        ValueError
            If the model was initialized with ``regression=True``.
        """
        self._check_classification()
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

        Raises
        ------
        ValueError
            If the model was initialized with ``regression=True``.
        """
        self._check_classification()
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

        Raises
        ------
        ValueError
            If the model was initialized with ``regression=True``.
        """
        self._check_classification()
        return classification_metrics.f1_score(y, self.predict(X), average=average)

    def rmse(self, X, y):
        """
        Compute the root mean squared error for regression.

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

        Raises
        ------
        ValueError
            If the model was not initialized with ``regression=True``.
        """
        self._check_regression()
        return metrics.rmse(y, self.predict(X))

    def mae(self, X, y):
        """
        Compute the mean absolute error for regression.

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

        Raises
        ------
        ValueError
            If the model was not initialized with ``regression=True``.
        """
        self._check_regression()
        return metrics.mae(y, self.predict(X))

    def mape(self, X, y):
        """
        Compute the mean absolute percentage error for regression.

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

        Raises
        ------
        ValueError
            If the model was not initialized with ``regression=True``.
        """
        self._check_regression()
        return metrics.mape(y, self.predict(X))

    def smape(self, X, y):
        """
        Compute the symmetric mean absolute percentage error for regression.

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

        Raises
        ------
        ValueError
            If the model was not initialized with ``regression=True``.
        """
        self._check_regression()
        return metrics.smape(y, self.predict(X))

    def mase(self, X, y):
        """
        Compute the mean absolute scaled error for regression.

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

        Raises
        ------
        ValueError
            If the model was not initialized with ``regression=True``.
        """
        self._check_regression()
        return metrics.mase(y, self.predict(X))

    def r_squared(self, X, y):
        """
        Compute the coefficient of determination for regression.

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

        Raises
        ------
        ValueError
            If the model was not initialized with ``regression=True``.
        """
        self._check_regression()
        return metrics.r_squared(y, self.predict(X))






