""" random_forest.py """

import numpy as np

from ..utils import classification_metrics
from ..utils import regression_metrics as metrics
from .decision_tree import DecisionTreeClassifier, DecisionTreeRegressor


# ------------------------------------------------------------------
#   Subclass DecisionTreeClassifier
# ------------------------------------------------------------------

# The random forest classifier builds off the DecisionTreeClassifier from decision_tree.py .

# The _RFDecisionTreeClassifier class adds an internal DecisionTreeClassifier subclass to add random feature subsetting 
# to the _best_split internal method. The original DecisionTreeClassifier class is untouched.

class _RFDecisionTreeClassifier(DecisionTreeClassifier):
    """
    DecisionTreeClassifier with random feature subsetting at each split.

    The class adds one parameter (max_features) and only changes fit() _best_split().
    All other methods and functions are unchanged from DecisionTreeClassifier.
    """

    def __init__(self, max_depth = 3, criterion = "entropy",
                 min_samples_split = 2, max_features = None, random_state = None):
        """
        Initialize the random-forest variant of the decision tree classifier.

        Calls the parent ``DecisionTreeClassifier.__init__`` and adds
        parameters for random feature subsetting and reproducibility.

        Parameters
        ----------
        max_depth: int, optional
            Maximum depth of the tree. Defaults to 3.
        criterion: {'entropy', 'gini'}, optional
            Impurity measure used for splitting. Defaults to 'entropy'.
        min_samples_split: int, optional
            Minimum samples required at a node to attempt a split.
            Defaults to 2.
        max_features: int or None, optional
            Number of features to consider at each split. If ``None``,
            all features are used. Defaults to ``None``.
        random_state: int or None, optional
            Seed for the internal ``numpy.random.Generator`` used to
            draw feature subsets. Defaults to ``None``.
        """   
        
        super().__init__(max_depth = max_depth, criterion = criterion,
                         min_samples_split = min_samples_split)
        self.max_features = max_features
        self.random_state = random_state
        self.rng_ = None

    def fit(self, X, y):
        """
        Create the rng once before tree construction, then delegate to
        the parent fit(). self.rng_ is then reused across all _best_split
        calls so each node draws a new, independent feature subset.
        """
        self.rng_ = np.random.default_rng(self.random_state)
        return super().fit(X, y)

    def _best_split(self, X, y):
        """
        Identical to the parent's _best_split, but only considers a random
        subset of features (of size self.max_features) at each call.
        """
        n_features = X.shape[1]

        # Draw the random feature subset
        k = self.max_features if self.max_features is not None else n_features
        k = min(k, n_features)
        feature_indices = self.rng_.choice(n_features, size=k, replace=False)

        best_feature = None
        best_threshold = None
        best_impurity = np.inf

        for feature_idx in feature_indices:
            # Candidate thresholds: all unique values in this feature column
            thresholds = np.unique(X[:, feature_idx])

            # Iterate across all unique values in the particular feature
            for threshold in thresholds:
                left_vals = X[:, feature_idx] <= threshold
                right_vals = X[:, feature_idx] > threshold

                y_left = y[left_vals]
                y_right = y[right_vals]

                # Skip if either partition is empty
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                imp = self._weighted_impurity(y_left, y_right)

                if imp < best_impurity:
                    best_impurity = imp
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold
    

    
# ------------------------------------------------------------------
#   Main class for the random forest classifier
# ------------------------------------------------------------------

class RandomForestClassifier:
    """
    A random forest classifier built from scratch.

    The algorithm trains an ensemble of decision trees, each on a bootstrapped
    sample of the training data. At every split, only a random subset of
    features is considered. Final predictions are made by majority vote over
    each individual decision tree prediction.

    Parameters
    ----------
    n_estimators: int, default=100
        Number of trees in the forest.
    max_depth: int, default=3
        Maximum depth of each tree.
    criterion: {'entropy', 'gini'}, default='entropy'
        Impurity measure used for splitting.
    min_samples_split: int, default=2
        Minimum samples required at a node to attempt a split.
    max_feature : int or None, default=None
        Number of features to consider at each split. If None, defaults
        to floor(sqrt(n_features)), which is the standard for classification.
    random_state: int or None, default=None
        Seed for reproducibility. Controls both bootstrap sampling and
        feature subsetting.

    Attributes
    ----------
    estimators_: list of _RFDecisionTreeClassifier
        The fitted trees. Available after calling fit().
    n_features_: int
        Number of features seen during training.
    n_classes_: int
        Number of unique classes seen during training.
    classes_: np.ndarray
        Unique class labels seen during training.

    Methods
    -------
    fit(X, y)
        Fit the forest to training data.
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

    Examples
    --------
    >>> import numpy as np
    >>> from random_forest import RandomForestClassifier
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = RandomForestClassifier(n_estimators=50, random_state=0)
    >>> clf.fit(X, y)
    >>> clf.score(X, y)
    """

    def __init__(self, n_estimators = 100, max_depth = 3, criterion = "entropy",
                 min_samples_split = 2, max_features = None, random_state = None):
        """
        Initialize the random forest classifier.

        Parameters
        ----------
        n_estimators: int, optional
            Number of decision trees to train. Defaults to 100.
        max_depth: int, optional
            Maximum depth of each individual tree. Defaults to 3.
        criterion: {'entropy', 'gini'}, optional
            Impurity measure used for splitting at each node.
            Defaults to 'entropy'.
        min_samples_split: int, optional
            Minimum number of samples required at a node to attempt a
            split. Defaults to 2.
        max_features: int or None, optional
            Number of features considered at each split. If ``None``,
            defaults to ``floor(sqrt(n_features))`` at fit time, which
            is the standard heuristic for classification. Defaults to
            ``None``.
        random_state: int or None, optional
            Seed for reproducibility. Controls both bootstrap sampling
            and the per-tree random feature subsets. Defaults to ``None``.
        """    
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

        self.estimators_ = []
        self.n_features_ = None
        self.n_classes_ = None
        self.classes_ = None


    # ------------------------------------------------------------------
    #   Fit the random forest
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Fit the random forest to training data.

        Each tree is trained on a bootstrap sample of the training data
        (same size as the original dataset, drawn with replacement).

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
        y: array-like, shape (n_samples,)

        Returns
        -------
        self
        """
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Default max_features = floor(sqrt(n_features)) for classification
        max_features = (self.max_features if self.max_features is not None
                        else max(1, int(np.floor(np.sqrt(n_features)))))

        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []

        for i in range(self.n_estimators):
            # Give each tree a distinct but reproducible random seed.
            tree_seed = int(rng.integers(0, 1_000_000))

            # Bootstrap the training data
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot, y_boot = X[indices], y[indices]

            tree = _RFDecisionTreeClassifier(
                max_depth = self.max_depth,
                criterion = self.criterion,
                min_samples_split = self.min_samples_split,
                max_features = max_features,
                random_state = tree_seed,
            )
            tree.fit(X_boot, y_boot)
            self.estimators_.append(tree)

        return self


    # ------------------------------------------------------------------
    #   Predict using the random forest
    # ------------------------------------------------------------------

    def predict(self, X):
        """
        Predict class labels by majority vote across all decision trees.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)

        Returns
        -------
        y_pred: np.ndarray, shape (n_samples,)
        """
        if not self.estimators_:
            raise RuntimeError("Call fit() before predict().")

        X = np.array(X)

        # Shape: (n_estimators, n_samples)
        # Make predictions for each decision tree in the random forest
        all_preds = np.array([tree.predict(X) for tree in self.estimators_])

        # Make predictions via majority vote for each sample
        n_samples = X.shape[0]
        y_pred = np.empty(n_samples, dtype=all_preds.dtype)

        for i in range(n_samples):
            votes = all_preds[:, i]
            classes, counts = np.unique(votes, return_counts=True)
            y_pred[i] = classes[np.argmax(counts)]

        return y_pred


    # ------------------------------------------------------------------
    #   Evaluation method
    # ------------------------------------------------------------------

    def score(self, X, y):
        """
        Return mean accuracy on the given data and labels.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
        y: array-like, shape (n_samples,)

        Returns
        -------
        accuracy: float
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



# ------------------------------------------------------------------
#   Subclass DecisionTreeRegressor
# ------------------------------------------------------------------

# The random forest regressor builds off the decision tree regressor.

# The class adds an internal DecisionTreeRegressor subclass to add random feature subsetting 
# to the _best_split internal method. The original DecisionTreeRegressor class is untouched.

class _RFDecisionTreeRegressor(DecisionTreeRegressor):
    """
    DecisionTreeRegressor with random feature subsetting at each split.

    The class adds one parameter (max_features) and changes fit() and _best_split().
    All other methods and functions are unchanged from DecisionTreeRegressor.
    """
    def __init__(self, max_depth = 3, min_samples_split = 2,
                 max_features = None, random_state = None):
        """
        Initialize the random-forest variant of the decision tree regressor.

        Calls the parent ``DecisionTreeRegressor.__init__`` and adds
        parameters for random feature subsetting and reproducibility.

        Parameters
        ----------
        max_depth: int, optional
            Maximum depth of the tree. Defaults to 3.
        min_samples_split: int, optional
            Minimum samples required at a node to attempt a split.
            Defaults to 2.
        max_features: int or None, optional
            Number of features to consider at each split. If ``None``,
            all features are used. Defaults to ``None``.
        random_state: int or None, optional
            Seed for the internal ``numpy.random.Generator`` used to
            draw feature subsets. Defaults to ``None``.
        """
    
        super().__init__(max_depth = max_depth, min_samples_split = min_samples_split)
        self.max_features = max_features
        self.random_state = random_state
        self.rng_ = None


    def fit(self, X, y):
        """
        Create the rng once before tree construction, then delegate to
        the parent fit(). self.rng_ is then reused across all _best_split
        calls so each node draws a new, independent feature subset.
        """
        self.rng_ = np.random.default_rng(self.random_state)
        return super().fit(X, y)


    def _best_split(self, X, y):
        """
        Identical to the parent's _best_split, but only considers a random
        subset of features (of size self.max_features) at each call.
        """
        n_features = X.shape[1]

        k = self.max_features if self.max_features is not None else n_features
        k = min(k, n_features)
        feature_indices = self.rng_.choice(n_features, size=k, replace=False)

        best_feature = None
        best_threshold = None
        best_mse = np.inf

        for feature_idx in feature_indices:
            # Candidate thresholds: all unique values in this feature column
            thresholds = np.unique(X[:, feature_idx])

            # Iterate across all unique values in the particular feature
            for threshold in thresholds:
                left_vals = X[:, feature_idx] <= threshold
                right_vals = X[:, feature_idx] > threshold

                y_left = y[left_vals]
                y_right = y[right_vals]

                # Skip if either partition is empty
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                mse = self._weighted_mse(y_left, y_right)

                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold



# ------------------------------------------------------------------
#   Main class for the random forest regressor
# ------------------------------------------------------------------

class RandomForestRegressor:
    """
    A random forest regressor built from scratch.

    The algorithm trains an ensemble of decision trees, each on a bootstrapped
    sample of the training data. At every split, only a random subset of
    features is considered. Final predictions are made by averaging each
    individual decision tree prediction.

    Parameters
    ----------
    n_estimators: int, default=100
        Number of trees in the random forest.
    max_depth: int, default=3
        Maximum depth of each tree.
    min_samples_split: int, default=2
        Minimum samples required at a node to attempt a split.
    max_features: int or None, default=None
        Number of features to consider at each split. If None, defaults
        to floor(n_features / 3), which is the standard for regression.
    random_state: int or None, default=None
        Seed for reproducibility.

    Attributes
    ----------
    estimators_: list of _RFDecisionTreeRegressor
        The fitted trees. Available after calling fit().
    n_features_: int
        Number of features seen during training.

    Methods
    -------
    fit(X, y)
        Fit the forest to training data.
    predict(X)
        Predict continuous target values by averaging tree predictions.
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

    Examples
    --------
    >>> import numpy as np
    >>> from random_forest import RandomForestRegressor
    >>> from sklearn.datasets import load_diabetes
    >>> X, y = load_diabetes(return_X_y=True)
    >>> reg = RandomForestRegressor(n_estimators=50, random_state=0)
    >>> reg.fit(X, y)
    >>> reg.score(X, y)
    """

    def __init__(self, n_estimators = 100, max_depth = 3, min_samples_split = 2,
                 max_features = None, random_state = None):
        """
        Initialize the random forest regressor.

        Parameters
        ----------
        n_estimators: int, optional
            Number of decision trees to train. Defaults to 100.
        max_depth: int, optional
            Maximum depth of each individual tree. Defaults to 3.
        min_samples_split: int, optional
            Minimum number of samples required at a node to attempt a
            split. Defaults to 2.
        max_features: int or None, optional
            Number of features considered at each split. If ``None``,
            defaults to ``floor(n_features / 3)`` at fit time, which is
            the standard heuristic for regression. Defaults to ``None``.
        random_state: int or None, optional
            Seed for reproducibility. Controls both bootstrap sampling
            and the per-tree random feature subsets. Defaults to ``None``.
        """
    
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

        self.estimators_ = []
        self.n_features_ = None

    # ------------------------------------------------------------------
    #   Fit the random forest
    # ------------------------------------------------------------------


    def fit(self, X, y):
        """
        Fit the random forest to training data.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
        y: array-like, shape (n_samples,)

        Returns
        -------
        self
        """
        X = np.array(X)
        y = np.array(y, dtype=float)

        n_samples, n_features = X.shape
        self.n_features_ = n_features

        # Default max_features = floor(n_features / 3) for regression
        max_features = (self.max_features if self.max_features is not None
                        else max(1, int(np.floor(n_features / 3))))

        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []

        for i in range(self.n_estimators):
            # Give each tree a distinct but reproducible random seed.
            tree_seed = int(rng.integers(0, 1_000_000))

            # Bootstrap the training data
            indices = rng.choice(n_samples, size = n_samples, replace = True)
            X_boot, y_boot = X[indices], y[indices]

            tree = _RFDecisionTreeRegressor(
                max_depth = self.max_depth,
                min_samples_split = self.min_samples_split,
                max_features = max_features,
                random_state = tree_seed,
            )
            tree.fit(X_boot, y_boot)
            self.estimators_.append(tree)

        return self


    # ------------------------------------------------------------------
    #   Predict using the random forest
    # ------------------------------------------------------------------

    def predict(self, X):
        """
        Predict target values by averaging predictions across all trees.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)

        Returns
        -------
        y_pred: np.ndarray, shape (n_samples,)
        """
        if not self.estimators_:
            raise RuntimeError("Call fit() before predict().")

        X = np.array(X)

        # Shape: (n_estimators, n_samples)
        # Make predictions for each decision tree in the random forest
        all_preds = np.array([tree.predict(X) for tree in self.estimators_])
        # axis = 0 averages across the rows (individual trees)
        return np.mean(all_preds, axis=0)


    # ------------------------------------------------------------------
    #   Evaluation method
    # ------------------------------------------------------------------

    def score(self, X, y):
        """
        Return the R-squared score on the given data.

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
