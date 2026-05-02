""" decision_tree.py"""

import numpy as np

from ..utils import classification_metrics
from ..utils import regression_metrics as metrics

# ------------------------------------------------------------------
#   Tree node classes
# ------------------------------------------------------------------

class _Leaf:
    """
    A terminal node to classify the data.

    Attributes
    ----------

    prediction: int or str
        The majority class among training samples that reached this leaf.
    class_counts: dict
        Dictionary {class_label: count} for all training samples that reached this leaf.
    """

    def __init__(self, y):
        """
        Initialize the leaf node class for the DecisionTreeClassifier.

        Parameters
        ----------
        y: np.ndarray, shape (n_samples,)
            Target class labels of all training samples that reached
            this node. Used to determine the majority class prediction
            and per-class counts.
        """
        
        classes, counts = np.unique(y, return_counts = True)
        self.class_counts = dict(zip(classes, counts))
        self.prediction = classes[np.argmax(counts)]


class _DecisionNode:
    """
    An internal node that holds a reference to the split rule and to the two child nodes.

    Attributes
    ----------

    feature_idx: int
        Index of the feature column used for splitting.
    threshold: float
        Training samples go left if x[feature_idx] <= threshold, else right.
    left: _Leaf or _DecisionNode
        Subtree for samples where condition is True.
    right: _Leaf or _DecisionNode
        Subtree for samples where condition is False.
    """

    def __init__(self, feature_idx, threshold, left, right):
        """
        Initialize an internal decision node.

        Parameters
        ----------
        feature_idx: int
            Index of the feature column used for splitting.
        threshold: float
            Split boundary. Samples with ``x[feature_idx] <= threshold``
            go to the left child; all others go to the right child.
        left: _Leaf or _DecisionNode
            Subtree for samples where the split condition is True.
        right: _Leaf or _DecisionNode
            Subtree for samples where the split condition is False.
        """    
        
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right



# ------------------------------------------------------------------
#   Main decision tree classifier
# ------------------------------------------------------------------

class DecisionTreeClassifier:
    """
    A decision tree classifier.

    Supports multi-class classification and two impurity criteria:
    entropy (information gain) and gini index.

    Parameters
    ----------
    max_depth: int, default = 3
        Maximum depth of the tree. Growth stops when this depth is reached.
    criterion: {'entropy', 'gini'}, default = 'entropy'
        The impurity measure used to evaluate splits.
    min_samples_split : int, default = 2
        Minimum number of samples required at a node to attempt a split.

    Attributes
    ----------
    tree_: _DecisionNode or _Leaf
        The fitted tree structure. Available after calling fit().
    n_classes_: int
        Number of unique classes seen during training.
    n_features_: int
        Number of features seen during training.

    Methods
    -------
    fit(X, y)
        Fit the classifier to training data.
    predict(X)
        Predict class labels for input samples.
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
    print_tree(node, spacing, feature_names)
        Print a readable representation of the fitted tree.

    Examples
    --------
    >>> import numpy as np
    >>> from decision_tree import DecisionTreeClassifier
    >>> X = np.array([[2.5, 1.0], [1.0, 3.0], [3.0, 2.0], [1.5, 0.5]])
    >>> y = np.array([0, 1, 0, 1])
    >>> clf = DecisionTreeClassifier(max_depth=3, criterion='entropy')
    >>> clf.fit(X, y)
    >>> clf.predict(X)
    array([0, 1, 0, 1])
    """

    def __init__(self, max_depth=3, criterion="entropy", min_samples_split=2):
        """
        Initialize the decision tree classifier.

        Parameters
        ----------
        max_depth: int, optional
            Maximum depth of the tree. Growth stops once this depth is
            reached. Must be >= 1. Defaults to 3.
        criterion: {'entropy', 'gini'}, optional
            Impurity measure used to evaluate candidate splits.
            'entropy' uses information gain; 'gini' uses the Gini index.
            Defaults to 'entropy'.
        min_samples_split: int, optional
            Minimum number of samples required at a node to attempt a
            split. Nodes with fewer samples are converted to leaves.
            Defaults to 2.

        Raises
        ------
        ValueError
            If ``criterion`` is not 'entropy' or 'gini', or if
            ``max_depth`` is less than 1.
        """     
        
        if criterion not in ("entropy", "gini"):
            raise ValueError("criterion must be one of 'entropy' or 'gini'.")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1.")

        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split

        self.tree_ = None
        self.n_classes_ = None
        self.n_features_ = None


    # ------------------------------------------------------------------
    #   Impurity measures
    # ------------------------------------------------------------------

    def _entropy(self, y):
        """
        Formula to compute the entropy for multiple classes: H = -sum(p_i * log2(p_i))

        Parameters
        ----------
        y: np.ndarray, shape (n,)
            Class labels at this node.

        Returns
        -------
        float
            Entropy value in [0, log2(n_classes)].
        """
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        # Only sum where p > 0 to avoid log2(0)
        return -np.sum(probs * np.log2(probs + 1e-12))

    def _gini(self, y):
        """
        Formula to compute the gini index for multiple classes: G = 1 - sum(p_i^2)

        Parameters
        ----------
        y: np.ndarray, shape (n,)
            Class labels at this node.

        Returns
        -------
        float
            Gini value in [0, 1 - 1/n_classes].
        """
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1.0 - np.sum(probs ** 2)

    def _impurity(self, y):
        """Return either the entropy or gini impurity measure based on self.criterion."""
        if self.criterion == "entropy":
            return self._entropy(y)
        return self._gini(y)

    def _weighted_impurity(self, y_left, y_right):
        """
        Weighted average impurity at the child nodes after a split.

        Parameters
        ----------
        y_left, y_right: np.ndarray
            Class labels in the left and right partitions.

        Returns
        -------
        float
            Weighted sum of child impurities.
        """
        p = float(len(y_left)) / (len(y_left) + len(y_right))
        return p * self._impurity(y_left) + (1 - p) * self._impurity(y_right)


    # ------------------------------------------------------------------
    #   Split methods
    # ------------------------------------------------------------------

    def _best_split(self, X, y):
        """
        Splits the data on the feature index and threshold that produce the
        lowest weighted impurity across all features and unique split values.

        Parameters
        ----------
        X: np.ndarray, shape (n_samples, n_features)
        y: np.ndarray, shape (n_samples,)

        Returns
        -------
        best_feature: int or None
            Index of the best feature.
        best_threshold: float or None
            Threshold value of the best split.
        """
        best_feature = None
        best_threshold = None
        best_impurity = np.inf

        # Iterate across all features
        for feature_idx in range(X.shape[1]):
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
    #   Tree construction
    # ------------------------------------------------------------------

    def _build_tree(self, X, y, depth):
        """
        Recursively build the tree, returning either a leaf node or decision node at each level.

        Parameters
        ----------
        X: np.ndarray, shape (n_samples, n_features)
        y: np.ndarray, shape (n_samples,)
        depth: int
            Current depth in the tree.

        Returns
        -------
        _Leaf or _DecisionNode
        """

        # Base case 1: max depth reached
        if depth >= self.max_depth:
            return _Leaf(y)

        # Base case 2: too few samples to split
        if len(y) < self.min_samples_split:
            return _Leaf(y)

        # Base case 3: node is pure (impurity == 0)
        if self._impurity(y) == 0:
            return _Leaf(y)

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)

        # Base case 4: no valid split found
        if best_feature is None:
            return _Leaf(y)

        # Partition data and recurse
        left_vals = X[:, best_feature] <= best_threshold
        right_vals = X[:, best_feature] > best_threshold


        left_subtree = self._build_tree(X[left_vals], y[left_vals], depth + 1)
        right_subtree = self._build_tree(X[right_vals], y[right_vals], depth + 1)

        return _DecisionNode(best_feature, best_threshold, left_subtree, right_subtree)


    # ------------------------------------------------------------------
    #   Tree predictions
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Fit the decision tree to training data.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training feature matrix.
        y: array-like, shape (n_samples,)
            Target class labels.

        Returns
        -------
        self
        """
        X = np.array(X)
        y = np.array(y)

        self.n_features_ = X.shape[1]
        self.n_classes_ = len(np.unique(y))

        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    def _predict_one(self, x, node):
        """
        Run the tree for a single training sample.

        Parameters
        ----------
        x  np.ndarray, shape (n_features,)
        node: _Leaf or _DecisionNode

        Returns
        -------
        Predicted class label.
        """
        # Check if the node is a leaf node - if so, return the predicted class
        if isinstance(node, _Leaf):
            return node.prediction

        # Decision node case
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)

        Returns
        -------
        y_pred: np.ndarray, shape (n_samples,)
            Predicted class labels.
        """

        if self.tree_ is None:
            raise RuntimeError("Call fit() before predict().")

        X = np.array(X)
        return np.array([self._predict_one(x, self.tree_) for x in X])


    # ------------------------------------------------------------------
    #   Evaluation
    # ------------------------------------------------------------------

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


    # ------------------------------------------------------------------
    #   Print tree
    # ------------------------------------------------------------------

    def print_tree(self, node = None, spacing = "", feature_names = None):
        """
        Print a readable representation of the tree.

        Parameters
        ----------
        node: _Leaf or _DecisionNode, optional
            Starting node. Defaults to root (self.tree_).
        spacing: str
            Indentation prefix used during recursion.
        feature_names: list of str, optional
            Names for each feature column. Defaults to 'Feature 0', etc.
        """
        if self.tree_ is None:
            raise RuntimeError("Call fit() before print_tree().")

        if node is None:
            node = self.tree_

        if isinstance(node, _Leaf):
            print(spacing + f"Predict: {node.prediction}  {node.class_counts}")
            return

        fname = (feature_names[node.feature_idx]
                 if feature_names else f"Feature {node.feature_idx}")
        print(spacing + f"[{fname} <= {node.threshold:.4f}]")

        print(spacing + "  ---> True")
        self.print_tree(node.left, spacing + "     ", feature_names)

        print(spacing + "  ---> False")
        self.print_tree(node.right, spacing + "     ", feature_names)



# ------------------------------------------------------------------
#   Leaf node class for DecisionTreeRegressor
# ------------------------------------------------------------------

class _Rleaf:
    """
    A teminal node that stores the predicted value for regression.

    Attributes:
    ----------
    prediction: float
        Mean of target values among training samples that reached this leaf.
    n_samples: int
        Number of training samples that reached this leaf.
    """
    # _DecisionNode is shared between classification and regression trees -
    # it only stores a feature index, splitting threshold, and two child nodes

    def __init__(self, y):
        """
        Initialize the leaf node class for the DecisionTreeRegressor.

        Parameters
        ----------
        y: np.ndarray, shape (n_samples,)
            Target class values of all training samples that reached
            this node. Used to determine the average target value.
        """  
        
        self.prediction = float(np.mean(y))
        self.n_samples = len(y)



# ------------------------------------------------------------------
#   Main decision tree regressor
# ------------------------------------------------------------------

class DecisionTreeRegressor(object):
    """
    A decision tree regressor.

    At each leaf, the prediction is the mean of training target values
    that reached that leaf. Splits are chosen to minimize the weighted MSE.

    Parameters
    ----------
    max_depth: int, default = 3
        Maximum depth of the tree. Growth stops when this depth is reached.
    min_samples_split: int, default = 2
        Minimumn number of samples required at a node to attempt a split.

    Attributes
    ----------
    tree_: _DecisionNode or _Rleaf
        The fitted tree structure. Available after calling fit().
    n_features_: int
        Number of features seen during training.
    n_outputs_: int
        Always 1 for this single-output regressor.

    Methods
    -------
    fit(X, y)
        Fit the regressor to training data.
    predict(X)
        Predict continuous target values for input samples.
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
    print_tree(node, spacing, feature_names)
        Print a readable representation of the fitted tree.

    Examples
    --------
    >>> import numpy as np
    >>> from decision_tree import DecisionTreeRegressor
    >>> X = np.array([[1.0], [2.0], [3.0], [4.0]])
    >>> y = np.array([1.5, 2.5, 3.5, 4.5])
    >>> reg = DecisionTreeRegressor(max_depth=3)
    >>> reg.fit(X, y)
    >>> reg.predict(X)
    array([1.5, 2.5, 3.5, 4.5])

    """

    def __init__(self, max_depth = 3, min_samples_split = 2):
        """
        Initialize the decision tree regressor.

        Parameters
        ----------
        max_depth: int, optional
            Maximum depth of the tree. Growth stops once this depth is
            reached. Defaults to 3.
        min_samples_split: int, optional
            Minimum number of samples required at a node to attempt a
            split. Nodes with fewer samples are converted to leaves.
            Defaults to 2.
        """
    
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.tree_ = None
        self.n_features_ = None
        self.n_outputs_ = 1


    # ------------------------------------------------------------------
    #   Impurity measure
    # ------------------------------------------------------------------

    # Function to compute mean-squared error to measure impurity
    def _mse(self, y):
        """
        Returns mean-squared error between actual y values and predicted y values

        This is the impurity measure for regression trees. A node whose targets
        are all identical has an MSE of 0, meaning it is a pure node. A higher
        MSE indicates a higher impurity.

        Parameters
        ----------
        y: np.ndarray, shape (n,)
            Target values at this node.

        Returns
        -------
        float
            MSE value >= 0.

        """

        y = np.array(y)
        y_hat = np.mean(y)
        return float(np.mean((y - y_hat) ** 2))

    # Function to compute the weighted MSE
    def _weighted_mse(self, y_left, y_right):
        """
        Weighted average MSE at the child nodes after a split.

        Parameters
        ----------
        y_left, y_right: np.ndarray
            Target values in each partition.

        Returns
        -------
        float
            Weighted sum of child MSEs.

        """

        p = float(len(y_left)) / (len(y_left) + len(y_right))
        return p * self._mse(y_left) + (1 - p) * self._mse(y_right)


    # ------------------------------------------------------------------
    #   Split methods
    # ------------------------------------------------------------------

    # Function to return the parameters that achieve the best split
    def _best_split(self, X, y):
        """
        Splits the data on the feature index and threshold that produce the
        lowest weighted MSE across all features and unique split values.

        Parameters
        ----------
        X: np.ndarray, shape (n_samples, n_features)
        y: np.ndarray, shape (n_samples,)

        Returns
        -------
        best_feature: int or None
            Index of the best feature.
        best_threshold: float or None
            Threshold value of the best split.

        """

        best_feature = None
        best_threshold = None
        best_mse = np.inf

        # Iterate across all features
        for feature_idx in range(X.shape[1]):
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
    #   Tree construction
    # ------------------------------------------------------------------

    def _build_tree(self, X, y, depth):
            """
            Recursively build the tree, returning either a leaf node or decision node at each level.

            Parameters
            ----------
            X: np.ndarray, shape (n_samples, n_features)
            y: np.ndarray, shape (n_samples,)
            depth: int
                Current depth in the tree.

            Returns
            -------
            _Rleaf or _DecisionNode

            """

            # Base case 1: max depth reached
            if depth >= self.max_depth:
                return _Rleaf(y)

            # Base case 2: too few samples to split
            if len(y) < self.min_samples_split:
                return _Rleaf(y)

            # Base case 3: node is pure (impurity == 0)
            if self._mse(y) == 0:
                return _Rleaf(y)

            # Find the best split
            best_feature, best_threshold = self._best_split(X, y)

            # Base case 4: no valid split found
            if best_feature is None:
                return _Rleaf(y)

            # Partition data and recurse
            left_vals = X[:, best_feature] <= best_threshold
            right_vals = X[:, best_feature] > best_threshold


            left_subtree = self._build_tree(X[left_vals], y[left_vals], depth + 1)
            right_subtree = self._build_tree(X[right_vals], y[right_vals], depth + 1)

            return _DecisionNode(best_feature, best_threshold, left_subtree, right_subtree)


    # ------------------------------------------------------------------
    #   Tree predictions
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Fit the decision tree to training data.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training feature matrix.
        y: array-like, shape (n_samples,)
            Continuous target values.

        Returns
        -------
        self

        """

        X = np.array(X)
        y = np.array(y, dtype = float)

        self.n_features_ = X.shape[1]
        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    def _predict_one(self, x, node):
        """
        Run the tree for a single training sample.

        Parameters
        ----------
        x: np.ndarray, shape (n_features,)
        node: _Rleaf or _DecisionNode

        Returns
        -------
        Predicted target value.
        """

        # Check if the node is a leaf node - if so, return the predicted class
        if isinstance(node, _Rleaf):
            return node.prediction

        # Decision node case
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        """
        Predict target values for samples in X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)

        Returns
        -------
        y_pred: np.ndarray, shape (n_samples,)
            Predicted target values.
        """

        if self.tree_ is None:
            raise RuntimeError("Call fit() before predict().")

        X = np.array(X)
        return np.array([self._predict_one(x, self.tree_) for x in X])


    # ------------------------------------------------------------------
    #   Evaluation
    # ------------------------------------------------------------------

    def score(self, X, y):
        """
        Return the R-squared score on the given data.

        R-squared = 1 - sse / ssto, where sse is the sum of squared
        residuals and ssto is the total sum of squares. A score of 1.0
        means perfect prediction; 0.0 means the model does no better than
        predicting the mean of y.

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


    # ------------------------------------------------------------------
    #   Print tree
    # ------------------------------------------------------------------

    def print_tree(self, node=None, spacing="", feature_names=None):
        """
        Print a eadable representation of the tree.

        Parameters
        ----------
        node: _Rleaf or _DecisionNode, optional
            Starting node. Defaults to root (self.tree_).
        spacing: str
            Indentation prefix used during recursion.
        feature_names: list of str, optional
            Names for each feature column. Defaults to 'Feature 0', etc.
        """

        if self.tree_ is None:
            raise RuntimeError("Call fit() before print_tree().")

        if node is None:
            node = self.tree_

        if isinstance(node, _Rleaf):
            print(spacing + f"Predict: {node.prediction:.4f}  (n={node.n_samples})")
            return

        fname = (feature_names[node.feature_idx]
                 if feature_names else f"Feature {node.feature_idx}")
        print(spacing + f"[{fname} <= {node.threshold:.4f}]")

        print(spacing + "  ---> True")
        self.print_tree(node.left, spacing + "     ", feature_names)

        print(spacing + "  ---> False")
        self.print_tree(node.right, spacing + "     ", feature_names)
