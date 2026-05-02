""" logistic_regression.py """

import numpy as np

from ..utils import classification_metrics as metrics

class LogisticRegression(object):
    """
    Single-neuron-based approach to logistic regression, trained with 
    full-batch gradient descent.

    Supports both binary classification (sigmoid output, binary
    cross-entropy loss) and multinomial classification (softmax output,
    categorical cross-entropy loss). The task type is specified at
    training time rather than initialization.

    Attributes
    ----------
    task: {'binary', 'multinomial'} or None
        The classification task, set when ``train()`` is called.
    weights: numpy.ndarray or None
        Weight vector of shape ``(n_features,)`` for binary tasks, or
        weight matrix of shape ``(n_features, n_classes)`` for multinomial
        tasks. Set during training.
    bias: float or numpy.ndarray or None
        Scalar bias for binary tasks, or bias vector of shape
        ``(n_classes,)`` for multinomial tasks. Set during training.
    losses: list of float
        Cross-entropy loss recorded at the end of each training epoch.

    Methods
    -------
    train(X, y, eta, epochs, task)
        Fit the model to training data.
    predict(X)
        Predict class labels.
    confusion_matrix(X, y)
        Return the confusion matrix.
    accuracy(X, y)
        Compute the fraction of correctly classified samples.
    precision(X, y, average)
        Compute precision with configurable averaging.
    recall(X, y, average)
        Compute recall with configurable averaging.
    f1_score(X, y, average)
        Compute the F1 score with configurable averaging.
    """

# ------------------------------------------------------------------
#   Initialize network
# ------------------------------------------------------------------

    def __init__(self):
        """ Initialize the logistic regression model with unset parameters. """

        self.task = None
        self.weights = None
        self.bias = None
        self.losses = []


# ------------------------------------------------------------------
#   Activation function
# ------------------------------------------------------------------

    # Define the sigmoid activation function to handle binary regression
    def sigmoid(self, z):
        """
        Apply the sigmoid activation function element-wise.

        Computes ``1 / (1 + exp(-z))``, with input clipping to [-500, 500]
        to prevent overflow. Used as the output activation for binary
        classification.

        Parameters
        ----------
        z: numpy.ndarray, shape (n_samples,)
            Pre-activation values.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Predicted probabilities in the range (0, 1).
        """   
        
        return 1.0/(1.0 + np.exp(-np.clip(z, -500, 500)))

    # Define the softmax activation function to handle multinomial regression
    def softmax(self, z):
        """
        Apply the softmax activation function row-wise.

        Computes ``exp(z) / sum(exp(z))`` along axis 1 (i.e., over classes
        for each sample). The input is shifted by its row-wise maximum
        before exponentiation for numerical stability. Used as the output
        activation for multinomial classification.

        Parameters
        ----------
        z: numpy.ndarray, shape (n_samples, n_classes)
            Pre-activation values.

        Returns
        -------
        numpy.ndarray, shape (n_samples, n_classes)
            Class probability distributions, where each row sums to 1.
        """    
       
        exp = np.exp(z - np.max(z, axis = 1, keepdims = True))
        return exp / np.sum(exp, axis = 1, keepdims = True)


# ------------------------------------------------------------------
#   Forward pass
# ------------------------------------------------------------------

    #Define a function to compute the feedforward values using a sigmoid activations function (y_pred)
    def feedforward(self, X):
        """
        Compute predicted probabilities for input data.

        Applies sigmoid for binary tasks and softmax for multinomial tasks.
        Raises ``RuntimeError`` if called before ``train()``.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_samples,)`` of probabilities for binary tasks, or
            shape ``(n_samples, n_classes)`` for multinomial tasks.

        Raises
        ------
        RuntimeError
            If the model has not yet been trained.
        """    
        
        if self.weights is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        
        # Matrix multiply the inputs and weights, then add the bias term
        z = X @ self.weights + self.bias
        if self.task == "binary":
            return self.sigmoid(z)
        else:
            return self.softmax(z)


# ------------------------------------------------------------------
#   Cost
# ------------------------------------------------------------------

    # Define the cross-entropy cost function and its derivative
    def _cost(self, a, y):
        """
        Compute the cross-entropy loss for the current task.

        For binary tasks, evaluates the binary cross-entropy formula with
        ``numpy.nan_to_num`` applied for numerical stability. For multinomial
        tasks, evaluates the categorical cross-entropy loss.

        Parameters
        ----------
        a: numpy.ndarray
            Predicted probabilities. Shape ``(n_samples,)`` for binary
            tasks or ``(n_samples, n_classes)`` for multinomial tasks.
        y: numpy.ndarray
            Observed labels. Shape ``(n_samples,)`` for binary tasks
            or one-hot encoded with shape ``(n_samples, n_classes)`` for
            multinomial tasks.

        Returns
        -------
        float
            Scalar cross-entropy loss averaged over all samples.
        """    
        
        # Formula for binary cross-entropy loss
        if self.task == "binary":
            return np.mean(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))
        # Formula for categorical cross-entropy loss
        else:
            return -np.mean(np.sum(y * np.nan_to_num(np.log(a)), axis = 1))

    def _cost_delta(self, a, y):
        """
        Compute the output-layer error signal for backpropagation.

        For cross-entropy paired with sigmoid or softmax output activations,
        the gradient simplifies to ``(a - y)``, canceling the activation
        derivative and yielding a clean update rule.

        Parameters
        ----------
        a: numpy.ndarray
            Predicted probabilities.
        y: numpy.ndarray
            Observed target labels.

        Returns
        -------
        numpy.ndarray
            Element-wise error signal ``(a - y)``, same shape as ``a``.
        """    
        
        return (a - y)


# ------------------------------------------------------------------
#   Training
# ------------------------------------------------------------------

    # Define a function to train the model using gradient descent
    def train(self, X, y, eta = 0.01, epochs = 1000, task = None):
        """
        Train the model using full-batch gradient descent.

        Initializes weights and bias (with Xavier scaling for multinomial
        tasks) and iteratively minimizes cross-entropy loss. The loss at
        each epoch is appended to ``self.losses``.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Training input data.
        y: numpy.ndarray
            Observed target labels. Shape ``(n_samples,)`` of binary labels
            (0 or 1) for binary tasks, or shape ``(n_samples, n_classes)``
            of one-hot encoded labels for multinomial tasks.
        eta: float, optional
            Learning rate. Defaults to 0.01.
        epochs: int, optional
            Number of full passes over the training data. Defaults to 1000.
        task: {'binary', 'multinomial'}
            Classification task type. Raises ``ValueError`` if not one of
            the accepted values.

        Returns
        -------
        self: LogisticRegression
            The fitted model instance.

        Raises
        ------
        ValueError
            If ``task`` is not 'binary' or 'multinomial'.
        """    
        
        if task not in {"binary", "multinomial"}:
            raise ValueError("Task must be one of 'binary' or 'multinomial'")

        # Initialize the task, number of samples, and number of features, 
        self.task = task
        n, p = X.shape

        if task == "binary":
            self.weights = np.random.randn(p)
            self.bias = np.random.randn()
        else:
            # For multinomial regression, y should be one-hot encoded with shape (n, k)
            # p is number of features in X, k is number of classes in y
            k = y.shape[1]
            self.weights = np.random.randn(p, k) / np.sqrt(p)
            self.bias = np.random.randn(k)


        # Perform forward pass and calculate derivative of the cost
        for _ in range(epochs):
            y_hat = self.feedforward(X)
            delta = self._cost_delta(y_hat, y)

            # Gradients of the weights and bias
            nabla_w = (X.T @ delta) / n
            nabla_b = np.mean(delta, axis = 0)

            # Update the weights and bias
            self.weights -= eta * nabla_w
            self.bias -= eta * nabla_b

            self.losses.append(self._cost(y_hat, y))

        return self


# ------------------------------------------------------------------
#   Prediction
# ------------------------------------------------------------------

    # Define a method to predict binary or multiple classes
    def predict(self, X):
        """
        Predict class labels for input samples.

        For binary tasks, thresholds the sigmoid output at 0.5. For
        multinomial tasks, returns the index of the highest softmax
        probability for each sample.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Predicted class labels. Binary values (0 or 1) for binary
            tasks, or integer class indices for multinomial tasks.
        """     
        
        y_hat = self.feedforward(X)
        if self.task == "binary":
            return (y_hat >= 0.5).astype(int)
        else:
            return np.argmax(y_hat, axis = 1)


# ------------------------------------------------------------------
#   Evaluation
# ------------------------------------------------------------------

    def _true_labels(self, y):
        """Return integer class labels for the current task."""
        if self.task == "multinomial" and np.asarray(y).ndim == 2:
            return np.argmax(y, axis=1)
        return np.asarray(y)

    #Define a function to compute the confusion matrix
    def confusion_matrix(self, X, y):
        """
        Compute the confusion matrix.

        For binary tasks, returns a 2x2 matrix arranged as::

            [[TN, FP],
             [FN, TP]]

        where the positive class is 1 and the negative class is 0.
        For multinomial tasks, returns a ``k x k`` matrix where entry
        ``[i, j]`` is the number of samples with true class ``i``
        predicted as class ``j``. For multinomial tasks, ``y`` should
        contain integer class labels (not one-hot encoded).

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Input data.
        y: numpy.ndarray, shape (n_samples,)
            Observed target labels.

        Returns
        -------
        numpy.ndarray, shape (2, 2) or (k, k)
            Confusion matrix.
        """    
        
        y = self._true_labels(y)
        labels = [0, 1] if self.task == "binary" else None
        return metrics.confusion_matrix(y, self.predict(X), labels=labels)

    # Define a method to get the accuracy score
    def accuracy(self, X, y):
        """
        Compute the proportion of correctly classified samples.

        For multinomial tasks, one-hot encoded ``y`` is converted to
        integer class labels before comparison.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Input data.
        y: numpy.ndarray
            Observed target labels. Shape ``(n_samples,)`` for binary tasks,
            or one-hot encoded with shape ``(n_samples, n_classes)`` for
            multinomial tasks.

        Returns
        -------
        float
            Proportion of samples correctly classified, in [0, 1].
        """    
        
        return metrics.accuracy(self._true_labels(y), self.predict(X))

    # Define a method to get the precision score
    def precision(self, X, y, average = "macro"):
        """
        Compute precision with configurable class averaging.

        For binary tasks the ``average`` parameter is ignored and precision
        is computed directly from the 2x2 confusion matrix. For multinomial
        tasks, per-class precision scores are combined using the chosen
        averaging strategy.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Input data.
        y: numpy.ndarray, shape (n_samples,)
            Observed integer class labels.
        average: {'macro', 'weighted', 'micro'}, optional
            Averaging strategy for multinomial tasks. 'macro' computes the
            unweighted mean across classes; 'weighted' computes the weighted
            average across classes; 'micro' aggregates counts globally before
            computing the score. Defaults to 'macro'.

        Returns
        -------
        float
            Precision score. Returns 0.0 for any class with no positive
            predictions.

        Raises
        ------
        ValueError
            If ``average`` is not one of the accepted values.
        """    
        
        y = self._true_labels(y)
        if self.task == "binary":
            return metrics.precision(
                y, self.predict(X), average="binary", labels=[0, 1],
                positive_label=1,
            )
        return metrics.precision(y, self.predict(X), average=average)

    # Define a method to get the recall score
    def recall(self, X, y, average = "macro"):
        """
        Compute recall with configurable class averaging.

        For binary tasks the ``average`` parameter is ignored and recall
        is computed directly from the 2x2 confusion matrix. For multinomial
        tasks, per-class recall scores are combined using the chosen
        averaging strategy.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Input data.
        y: numpy.ndarray, shape (n_samples,)
            Observed integer class labels.
        average: {'macro', 'weighted', 'micro'}, optional
            Averaging strategy for multinomial tasks. 'macro' computes the
            unweighted mean across classes; 'weighted' computes the weighted
            average across classes; 'micro' aggregates counts globally before
            computing the score. Defaults to 'macro'.

        Returns
        -------
        float
            Recall score. Returns 0.0 for any class with no actual positive
            samples.

        Raises
        ------
        ValueError
            If ``average`` is not one of the accepted values.
        """     
        
        y = self._true_labels(y)
        if self.task == "binary":
            return metrics.recall(
                y, self.predict(X), average="binary", labels=[0, 1],
                positive_label=1,
            )
        return metrics.recall(y, self.predict(X), average=average)

    #Define a method to get the f1 score
    def f1_score(self, X, y, average = "macro"):
        """
        Compute the F1 score with configurable class averaging.

        For binary tasks the ``average`` parameter is ignored and the F1
        score is computed as the harmonic mean of precision and recall.
        For multinomial tasks, per-class F1 scores are combined using the
        chosen averaging strategy. The micro-averaged F1 is equivalent to
        micro-averaged precision (and recall) when all classes are considered.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Input data.
        y: numpy.ndarray, shape (n_samples,)
            Observed integer class labels.
        average : {'macro', 'weighted', 'micro'}, optional
            Averaging strategy for multinomial tasks. Defaults to 'macro'.

        Returns
        -------
        float
            F1 score. Returns 0.0 for any class where both precision and
            recall are zero.

        Raises
        ------
        ValueError
            If ``average`` is not one of the accepted values.
        """
   
        y = self._true_labels(y)
        if self.task == "binary":
            return metrics.f1_score(
                y, self.predict(X), average="binary", labels=[0, 1],
                positive_label=1,
            )
        return metrics.f1_score(y, self.predict(X), average=average)



