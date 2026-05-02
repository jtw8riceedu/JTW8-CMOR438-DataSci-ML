""" perceptron.py """

import numpy as np

from ..utils import classification_metrics as metrics

class Perceptron(object):
    """
    A single-layer perceptron trained with gradient descent.

    Implements a single neuron with a configurable activation function,
    trained by minimizing mean squared error via full-batch gradient
    descent. Supports ReLU, Leaky ReLU, tanh, and unit-step activations.
    Intended for binary classification tasks where labels are +1 and -1.

    Parameters
    ----------
    activation: {'relu', 'leakyrelu', 'tanh', 'unitstep'}
        Activation function to apply to the pre-activation output.
        Raises ``ValueError`` if not one of the accepted values.

    Attributes
    ----------
    weights: numpy.ndarray or None
        Weight vector of shape ``(n_features,)``, set during training.
    bias: float or None
        Scalar bias term, set during training.
    losses: list of float
        MSE loss recorded at the end of each training epoch.

    Methods
    -------
    train(X, y, eta, epochs)
        Fit the perceptron to training data.
    predict(X)
        Predict binary class labels (+1 or -1).
    accuracy(X, y)
        Compute the fraction of correctly classified samples.
    confusion_matrix(X, y)
        Return the 2x2 confusion matrix.
    precision(X, y)
        Compute precision.
    recall(X, y)
        Compute recall.
    f1_score(X, y)
        Compute the F1 score.
    """
# ------------------------------------------------------------------
#   Initialize network
# ------------------------------------------------------------------

    def __init__(self, activation = None):
        """
        Initialize the perceptron.

        Parameters
        ----------
        activation: {'relu', 'leakyrelu', 'tanh', 'unitstep'}
            Activation function for the output unit. Raises ``ValueError``
            if not one of the accepted values.
        """
        
        if activation not in {"relu", "leakyrelu", "tanh", "unitstep"}:
            raise ValueError("Unknown activation function (should be one of 'relu', 'leakyrelu', 'tanh', or 'unitstep')")
        self.activation = activation
        self.weights = None
        self.bias = None
        self.losses = []


# ------------------------------------------------------------------
#   Activation function
# ------------------------------------------------------------------

    #Define a method containing the activation functions
    def _output_activation(self, z):
        """
        Apply the specified activation function element-wise.

        Parameters
        ----------
        z: numpy.ndarray, shape (n_samples,)
            Pre-activation values.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Activated output values.

        Raises
        ------
        ValueError
            If ``self.activation`` is not a recognized activation type.
        """
       
        # Relu activation function
        if self.activation == "relu":
            return np.maximum(0, z)
        
        # Leaky Relu activation function
        elif self.activation == "leakyrelu":
            return np.where(z >= 0, z, 0.01 * z)
        
        # Tanh activation function
        elif self.activation == "tanh":
            return np.tanh(z)
        
        # Unit-step activation function
        elif self.activation == "unitstep":
            return np.where(z >= 0, 1, -1)
        
        else:
            raise ValueError("Unknown activation function (should be one of 'relu', 'leakyrelu', 'tanh', or 'unitstep')")

    # Define a method containing the derivatives of the activation functions
    def _activation_derivative(self, z):
        """
        Compute the derivative of the specified activation function element-wise.

        For the unit-step activation, the derivative is approximated as 1
        everywhere to allow gradient-based training.

        Parameters
        ----------
        z: numpy.ndarray, shape (n_samples,)
            Pre-activation values.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Element-wise derivatives of the activation function.
        """   
        
        # Relu activation derivative
        if self.activation == "relu":
            return np.where(z >= 0, 1.0, 0.0)
       
        # Leaky Relu activation function derivative
        elif self.activation == "leakyrelu":
            return np.where(z >= 0, 1.0, 0.01)
        
        # Tanh activation function derivative
        elif self.activation == "tanh":
            return 1 - np.tanh(z) ** 2
       
        # Unit-step activation function derivative
        elif self.activation == "unitstep":
            return np.ones_like(z)

# ------------------------------------------------------------------
#   Forward pass
# ------------------------------------------------------------------

    # Define a function to compute the feedforward values (y_pred)
    def feedforward(self, X):
        """
        Perform a forward pass and return the output activations.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Activated output values.
        """
        
        # Matrix multiply the inputs and weights, then add the bias term
        z =  X @ self.weights + self.bias
        return self._output_activation(z)


# ------------------------------------------------------------------
#   Cost
# ------------------------------------------------------------------

    # Define a function for MSE cost
    def _cost(self, y_hat, y):
        """
        Compute the mean squared error between predictions and targets.

        Parameters
        ----------
        y_hat: numpy.ndarray, shape (n_samples,)
            Predicted output values.
        y: numpy.ndarray, shape (n_samples,)
            Observed target values.

        Returns
        -------
        float
            Scalar MSE loss.
        """
        
        return np.mean((y_hat - y) ** 2)

    # Define a function to compute the derivative of the cost function wrt y_hat
    def _cost_delta(self, y_hat, y):
        """
        Compute the gradient of the MSE cost with respect to the predictions.

        Parameters
        ----------
        y_hat: numpy.ndarray, shape (n_samples,)
            Predicted output values.
        y: numpy.ndarray, shape (n_samples,)
            Observed target values.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Element-wise gradient ``(y_hat - y)``.
        """
        
        return y_hat - y


# ------------------------------------------------------------------
#   Training
# ------------------------------------------------------------------

    # Define a function to train the model using gradient descent
    def train(self, X, y, eta = 0.01, epochs = 1000):
        """
        Train the perceptron using full-batch gradient descent.

        Initializes weights and bias from a standard normal distribution,
        then iteratively updates them by backpropagating the MSE gradient
        through the activation function. The loss at each epoch is appended
        to ``self.losses``.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Training input data.
        y: numpy.ndarray, shape (n_samples,)
            Observed target values (+1 or -1).
        eta: float, optional
            Learning rate. Defaults to 0.01.
        epochs : int, optional
            Number of full passes over the training data. Defaults to 1000.

        Returns
        -------
        self: Perceptron
            The fitted perceptron instance.
        """
        # Initialize number of samples, number of features, weights, and bias
        n, p = X.shape
        self.weights = np.random.randn(p)
        self.bias = np.random.randn()

        # Perform forward pass and calculate derivative of the cost
        for _ in range(epochs):
            z = X @ self.weights + self.bias
            y_hat = self._output_activation(z)
            delta = self._cost_delta(y_hat, y) * self._activation_derivative(z)

            # Gradients of the weights and bias
            nabla_w = (X.T @ delta) / n
            nabla_b = np.mean(delta)

            # Update weights and bias
            self.weights -= eta * nabla_w
            self.bias -= eta * nabla_b

            self.losses.append(self._cost(y_hat, y))

        return self


# ------------------------------------------------------------------
#   Evaluation
# ------------------------------------------------------------------

    # Define a function to predict classes
    def predict(self, X):
        """
        Predict binary class labels for input samples.

        For the unit-step activation, the raw output is returned directly.
        For all other activations, predictions are thresholded at zero:
        values >= 0 map to +1 and values < 0 map to -1.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Predicted class labels, each either +1 or -1.
        """      
        
        y_hat = self.feedforward(X)
        if self.activation == "unitstep":
            return y_hat
        else:
            return np.where(y_hat >= 0, 1, -1)

    # Define an accuracy function
    def accuracy(self, X, y):
        """
        Compute the proportion of correctly classified samples.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Input data.
        y: numpy.ndarray, shape (n_samples,)
            Observed target values (+1 or -1).

        Returns
        -------
        float
            Proportion of samples correctly classified, in [0, 1].
        """
        
        return metrics.accuracy(y, self.predict(X))

    # Define a confusion matrix function
    def confusion_matrix(self, X, y):
        """
        Compute the 2x2 confusion matrix.

        The matrix is arranged as:

            [[TN, FP],
             [FN, TP]]

        where the positive class is +1 and the negative class is -1.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Input data.
        y: numpy.ndarray, shape (n_samples,)
            Observed target values (+1 or -1).

        Returns
        -------
        numpy.ndarray, shape (2, 2)
            Confusion matrix with counts of TN, FP, FN, and TP.
        """
        
        return metrics.confusion_matrix(y, self.predict(X), labels=[-1, 1])

    # Define a method to get the precision score
    def precision(self, X, y):
        """
        Compute precision: TP / (TP + FP). Calculates the proportion of correct
        positive predictions for every positive prediction made.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Input data.
        y: numpy.ndarray, shape (n_samples,)
            Observed target values (+1 or -1).

        Returns
        -------
        float
            Precision score. Returns 0.0 if there are no positive predictions.
        """ 
        
        return metrics.precision(
            y, self.predict(X), average="binary", labels=[-1, 1],
            positive_label=1,
        )

    # Define a method to get the recall score
    def recall(self, X, y):
        """
        Compute recall: TP / (TP + FN). Calculates the proportion of correct
        positive predictions for every actual positive sample.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Input data.
        y: numpy.ndarray, shape (n_samples,)
            Ground-truth labels (+1 or -1).

        Returns
        -------
        float
            Recall score. Returns 0.0 if there are no actual positive samples.
        """ 
        
        return metrics.recall(
            y, self.predict(X), average="binary", labels=[-1, 1],
            positive_label=1,
        )

    # Define a method to get the f1 score
    def f1_score(self, X, y):
        """
        Compute the F1 score: harmonic mean of precision and recall. Examines
        both precision and recall to evaluate model performance.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Input data.
        y: numpy.ndarray, shape (n_samples,)
            Ground-truth labels (+1 or -1).

        Returns
        -------
        float
            F1 score. Returns 0.0 if both precision and recall are zero.
        """ 
        
        return metrics.f1_score(
            y, self.predict(X), average="binary", labels=[-1, 1],
            positive_label=1,
        )


