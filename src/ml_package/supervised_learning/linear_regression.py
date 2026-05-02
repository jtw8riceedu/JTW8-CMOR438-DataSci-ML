""" linear_regression.py """

import numpy as np

from ..utils import regression_metrics as metrics

class LinearRegression(object):
    """
    Single-neuron-based approach to linear regression, trained with 
    full-batch gradient descent.

    Fits a linear model ``y = X @ weights + bias`` by minimizing mean
    squared error. Provides a broad set of evaluation metrics covering
    error magnitude, percentage error, and model selection criteria.

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
        Fit the model to training data.
    feedforward(X)
        Compute predictions for input data.
    predict(X)
        Wrapper to return the feedforward values of the model.
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
    r_squared_adj(X, y)
        Adjusted coefficient of determination.
    aic(X, y)
        Akaike's Information Criterion.
    aicc(X, y)
        Corrected Akaike's Information Criterion.
    bic(X, y)
        Bayesian Information Criterion.
    """

# ------------------------------------------------------------------
#   Initialize network
# ------------------------------------------------------------------

    def __init__(self):
        """ Initialize the linear regression model with unset parameters. """ 
        
        self.weights = None
        self.bias = None
        self.losses = []

# ------------------------------------------------------------------
#   Activation function
# ------------------------------------------------------------------

    # Define a function for a linear output activation
    def _output_activation(self, z):
        """
        Apply a linear activation function.

        Returns the pre-activation values unchanged, corresponding to
        a standard linear output for regression tasks.

        Parameters
        ----------
        z: numpy.ndarray, shape (n_samples,)
            Pre-activation values.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Unchanged input values.
        """ 
        
        return z

# ------------------------------------------------------------------
#   Forward pass
# ------------------------------------------------------------------

    # Define a function to compute the feedforward values using a linear 
    # activation function
    def feedforward(self, X):
        """
        Compute predicted values for input data.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Predicted continuous output values.
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
        
        return metrics.mean_squared_error(y, y_hat)

    # Define a function to compute the derivative of the cost function w.r.t y_hat
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
        Train the model using full-batch gradient descent.

        Initializes weights and bias from a standard normal distribution,
        then iteratively minimizes MSE by updating parameters along the
        negative gradient. The loss at each epoch is appended to
        ``self.losses``.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Training input data.
        y: numpy.ndarray, shape (n_samples,)
            Observed continuous target values.
        eta: float, optional
            Learning rate. Defaults to 0.01.
        epochs: int, optional
            Number of full passes over the training data. Defaults to 1000.

        Returns
        -------
        self: LinearRegression
            The fitted model instance.
        """  
        
        # Initialize the number of samples, number of features, weights, and bias
        n, p = X.shape
        self.weights = np.random.randn(p)
        self.bias = np.random.randn()

         # Perform forward pass and calculate derivative of the cost
        for _ in range(epochs):
            y_hat = self.feedforward(X)
            delta = self._cost_delta(y_hat, y)


            # Gradients of the weights and bias
            nabla_w = (X.T @ delta) / n
            nabla_b = np.mean(delta)

            # Update weights and bias
            self.weights -= eta * nabla_w
            self.bias -= eta * nabla_b

            self.losses.append(self._cost(y_hat, y))

        return self

# ------------------------------------------------------------------
#   Predictions
# ------------------------------------------------------------------

    def predict(self, X):
        """ 
        Wrapper to return the feedforward values of the model. Included for
        consistency with other classes in the package.
        
        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Predicted continuous output values.
        """
        
        return self.feedforward(X)


# ------------------------------------------------------------------
#   Evaluation
# ------------------------------------------------------------------

    # Methods for error evaluation

    # Define a function for root mean-squared error
    def rmse(self, X, y):
        """
        Compute the root mean squared error.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Input data.
        y: numpy.ndarray, shape (n_samples,)
            Observed target values.

        Returns
        -------
        float
            Square root of the mean squared error.
        """ 
        
        return metrics.rmse(y, self.predict(X))

    # Define a function for mean absolute error
    def mae(self, X, y):
        """
        Compute the mean absolute error.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Input data.
        y: numpy.ndarray, shape (n_samples,)
            Observed target values.

        Returns
        -------
        float
            Average absolute difference between predictions and targets.
        """    
        
        return metrics.mae(y, self.predict(X))

    # Define a function for mean absolute percentage error
    def mape(self, X, y):
        """
        Compute the mean absolute percentage error.

        Expresses the average prediction error as a percentage of the
        true values. Note: undefined when any element of ``y`` is zero.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Input data.
        y: numpy.ndarray, shape (n_samples,)
            Observed target values (should be non-zero).

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
        X: numpy.ndarray, shape (n_samples, n_features)
            Input data.
        y: numpy.ndarray, shape (n_samples,)
            Observed target values.

        Returns
        -------
        float
            SMAPE as a percentage.
        """

        return metrics.smape(y, self.predict(X))

    #Define a function for mean absolute scaled error
    def mase(self, X, y):
        """
        Compute the mean absolute scaled error.

        Scales the MAE by the MAE of a naive one-step seasonal random
        walk forecast (i.e., the mean absolute difference of consecutive
        target values). Values less than 1 indicate the model outperforms
        the naive baseline.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Input data.
        y: numpy.ndarray, shape (n_samples,)
            Observed target values.

        Returns
        -------
        float
            MASE score.
        """  
        
        return metrics.mase(y, self.predict(X))


    # Methods for model selection criterion

    # Define a function for r-squared
    def r_squared(self, X, y):
        """
        Compute the coefficient of determination (R-squared).

        Measures the proportion of variance in the target that can be 
        explained by the model. Returns 1.0 for a perfect fit and can 
        be negative for a model worse than predicting the mean.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Input data.
        y: numpy.ndarray, shape (n_samples,)
            Observed target values.

        Returns
        -------
        float
            R-sqaured score.
        """   
        
        return metrics.r_squared(y, self.predict(X))

    # Define a function for r-squared adjusted
    def r_squared_adj(self, X, y):
        """
        Compute the adjusted coefficient of determination (adjusted R-squared).

        Penalizes R-squared for the number of predictors in the model, providing
        a fairer comparison across models with different numbers of features.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Input data.
        y: numpy.ndarray, shape (n_samples,)
            Observed target values.

        Returns
        -------
        float
            Adjusted R-squared score.
        """  
        
        return metrics.adjusted_r_squared(y, self.predict(X), X.shape[1])

    def r_sqaured_adj(self, X, y):
        """Backward-compatible alias for ``r_squared_adj``."""
        return self.r_squared_adj(X, y)

    # Define a method for Akaike's Information Criterion
    def aic(self, X, y):
        """
        Compute Akaike's Information Criterion (AIC).

        Estimates the relative quality of the model by balancing goodness
        of fit against model complexity (number of parameters ``k = p + 1``,
        counting the bias). Lower values indicate a preferred model.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Input data.
        y : numpy.ndarray, shape (n_samples,)
            Observed target values.

        Returns
        -------
        float
            AIC score.
        """   
        
        return metrics.aic(y, self.predict(X), X.shape[1] + 1)

    # Define a method for corrected Akaike's Information Criterion
    def aicc(self, X, y):
        """
        Compute the corrected Akaike's Information Criterion (AICc).

        Applies a correction term to the AIC that reduces bias when the
        number of samples is small relative to the number of parameters.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Input data.
        y: numpy.ndarray, shape (n_samples,)
            Observed target values.

        Returns
        -------
        float
            AICc score.
        """   
        
        return metrics.aicc(y, self.predict(X), X.shape[1] + 1)

    # Define a method for Bayesian Information Criterion
    def bic(self, X, y):
        """
        Compute the Bayesian Information Criterion (BIC).

        Similar to AIC but applies a stronger penalty for model complexity
        that grows logarithmically with the sample size. Generally favors
        simpler models than AIC. Useful for choosing parsimonious models.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Input data.
        y: numpy.ndarray, shape (n_samples,)
            Observed target values.

        Returns
        -------
        float
            BIC score.
        """    
        
        return metrics.bic(y, self.predict(X), X.shape[1] + 1)
