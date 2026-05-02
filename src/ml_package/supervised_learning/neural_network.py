""" neural_network.py """

import numpy as np

# ------------------------------------------------------------------
#   MSE cost class
# ------------------------------------------------------------------

# Define the mean-squared error cost class
class MSE:
    """
    Mean Squared Error (MSE) cost function.

    Computes the average squared Euclidean distance between the network's
    output activations ``a`` and the target values ``y``. A factor of 0.5 
    is included to simplify the gradient expression. Intended usage is for 
    regression tasks with a linear output layer.

    Methods
    -------
    cost(a, y)
        Compute the MSE cost over a batch.
    delta(z, a, y)
        Compute the error delta from the output layer for backpropagation.
    """

    def cost(self, a, y):
        return 0.5 * np.mean(np.linalg.norm(a - y, axis = 0)**2)
        """
        Compute the mean squared error over a batch.

        Calculates the MSE between the output activation ``a`` and the
        target value ``y`` across all samples in the batch, where the 
        norm is taken column-wise (one sample per column).

        Parameters
        ----------
        a: numpy.ndarray, shape (n_outputs, n_samples)
            Predicted output activations from the network.
        y: numpy.ndarray, shape (n_outputs, n_samples)
            Observed target values.

        Returns
        -------
        float
            Scalar MSE cost averaged over all samples in the batch.
        """

    def delta(self, z, a, y):
        """
        Compute the output-layer error signal (delta) for backpropagation.

        For MSE combined with a linear output activation, the gradient of
        the cost with respect to the pre-activation ``z`` simplifies to
        ``(a - y)``, since the derivative of the linear activation is 1.

        Parameters
        ----------
        z: numpy.ndarray, shape (n_outputs, n_samples)
            Pre-activation values of the output layer (unused here, but
            retained for consistentcy with other cost classes).
        a: numpy.ndarray, shape (n_outputs, n_samples)
            Output activations of the network.
        y: numpy.ndarray, shape (n_outputs, n_samples)
            Observed target values.

        Returns
        -------
        numpy.ndarray, shape (n_outputs, n_samples)
            Element-wise error signal ``(a - y)`` for the output layer.
        """
        # We don't include a constant factor out front because it gets
        # absorbed into the learning rate in gradient descent
        return (a - y)


# ------------------------------------------------------------------
#   Cross-entropy cost class
# ------------------------------------------------------------------

# Define the cross-entropy cost class
class CrossEntropy:
    """
    Cross-Entropy cost function.

    Computes the binary or categorical cross-entropy loss between the
    network's output activations and the target labels. Note that 
    ``numpy.nan_to_num`` is used to ensure numerical stability and handle
    edge cases (e.g., ``log(0)``). Intended usage is for binary classification
    tasks with a sigmoid output layer, or for multiple classification tasks
    with a softmax output layer.

    Methods
    -------
    cost(a, y)
        Compute the scalar cross-entropy cost over a batch.
    delta(z, a, y)
        Compute the output-layer error signal for backpropagation.
    """
    
    # Cross-entropy cost
    def cost(self, a, y):
        """
        Compute the cross-entropy cost over a batch.

        Evaluates the binary cross-entropy formula
        ``mean(sum(-y*log(a) - (1-y)*log(1-a)))`` element-wise across
        all output units and samples.

        Parameters
        ----------
        a: numpy.ndarray, shape (n_outputs, n_samples)
            Predicted output activations (probabilities in (0, 1)).
        y: numpy.ndarray, shape (n_outputs, n_samples)
            Observed target values (binary or one-hot encoded).

        Returns
        -------
        float
            Scalar cross-entropy cost averaged over all samples in the batch.
        """
        return np.mean(np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)), axis = 0))
    
    # Derivative of cross-entropy cost
    def delta(self, z, a ,y):
        """
        Compute the output-layer error signal (delta) for backpropagation.

        For cross-entropy paired with a sigmoid or softmax output activation,
        the gradient of the cost with respect to the pre-activation ``z``
        simplifies to ``(a - y)``.

        Parameters
        ----------
        z: numpy.ndarray, shape (n_outputs, n_samples)
            Pre-activation values of the output layer (unused here, but
            retained for a consistent interface with other cost classes).
        a: numpy.ndarray, shape (n_outputs, n_samples)
            Output activations of the network (predicted probabilities).
        y: numpy.ndarray, shape (n_outputs, n_samples)
            Observed target values.

        Returns
        -------
        numpy.ndarray, shape (n_outputs, n_samples)
            Element-wise error signal ``(a - y)`` for the output layer.
        """
        return (a - y)


# ------------------------------------------------------------------
#   Neural network class
# ------------------------------------------------------------------

class NeuralNetwork(object):
    """
    neural_network.py

    A comprehensive feedforward neural network trained with
    mini-batch stochastic gradient descent (SGD) and backpropagation.

    Supports three task types: regression, binary classification, and
    multiclass classification. Each task comes with a matching output 
    activation and default cost function. Hidden layers use sigmoid 
    activations; the output layer uses a linear, sigmoid, or softmax 
    activation depending on the task.

    Parameters
    ----------
    layer_sizes: list of int
        Number of neurons in each layer, from input to output.
        Must contain at least two elements.
    cost: MSE or CrossEntropy, optional
        Cost function instance. Defaults to ``MSE()`` for regression
        and ``CrossEntropy()`` for binary/multiclass classification.
    task: {'regression', 'binary', 'multiclass'}
        Learning task that determines the output activation function
        and default cost. Required; raises ``ValueError`` if omitted
        or invalid.

    Attributes
    ----------
    num_layers: int
        Total number of layers (input + hidden + output).
    layer_sizes: list of int
        Neuron counts per layer as provided at construction time.
    task: str
        The learning task type ('regression', 'binary', or 'multiclass').
    cost: MSE or CrossEntropy
        The active cost function instance.
    weights: list of numpy.ndarray
        Weight matrices for each layer transition, initialized with
        Xavier scaling (``randn / sqrt(n_inputs)``).
    biases: list of numpy.ndarray
        Bias vectors for each layer (excluding the input layer),
        initialized from a standard normal distribution.

    Methods
    -------
    feedforward(a)
        Run a forward pass and return the final output activations.
    SGD(training_data, epochs, mini_batch_size, eta, ...)
        Train the network using mini-batch stochastic gradient descent.
    evaluate_classification(test_data)
        Count correct predictions for classification tasks.
    evaluate_regression(validation_data)
        Compute mean squared error for regression tasks.
    total_cost(validation_data)
        Compute the average cost over a validation set.
    """

# ------------------------------------------------------------------
#   Initialize the network
# ------------------------------------------------------------------

    def __init__(self, layer_sizes, cost = None, task = None):
        """
        Initialize the neural network.

        Validates the provided layer sizes and task type, sets the cost
        function (defaulting based on task if not supplied), and initializes
        weights with Xavier scaling and biases from a standard normal
        distribution.

        Parameters
        ----------
        layer_sizes: list of int
            Contains the number of neurons in each layer of the network. Must 
            have at least 2 elements (an input and an output layer).
        cost: MSE or CrossEntropy, optional
            Cost function instance. Defaults to ``MSE()`` for regression
            and ``CrossEntropy()`` for binary or multiclass classification.
        task: {'regression', 'binary', 'multiclass'}
            The learning task type. Determines the output activation and
            the default cost function. Raises ``ValueError`` if not one of
            the accepted values.

        Raises
        ------
        ValueError
            If ``layer_sizes`` has fewer than 2 elements, or if ``task``
            is not one of 'regression', 'binary', or 'multiclass'.
        """
        if len(layer_sizes) <2:
            raise ValueError("Network must have at least 2 layers (input and output)")

        if task not in {"regression", "binary", "multiclass"}:
            raise ValueError("Task must be one of 'regression', 'binary', or 'multiclass'")

        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.task = task

        #Match regression with MSE cost, and classification tasks with cross-entropy cost
        if cost is None:
            if task == "regression":
                cost = MSE()
            else:
                cost = CrossEntropy()
        self.cost = cost

        #Initialize the weights and biases randomly
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

        
# ------------------------------------------------------------------
#   Activation functions
# ------------------------------------------------------------------
    
    # Define the sigmoid activation function
    def sigmoid(self, z):
        """
        Apply the sigmoid activation function element-wise.

        Computes ``1 / (1 + exp(-z))``, with input clipping to [-500, 500]
        to prevent overflow in the exponential.

        Parameters
        ----------
        z: numpy.ndarray
            Pre-activation values of any shape.

        Returns
        -------
        numpy.ndarray
            Activation values in the range (0, 1), same shape as ``z``.
        """
        return 1.0/(1.0 + np.exp(-np.clip(z, -500, 500)))

    # Define the derivative of the sigmoid activation function
    def d_sigmoid(self, z):
        """
        Compute the derivative of the sigmoid function element-wise.

        Evaluates ``sigmoid(z) * (1 - sigmoid(z))``, which is to be used 
        during backpropagation.

        Parameters
        ----------
        z: numpy.ndarray
            Pre-activation values of any shape.

        Returns
        -------
        numpy.ndarray
            Element-wise sigmoid derivatives, same shape as ``z``.
        """
        sig = self.sigmoid(z)
        return sig * (1.0 - sig)


    # Define the softmax activation function
    def softmax(self, z):
        """
        Apply the softmax activation function column-wise.

        Computes ``exp(z) / sum(exp(z))`` along axis 0 (i.e., over output
        units for each sample). The input is shifted by its column-wise
        maximum before exponentiation for numerical stability.

        Parameters
        ----------
        z: numpy.ndarray, shape (n_outputs, n_samples)
            Pre-activation values of the output layer.

        Returns
        -------
        numpy.ndarray, shape (n_outputs, n_samples)
            Class probability distributions, where each column sums to 1.
        """    
        exp = np.exp(z - np.max(z, axis = 0, keepdims = True))
        return exp / np.sum(exp, axis = 0, keepdims = True)


    def _output_activation(self, z):
        """
        Match the appropriate output activation with the specified task.

        Selects the activation function for the final layer based on
        ``self.task``: linear for regression, sigmoid for binary 
        classification, and softmax for multiclass classification.

        Parameters
        ----------
        z: numpy.ndarray, shape (n_outputs, n_samples)
            Pre-activation values of the output layer.

        Returns
        -------
        numpy.ndarray, shape (n_outputs, n_samples)
            Activated output values.

        Raises
        ------
        ValueError
            If ``self.task`` is not one of the recognized task types.
        """
       
        # Linear output for regression
        if self.task == "regression":
            return z
        # Sigmoid output for binary classification
        elif self.task == "binary":
            return self.sigmoid(z)
        # Softmax output for multiple classification
        elif self.task == "multiclass":
            return self.softmax(z)
        else:
            raise ValueError("Unknown task type (should be one of 'regression', 'binary', or 'multiclass')")


# ------------------------------------------------------------------
#   Forward pass
# ------------------------------------------------------------------

    # Public forward pass method used for evaluation (returns final output only)
    # Apply a linear output layer to be able to handle regression tasks
    def feedforward(self, a):
        """
        Perform a forward pass through the network; return the output of the
        network if ``a`` is input.

        Propagates input ``a`` through all layers, applying sigmoid
        activations to all hidden layers and the appropriate activation
        to the output layer. This public method returns only the final
        output and is intended for inference and evaluation.

        For classification tasks, the method returns the raw network output 
        (probabilities). Thus, post-processing must be applied manually. For
        binary classification, run ``int(output > 0.5)``, and for multiple
        classification, run ``np.argmax(output)``.

        Parameters
        ----------
        a: numpy.ndarray, shape (n_inputs, n_samples)
            Input data, where each column is one sample.

        Returns
        -------
        numpy.ndarray, shape (n_outputs, n_samples)
            Output activations of the final layer.
        """
        
        # Pre-activation values are calculated as ``np.dot(w, a) + b
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = np.dot(w, a) + b
            if i < len(self.weights) - 1:
                a = self.sigmoid(z)
            else:
                a = self._output_activation(z)
        return a


    # Internal forward pass method (saves all activation and pre-activation 
    # values for backpropagation)
    def _forward_pass(self, a):
        """
        Perform an internal forward pass, saving intermediate values.

        Propagates input ``a`` through all layers while storing every
        activation and pre-activation (z) value. These cached values are
        required by the backpropagation algorithm to compute gradients.

        Parameters
        ----------
        a: numpy.ndarray, shape (n_inputs, n_samples)
            Input data, where each column is one sample.

        Returns
        -------
        activations: list of numpy.ndarray
            Activations at every layer, including the input. Length is
            ``num_layers`` (i.e., one entry per layer).
        z_vals: list of numpy.ndarray
            Pre-activation values at every layer except the input. Length
            is ``num_layers - 1``.
        """
        activations = [a]
        z_vals = []

        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = np.dot(w, a) + b
            z_vals.append(z)
            if i < len(self.weights) - 1:
                a = self.sigmoid(z)
            else:
                a = self._output_activation(z)
            activations.append(a)
        return activations, z_vals


# ------------------------------------------------------------------
#   Training
# ------------------------------------------------------------------

    # Define a stochastic gradient descent method
    # Randomly shuffle training data, partition into mini batches, and compute 
    # a single step of gradient descent for each mini batch.
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data=None, monitor_cost=False):
        """
        Train the network using mini-batch stochastic gradient descent.

        In each epoch, the code starts by randomly shuffling the training data,
        and then it partitions the training data into mini batches. Then, for
        each mini-batch, we apply a single step of gradient descent. Optionally 
        evaluates the network on a validation set at the end of every epoch, 
        reporting either validation MSE (regression) or classification accuracy,
        and optionally the total validation cost.

        Parameters
        ----------
        training_data: iterable of (x, y) tuples
            Training samples, where ``x`` is an input column vector and
            ``y`` is the corresponding target.
        epochs: int
            Number of full passes over the training data.
        mini_batch_size: int
            Number of samples per mini-batch.
        eta: float
            Learning rate applied to each weight and bias update.
        validation_data: iterable of (x, y) tuples, optional
            Held-out data used for per-epoch evaluation. If ``None``,
            no validation metrics are reported.
        monitor_cost: bool, optional
            If ``True`` and ``validation_data`` is provided, also logs the
            total cost on the validation set each epoch. Defaults to
            ``False``.
        """

        training_data = list(training_data)
        n = len(training_data)
        if validation_data is not None:
            validation_data = list(validation_data)
            n_val = len(validation_data)

        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            # Log message for this epoch
            log = f"Epoch {j+1}/{epochs}"

            if validation_data is not None:
                if isinstance(self.cost, MSE):
                    # If doing regression, report MSE
                    mse = self.evaluate_regression(validation_data)
                    log += f" — Validation MSE: {mse:.4f}"
                else:
                    # If doing classiciation, report accuracy
                    accuracy = self.evaluate_classification(validation_data)
                    log += f" — Validation accuracy: {accuracy}/{n_val}"

            if monitor_cost and validation_data is not None:
                cost = self.total_cost(validation_data)
                log += f" — Validation cost: {cost:.4f}"

            print(log)

    # Define a method containing our update rules for SGD
    def update_mini_batch(self, mini_batch, eta):
        """
        Apply a single gradient descent update using one mini-batch.

        Stacks the mini-batch samples into matrices, computes gradients
        via backpropagation, and updates all weights and biases in-place
        using the averaged gradients scaled by the learning rate.

        Parameters
        ----------
        mini_batch: list of (x, y) tuples
            A subset of training samples for this update step. Each ``x``
            is an input column vector and each ``y`` is the target.
        eta: float
            Learning rate that scales the gradient update.
        """  
        
        X = np.hstack([x for x, y in mini_batch])
        Y = np.hstack([y for x, y in mini_batch])

        # Gradients of the biases and weights
        nabla_b, nabla_w = self.backprop(X, Y)

        # Update the weights and biases
        self.weights = [w - eta * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - eta * nb
                       for b, nb in zip(self.biases, nabla_b)]


# ------------------------------------------------------------------
#   Backpropagation
# ------------------------------------------------------------------

    # Define a backpropagation method (backpropagates the entire mini-batch 
    # in a single forward and backward pass)
    def backprop(self, X, Y):
        """
        Compute gradients via backpropagation for a full mini-batch.

        Runs a single vectorized forward pass over the entire mini-batch,
        then propagates the error signal backward through the network to
        compute the gradient of the cost (C_x) with respect to every weight
        and bias. Gradients are averaged over all samples in the batch.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_inputs, n_samples)
            Input data matrix, one sample per column.
        Y: numpy.ndarray, shape (n_outputs, n_samples)
            Target matrix, one target per column.

        Returns
        -------
        nabla_b: list of numpy.ndarray
            Gradient of the cost with respect to each bias vector,
            in the same order as ``self.biases``.
        nabla_w: list of numpy.ndarray
            Gradient of the cost with respect to each weight matrix,
            in the same order as ``self.weights``.
        """

        # Numer of columns in X matrix
        m = X.shape[1]

        # Forward pass
        activations, z_vals = self._forward_pass(X)


        # Backward pass
        # Select the final pre-activation and output activation values
        delta = self.cost.delta(z_vals[-1], activations[-1], Y)

        nabla_b = [None] * (self.num_layers - 1)
        nabla_w = [None] * (self.num_layers - 1)

        nabla_b[-1] = delta.sum(axis = 1, keepdims = True) / m
        nabla_w[-1] = np.dot(delta, activations[-2].T) / m

        for l in range(2, self.num_layers):
            dsig = self.d_sigmoid(z_vals[-l])

            # Take the delta from and weights connecting to the next
            # layer, propagate it backward, then multiply by the 
            # sigmoid derivative at the current layer

            delta = np.dot(self.weights[-l + 1].T, delta) * dsig

            # Select activations from the previous layer
            nabla_b[-l] = delta.sum(axis = 1, keepdims = True) / m
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T) / m
        
        return (nabla_b, nabla_w)


# ------------------------------------------------------------------
#   Evaluation
# ------------------------------------------------------------------

    #Define a method to evaluate the neural network outputs
    def evaluate_classification(self, test_data):
        """
        Count the number of correctly classified samples.

        For binary classification, a prediction is considered positive when
        the output activation exceeds 0.5. For multiclass classification,
        the predicted class is the index of the maximum output activation,
        compared against the index of the maximum in the one-hot target.

        Parameters
        ----------
        test_data: iterable of (x, y) tuples
            Test samples. For binary tasks ``y`` should be a scalar label;
            for multiclass tasks ``y`` should be a one-hot encoded vector.

        Returns
        -------
        int
            Total number of correctly classified samples.
        """
        
        if self.task == "binary":
            test_results = [
                (int(self.feedforward(x).item() > 0.5), int(np.asarray(y).item()))
                for (x, y) in test_data
            ]
        else:
            test_results = [
                (np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in test_data
            ]

        return sum(int(pred == label) for (pred, label) in test_results)

    def evaluate_regression(self, validation_data):
        """
        Compute the mean squared error on a regression validation set.

        Runs a forward pass on each sample and averages the per-sample
        MSE over the entire validation set.

        Parameters
        ----------
        validation_data: iterable of (x, y) tuples
            Validation samples, where ``x`` is an input vector and ``y``
            is the continuous target.

        Returns
        -------
        float
            Average mean squared error across all validation samples.
        """ 

        total = sum(np.mean((self.feedforward(x) - y)**2)
                    for x, y in validation_data)
        return total / len(validation_data)

    def total_cost(self, validation_data):
        """
        Compute the average cost over a validation set.

        Runs a forward pass on each sample and evaluates the cost function,
        then returns the mean cost across the entire set. Works with any
        cost function that implements the ``cost(a, y)`` interface.

        Parameters
        ----------
        validation_data : iterable of (x, y) tuples
            Validation samples used to compute the total cost.

        Returns
        -------
        float
            Mean cost over all samples in ``validation_data``.
        """   
         
        total = 0.0
        for x, y in validation_data:
            a = self.feedforward(x)
            total += self.cost.cost(a, y)
        return total / len(validation_data)
