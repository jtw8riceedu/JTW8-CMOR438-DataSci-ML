# Linear Regression

This notebook introduces a single-neuron-based approach to linear regression trained with full-batch gradient descent. The model fits a linear relationship between input features and a continuous target variable by minimizing the mean squared error between predictions and observed values. Since the LinearRegression class takes a single-neuron approach, the class was derived from the more general NeuralNetwork class. In fact, linear regression can be performed by instantiating NeuralNetwork([p, 1], task = "regression"), which signals that we are giving each feature *p* an input neuron and creating a network with only one layer (the output layer). 

 A more detailed overview of the linear regression algorithm can be found below.


## Overview of the Algorithm

### 1. Initialize Weights and Bias

The first step is to initialize the model's parameters. In the LinearRegression class, the weights and bias are both sampled from a standard normal distribution. These initial values serve as the starting point for gradient descent.


### 2. Feedforward Phase

At each training step, the model computes a predicted value $\hat{y}$ for each input sample by taking the dot product of the inputs and weights and adding the bias:

$$
\hat{y} = \mathbf{X} \mathbf{w} + b
$$

Since this is a regression task, a **linear activation function** is used, meaning the pre-activation value is returned unchanged. This allows the model to predict values across the full real number line rather than compressing them to a fixed range.


### 3. Compute the Cost

The model's performance is measured using the **mean squared error (MSE)** cost function, which computes the average squared difference between the predictions $\hat{y}$ and the true target values $y$:

$$
C = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2
$$

MSE is well-suited for regression because it penalizes larger errors more heavily and has a smooth gradient that makes it easy to optimize.


### 4. Backpropagation and Parameter Updates

The model improves by computing the gradient of the cost with respect to the weights and bias, then updating the parameters in the direction that reduces the cost. The gradient of the MSE cost with respect to the predictions is:

$$
\delta = \hat{y} - y
$$

Here, we can exclude the fraction out front as it gets absorbed into the learning rate during gradient descent. The gradients with respect to the weights and bias are then:

$$
\nabla_{\mathbf{w}} C = \frac{1}{n} \mathbf{X}^T \delta, \qquad \nabla_b C = \frac{1}{n} \sum_{i=1}^{n} \delta_i
$$

The weights and bias are updated by stepping in the negative gradient direction, scaled by a learning rate $\eta$:

$$
\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_{\mathbf{w}} C
$$

$$
b \leftarrow b - \eta \nabla_b C
$$

This process is repeated for a specified number of epochs. Unlike stochastic or mini-batch gradient descent, this implementation uses **full-batch gradient descent**, meaning the gradients are computed over the entire training set at each step. The MSE loss at the end of each epoch is recorded in `self.losses` for monitoring convergence.