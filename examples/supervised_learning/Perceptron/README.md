# Perceptron

This notebook introduces the `Perceptron` class, a single-layer neural network unit trained with full-batch gradient descent. The perceptron model works by taking in a weighted combination of inputs and passing it through a nonlinear activation function to produce a binary output. It is designed for binary classification tasks where labels are $+1$ and $-1$. 

A more detailed overview of the algorithm can be found below. Furthermore, a more general explanation of forward pass, gradient descent, and backpropagation, which are used in `Perceptron`, can be found in the Neural Networks README.

## Overview of the Algorithm

### 1. Initialize Weights and Bias

The `Perceptron` class requires an activation function to be specified at initialization. The different activation functions supported include ReLU, Leaky ReLU, tanh, and unit-step. The weights and bias are initialized from a standard normal distribution at train time.


### 2. Feedforward Phase

For each input, the model first computes a **pre-activation value**, which multiplies the weights by the input values and adds the bias. Then, the model applies the chosen **activation function**:

$$
z = \mathbf{X} \mathbf{w} + b, \qquad \hat{y} = f(z)
$$

The activation functions and their derivatives are defined as follows:

* **ReLU** returns positive values unchanged and negative values as zeros: $f(z) = \max(0, z)$, with derivative $f'(z) = \mathbf{1}[z \geq 0]$. 

* **Leaky ReLU** is similar but allows a small negative slope: $f(z) = z$ if $z \geq 0$, else $0.01z$, with derivative $f'(z) = 1$ if $z \geq 0$, else $0.01$. Leaky ReLU ensures a non-zero gradient that will allow the network to continue learning. This helps combat the "dying neuron" problem that the ReLU function can create, in which neurons stop learning if they only receive negative values (due to the outputs and gradients becoming zero). More information on ReLU versus Leaky ReLU can be found on this [Medium article](https://medium.com/@sreeku.ralla/activation-functions-relu-vs-leaky-relu-b8272dc0b1be).

* The **tanh** activation maps values to $(-1, 1)$: $f(z) = \tanh(z)$, with derivative $f'(z) = 1 - \tanh^2(z)$. 

* The **unit-step** function outputs $+1$ for non-negative values and $-1$ otherwise, with its derivative approximated as 1 everywhere.

The unit-step function is the base activation function that the perceptron model was built off. However, the step function is not differentiable, so it cannot be trained using gradient descent. Thus, as aforementioned, the `Perceptron` clas estimates the unit-step derivative as 1 to allow gradient-based training. Over time, the perceptron model has evolved to support differentiable non-linear functions like the ReLU and tanh, which is why the class includes them.

### 3. Compute the Cost

The `Perceptron` class uses **mean squared error** as its cost function:

$$
C = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2
$$

The gradient of the MSE cost with respect to the predictions is:

$$
\frac{\partial C}{\partial \hat{y}} = \hat{y} - y
$$


### 4. Backpropagation and Parameter Updates

Because the perceptron has a single nonlinear activation, the error signal must be propagated back through the activation derivative. The combined delta at the output is:

$$
\delta = (\hat{y} - y) \odot f'(z)
$$

where $\odot$ is the elementwise product. The gradients with respect to the weights and bias are then:

$$
\nabla_{\mathbf{w}} C = \frac{1}{n} \mathbf{X}^T \delta, \qquad \nabla_b C = \frac{1}{n} \sum_{i=1}^n \delta_i
$$

The parameters are updated by stepping in the negative gradient direction:

$$
\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_{\mathbf{w}} C, \qquad b \leftarrow b - \eta \nabla_b C
$$

This process repeats for a specified number of epochs, with the MSE loss recorded each epoch in `self.losses`. At prediction time, the raw activation output is used directly for unit-step (which already maps to $\pm 1$), while all other activations threshold the output at zero, mapping non-negative values to $+1$ and negative values to $-1$.