# Neural Networks 

This notebook introduces a comprehensive feedforward neural network capable of handling regression and classification tasks. A neural network consists of multiple layers of "neurons", which are essentially computational units that take input values, apply weights to scale these values, and pass them through an activation function to return an output. Each output is then passed onto neurons in the next layer. The network in general is composed of three parts: 

* The input layer takes in the feature data 
* The hidden layer(s) transform the data according to the activation function 
* The output layer yields the final predictions  


## Overview of the Algorithm 

**1. Initialize Weights and Biases**

The first step in the neural network algorithm is to initialize the weights and biases. In the NeuralNetwork class, the weights and biases are initialized with values from a standard Gaussian distribution, with the weights having further Xavier scaling. The weights connect neurons in one layer to neurons in the next, and they represent the relative importance of each output from the previous layer in activating neurons in the next layer. The biases, on the other hand, belong to each neuron in the hidden and output layers and help the network learn better by shifting each neuron's output. 

 

**2. Feedforward Phase**

The feedforward phase pertains to the movement of input data through the hidden layers and to the output layer. In this phase, the dot product between the input values and weights is computed, and a bias term is added. This value is called the pre-activation value (*z*): 

$$ 
z^l = w^l a^{l-1} + b^l
$$

The pre-activation value is then passed into an activation function to produce an output from each neuron:

$$
a^l = \sigma(z^l)
$$

In the NeuralNetwork class, the neurons in the hidden layers apply the sigmoid activation function, which is defined as: 

$$
\sigma(z) = \frac{1}{1 + e^{-z}} 
$$

With sigmoid neurons, small changes in the weights and bias produce small changes in the neuron's output, which ultimately helps the network learn. The activation function in the output layer differs based on the task at hand. For regression, a linear activation function is used so that values are not compressed to any scale. For binary classification, a sigmoid activation is used because it outputs values between 0 and 1, which can be thought of as the probability the input belongs to the positive class. For multiple classification, a softmax activation is used because it outputs a probability distribution with output values corresponding to each class. 

 

**3. Backpropagation with Stochastic Gradient Descent**

The goal of the algorithm is to find the best weights and biases to map the inputs to outputs. This is achieved by minimizing some cost function with respect to the weights and biases. For regression tasks, the NeuralNetwork class uses the mean-squared error cost:

$$
C = \frac{1}{2n} \sum_x \| y(x) - a^L(x) \|^2
$$

while for classification tasks, the cross-entropy cost is used. Mean-squared error is strong for regression because it measures the average squared distance between the network's predictions and the target values, both of which are continuous values. Cross-entropy is strong for classification because it assumes outputs are probabilities and heavily penalizes wrong predictions, which helps improve network learning. 

The network improves itself using a method called stochastic gradient descent. This method minimizes the gradient of the cost function (*C*), which refers to the vector of partial derivatives of the cost with respect to the weights and biases. Under stochastic gradient descent, we first estimate the gradient $\nabla C$ by computing $\nabla C_x$ for each training input in a random sample. Each group of training inputs is referred to as a mini-batch. Once we estimate the gradients, we multiply by a small learning rate ($\eta$), and subtract this value from the current values of the weights and biases. This essentially means we are taking small steps in the direction that does the most to immediately decrease the cost function, with the ultimate goal of finding the weights and biases that minimize the cost function. After performing gradient descent over one mini-batch, we randomly sample another group of training examples, then continue until all the training inputs are used. This completes one epoch of training. 

The driver behind stochastic gradient descent is the backpropagation method, which computes the gradients that are used to update the weights and biases.  Backpropagation works as follows: first, data are input into the network and fedforward through the hidden layers and to the output layer. At the output layer, the output error $\delta^L$ is computed, which is the difference between the predicted value and the actual target value:

$$
\delta^L = \nabla_a C \odot \sigma'(z^L)
$$

We then calculate the error from the previous layer in terms of the error in the next layer:

$$
\delta^l = \left( (w^{l+1})^T \delta^{l+1} \right) \odot \sigma'(z^l)
$$

We then move backwards through the network using this equation, computing the errors at each layer in terms of the errors in the next layer. The errors at each layer are used to compute the partial derivatives of the cost with respect to the biases and weights at each layer:

$$
\frac{\partial C}{\partial b^l_j} = \delta^l_j
$$

$$
\frac{\partial C}{\partial w^l_{jk}} = a^{l-1}_k \, \delta^l_j
$$

This ultimately allows the network to associate errors in one layer back to the neurons in the previous layer and then update the weights and biases so that the errors are minimized in the next round of training:

$$
w^l \leftarrow w^l - \frac{\eta}{m} \sum_x \delta^{x,l} (a^{x,l-1})^T
$$

$$
b^l \leftarrow b^l - \frac{\eta}{m} \sum_x \delta^{x,l}
$$