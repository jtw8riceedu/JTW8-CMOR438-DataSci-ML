# Logistic Regression

This notebook introduces the LogisticRegression class, a single-neuron-based approach to classification trained with full-batch gradient descent. The model supports both binary classification (two classes) and multinomial classification (three or more classes) by applying different output activation functions and cost functions depending on the task. Since the LogisticRegression class takes a single-neuron approach, the class was derived from the more general NeuralNetwork class. In fact, logistic regression can be performed by the NeuralNetwork class. Binary logistic regression can be performed by instantiating NeuralNetwork([p, 1], task = "binary"), which signals that we are giving each feature *p* an input neuron and creating a network with only one neuron in the output layer. Multinomial regression can be performed by instantiating NeuralNetwork([p, k], task = "multiclass"), which signals that there are *k* neurons in the output layer, each pertaining to a class from the data.

A more detailed overview of both binary and multinomial logistic regression can be found below.

## Overview of the Algorithm

### 1. Initialize Weights and Bias

The LogisticRegression class initializes parameters differently depending on the task. For binary classification, the weights and bias are sampled from a standard normal distribution. For multinomial classification with $K$ classes, the weight matrix $W \in \mathbb{R}^{p \times K}$ is Xavier-initialized to help stabilize training:

$$
W \sim \frac{\mathcal{N}(0, I)}{\sqrt{p}}, \quad \mathbf{b} \in \mathbb{R}^K \sim \mathcal{N}(0, I)
$$


### 2. Feedforward Phase

For each input, the model first computes a pre-activation value by first multiplying inputs and weights, then adding the bias. For binary classification, the pre-activation value is passed through the sigmoid activation fundtion:

$$
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \mathbf{X}\mathbf{w} + b
$$

For multinomial classification, the pre-activation value is passed through the softmax activation function.

$$
\hat{p}_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, \quad z = \mathbf{X} W + \mathbf{b}
$$

where $\hat{p}_k$ is the predicted probability of class $k$ for a given input.

The sigmoid activation is used for binary classification because it outputs values between 0 and 1, which can be thought of as the probability the input belongs to the positive class. For multiple classification, a softmax activation is used because it outputs a probability distribution with output values corresponding to each class. 

### 3. Compute the Cost

For binary classification, the binary cross-entropy cost is used:

$$
C = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

For multinomial classification, categorical cross-entropy is used:

$$
C = -\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} y_{ik} \log \hat{p}_{ik}
$$

where $y_{ik}$ is the one-hot encoded label for sample $i$ and class $k$. Cross-entropy is preferred for classification because it heavily penalizes confident wrong predictions and naturally pairs with sigmoid and softmax outputs.


**4. Backpropagation and Parameter Updates**

A key property of pairing cross-entropy loss with sigmoid or softmax activations is that the gradient of the cost with respect to the pre-activation value simplifies cleanly to:

$$
\delta = \hat{y} - y
$$

This cancels the activation derivative and produces a clean, efficient update rule. The weight and bias gradients are then:

$$
\nabla_W C = \frac{1}{n} \mathbf{X}^T \delta, \qquad \nabla_{\mathbf{b}} C = \frac{1}{n} \sum_{i=1}^{n} \delta_i
$$

Parameters are updated by stepping in the negative gradient direction:

$$
W \leftarrow W - \eta \nabla_W C, \qquad \mathbf{b} \leftarrow \mathbf{b} - \eta \nabla_{\mathbf{b}} C
$$

This is repeated for a specified number of epochs, with the cross-entropy loss recorded each epoch in `self.losses`. Predictions are made by thresholding the sigmoid output at 0.5 for binary tasks, or by taking the $\arg\max$ of the softmax output for multinomial tasks.