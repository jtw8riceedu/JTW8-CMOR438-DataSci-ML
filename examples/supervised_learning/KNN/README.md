# K-Nearest Neighbors (KNN)

This notebook introduces the `KNN` class, an algorithm that makes predictions based on the labels of the most similar training examples. The `KNN` class supports both classification (returning the majority class) and regression (returning the mean target value).

A more detailed overview can be found below.


## Overview of the Algorithm

### 1. Store Training Data (Fit Phase)

With KNN, there is no parameter optimization at fit time. The `fit()` method simply stores the training feature matrix $X_{\text{train}}$ and target vector $y_{\text{train}}$ for use during prediction. It is important to note that `KNN` uses **Euclidean distance** and is therefore sensitive to feature scale — features should be standardized before fitting when columns use different units or ranges. It is recommended to use `StandardScaler` before proceeding with `KNN`.


### 2. Compute Distances

At prediction time, the algorithm computes the **Euclidean distance** between a query point $x$ and every point in the training set. For two points $p$ and $q$ with $n$ features, the Euclidean distance is:

$$
d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}
$$

This is equivalent to $\sqrt{(p - q) \cdot (p - q)}$, which is how it is computed in the `KNN` class.


### 3. Identify the $k$ Nearest Neighbors

Once all pairwise distances have been computed, the training indices are sorted by distance in ascending order and the **first $k$ samples are selected**. The value of $k$ controls the **bias-variance tradeoff**: small values of $k$ produce low-bias, high-variance models that closely follow the training data, while large values of $k$ produce smoother, lower-variance predictions at the cost of potentially higher bias.


### 4. Aggregate Neighbor Labels

**Classification:** The predicted class is the majority label among the $k$ nearest neighbors. In the case of a tie, the class with the lowest index among tied classes is returned by NumPy's `argmax`.

**Regression:** The predicted value is the mean of the target values of the $k$ nearest neighbors.