# Decision Trees

This notebook introduces the `DecisionTreeClassifier` and `DecisionTreeRegressor` classes, which build binary tree structures that recursively partition the feature space to make predictions. At each internal node, the tree compares a single feature to a certain threshold, and routes each data point accordingly. Leaf nodes store the final prediction. A brief overview of how decision trees work can be found below. Another great resource that covers concepts including impurity, information gain, and recursion is this [Towards Data Science article](https://towardsdatascience.com/decision-trees-explained-entropy-information-gain-gini-index-ccp-pruning-4d78070db36c/). 


## Overview of the Algorithm

### 1. Choose an Impurity Measure

An **impurity measure** quantifies how mixed the class labels (or target values) are at a given node. For classification, the `DecisionTreeClassifier` supports two impurity measures. **Entropy** measures the average information content of the class distribution at a node:

$$
H(y) = -\sum_{k} p_k \log_2 p_k
$$

The **Gini index** measures the probability of a randomly chosen sample being misclassified if its label were assigned randomly according to the class distribution:

$$
G(y) = 1 - \sum_{k} p_k^2
$$

In both formulas, $p_k$ is the proportion of samples at the node belonging to class $k$. A pure node (all samples from one class) has impurity 0; a node that is equally distributed between classes has the highest impurity (1 for entropy, 0.5 for Gini).

For the `DecisionTreeRegressor`, the impurity measure is **mean squared error** within the node, which is minimized when the leaf prediction (the mean of all target values at the node) fits the targets well:

$$
\text{MSE}(y) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2
$$


### 2. Find the Best Split

At each node, the algorithm searches over all features and all unique values of each feature as candidate thresholds. For a candidate split that partitions samples into a left group $y_L$ and a right group $y_R$, the **weighted child impurity** is:

$$
\text{Impurity}(y_L, y_R) = \frac{|y_L|}{|y_L| + |y_R|} \cdot \text{Impurity}(y_L) + \frac{|y_R|}{|y_L| + |y_R|} \cdot \text{Impurity}(y_R)
$$

The split that minimizes this weighted impurity is chosen. Samples with feature value $\leq$ threshold go left; all others go right.


**3. Recursively Build the Tree**

Starting at the root and working downward, the algorithm recursively applies the best-split logic at each node. A node is converted to a leaf — stopping further growth — when any of the following conditions are met: the maximum tree depth `max_depth` is reached, the node contains fewer than `min_samples_split` samples, the node is already pure (impurity = 0), or no valid split can be found that separates samples into two non-empty partitions.

Leaf predictions are determined as follows. For the classifier, each leaf stores the **majority class** among its training samples:

$$
\hat{y} = \underset{k}{\arg\max} \sum_{i} \mathbf{1}[y_i = k]
$$

For the regressor, each leaf stores the **mean of its training target values**:

$$
\hat{y} = \bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i
$$


### 4. Predict

At inference time, each input sample is routed from the root down through the tree by evaluating the binary condition at each internal node, until it reaches a leaf. The prediction stored at that leaf is returned.