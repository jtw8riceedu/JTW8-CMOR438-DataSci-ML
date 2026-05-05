# Random Forests

This notebook introduces the `RandomForestClassifier` and `RandomForestRegressor` classes, which are ensemble methods that train a large number of decision trees and aggregate their predictions. Each tree in the forest is trained on a **different random sample of the data** (bootstrap sample) and uses only a **random subset of features at each split**. This combination of randomness aims to combat any overfitting within any individual decision tree.

A more detailed overview of random forests can be found below.

## Overview of the Algorithm

### 1. Bootstrap Sampling

For each of the $T$ trees in the forest, a **bootstrap sample** is drawn from the training data. A bootstrap sample is formed by drawing $n$ samples from the $n$ training points uniformly at random *with replacement*, meaning some training points may appear multiple times and others not at all. Each tree's training set is therefore slightly different, which promotes diversity across the ensemble.

### 2. Random Feature Subsetting

Each tree in the `RandomForestClassifier` and `RandomForestRegressor` overrides the standard decision tree's split search with a version that only considers a randomly chosen subset of $m$ features at each node, rather than all $p$ features. This distinguishes random forests from a simple bagged ensemble of trees.

For classification, the default number of features considered at each split is:

$$
m = \lfloor \sqrt{p} \rfloor
$$

For regression, the default is:

$$
m = \lfloor p / 3 \rfloor
$$

At each node, $m$ features are sampled without replacement and only those features are considered as candidates for the best split. This means different trees tend to specialize in different subsets of features, reducing the correlation between trees and lowering the overall ensemble variance.


### 3. Grow Each Tree

Each tree is grown to its full depth (subject to `max_depth` and `min_samples_split` constraints) on its bootstrap sample, using the random feature subsetting described above at every internal node. The split criterion, impurity measures, and leaf prediction rules are identical to those of the underlying `DecisionTreeClassifier` or `DecisionTreeRegressor` — see the Decision Tree README for details.


### 4. Aggregate Predictions

Once all $T$ trees are trained, predictions are made by aggregating their individual outputs.

**Classification (majority vote):** Each tree predicts a class label $\hat{y}^{(t)}$ for the query input. The ensemble predicts the class that receives the most votes:

$$
\hat{y} = \underset{k}{\arg\max} \sum_{t=1}^{T} \mathbf{1}[\hat{y}^{(t)} = k]
$$

**Regression (averaging):** Each tree predicts a continuous value $\hat{y}^{(t)}$. The ensemble prediction is the mean across all trees:

$$
\hat{y} = \frac{1}{T} \sum_{t=1}^{T} \hat{y}^{(t)}
$$

Averaging reduces the variance of the prediction relative to any single tree, which is why random forests typically outperform individual decision trees on unseen data. Increasing $T$ generally improves stability but with diminishing returns; the forest does not overfit as $T$ grows, unlike individual trees that can overfit by growing too deep.