# Voting Classifiers and Regressors

This notebook introduces voting ensembles, a family of ensemble learning methods that combine the predictions of multiple base estimators to produce a single, more robust prediction. The core idea is that a group of diverse models, each with different strengths and weaknesses, will collectively outperform any individual model by reducing variance and correcting for individual errors. The HardVotingClassifier and VotingRegressor classes in this package support models with prediction methods (i.e. supervised learning), such as KNN and linear regression. Note that the voting models do not support instances of the NeuralNetwork class as there is no clear predict method defined. 

A more detailed overview of how voting classifiers and voting regressors work can be found below.


## Overview of the Algorithm

### 1. Fit Base Estimators

The first step is to train each base estimator independently on the full training dataset. Because the base estimators are trained independently, their errors are at least partially uncorrelated, which enables the ensemble models to have higher accuracies than any single model. Let $\hat{f}_1, \hat{f}_2, \ldots, \hat{f}_M$ denote the $M$ fitted base estimators.


### 2. Collect Predictions

At inference time, each base estimator produces a prediction for the input $x$. For classification, each algorithm in this package outputs a predicted class label. For regression, each algorithm outputs a continuous predicted value. Let $\hat{y}_m$ denote the prediction of the $m$-th estimator for input $x$:

$$
\hat{y}_m = \hat{f}_m(x), \quad m = 1, 2, \ldots, M
$$


### 3. Aggregate Predictions

The final prediction is produced by aggregating the individual predictions.

**HardVotingClassifier**

Under hard voting, each base classifier casts a vote for a single class label, and the class with the most votes is returned as the final prediction. For a classification problem with classes $\{1, 2, \ldots, K\}$, the predicted class $\hat{y}$ is:

$$
\hat{y} = \underset{k \in \{1, \ldots, K\}}{\arg\max} \sum_{m=1}^{M} \mathbf{1}[\hat{y}_m = k]
$$

where $\mathbf{1}[\hat{y}_m = k]$ is an indicator that equals 1 if the $m$-th estimator predicted class $k$. 


**VotingRegressor**

For regression, the VotingRegressor returns the average of the predicted values from the base estimators:

$$
\hat{y} = \frac{1}{M} \sum_{m=1}^{M} \hat{y}_m
$$

Averaging reduces the variance of the final prediction relative to any individual estimator, which typically improves performance on unseen data — particularly when the base estimators tend to overfit in different ways.


### 4. Weighted Voting

The VotingClassifier and VotingRegressor classes also support assigning a weight $w_m$ to each base estimator, allowing the user to specify which models should have more or less importance in the aggregated predictions. Weights are typically assigned according to which models are more accurate or trusted by the user.