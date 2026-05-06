# Source Code

This directory contains the reusable Python source code for the `ml_package` package. The code is organized by machine learning task type so that algorithms, utilities, and tests are easy to connect to one another.

```text
src/
+-- ml_package/
|   +-- supervised_learning/
|   +-- unsupervised_learning/
|   +-- utils/
|   +-- __init__.py
+-- ml_package.egg-info/
```

The `ml_package.egg-info/` directory is generated packaging metadata. The main code to read and edit is in `ml_package/`.

## Package Entry Point

`ml_package/__init__.py` exposes the main classes and helper modules so they can be imported directly from the package:

```python
from ml_package import KNN, RandomForestClassifier, PCA, StandardScaler
```

This keeps the notebooks concise while still preserving the internal folder structure.

## Supervised Learning

`ml_package/supervised_learning/` contains algorithms that learn from labeled data.

- `linear_regression.py`: single-output linear regression trained with gradient descent, with regression metric helpers.
- `logistic_regression.py`: binary and multinomial logistic regression using sigmoid or softmax outputs.
- `perceptron.py`: perceptron-style model with configurable activation functions.
- `neural_network.py`: feedforward neural network for regression, binary classification, and multiclass classification.
- `knn.py`: K-Nearest Neighbors for classification and regression.
- `decision_tree.py`: decision tree classifier and regressor implementations.
- `random_forest.py`: random forest classifier and regressor built from decision tree estimators.
- `voting.py`: hard voting classifier and voting regressor for combining multiple fitted models.

## Unsupervised Learning

`ml_package/unsupervised_learning/` contains algorithms that discover structure without target labels.

- `pca.py`: Principal Component Analysis for dimensionality reduction.
- `kmeans.py`: K-Means clustering with centroid updates and inertia tracking.
- `dbscan.py`: density-based clustering with noise-point detection.

## Utilities

`ml_package/utils/` contains shared functionality used by algorithms, notebooks, and tests.

- `preprocessing.py`: train/test splitting, train/validation/test splitting, K-fold splitting, randomized search, `StandardScaler`, and `MinMaxScaler`.
- `classification_metrics.py`: confusion matrix, accuracy, precision, recall, and F1 score.
- `regression_metrics.py`: MSE, RMSE, MAE, MAPE, SMAPE, MASE, R-squared, adjusted R-squared, AIC, AICc, and BIC.

## Related Tests

The `tests/` directory mirrors the package organization:

- `tests/supervised_learning/`
- `tests/unsupervised_learning/`
- `tests/utils/`

Run the full test suite from the repository root with:

```bash
pytest
```
