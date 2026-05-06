# Example Notebooks

This directory contains Jupyter notebooks that demonstrate the algorithms implemented in `src/ml_package/`. Each example folder includes a notebook, a small dataset, and an algorithm-specific `README.md` that explains the method in more detail.

The examples are grouped by learning type:

```text
examples/
+-- supervised_learning/
|   +-- Decision Trees/
|   +-- KNN/
|   +-- Linear Regression/
|   +-- Logistic Regression/
|   +-- Neural Networks/
|   +-- Perceptron/
|   +-- Random Forests/
|   +-- Voting Models/
+-- unsupervised_learning/
    +-- DBSCAN/
    +-- K-Means Clustering/
    +-- PCA/
```

## Running the Notebooks

From the repository root, install the dependencies and package:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Then open the notebooks in Jupyter or VS Code. The notebooks are designed to run against the local package code and the datasets stored next to each notebook.

## Supervised Learning

Supervised learning notebooks train models using input features and known target values.

| Folder | Notebook | Dataset | What it demonstrates |
| --- | --- | --- | --- |
| `Linear Regression/` | `linear_regression.ipynb` | `airfoil_self_noise.dat` | Fits a continuous target with the custom `LinearRegression` model and evaluates regression error. |
| `Logistic Regression/` | `logistic_regression.ipynb` | `Raisin_Dataset.csv` | Applies binary or multiclass classification with the custom `LogisticRegression` class. |
| `KNN/` | `knn.ipynb` | `Dry_Bean_Dataset.csv` | Uses nearest-neighbor predictions for classification or regression-style workflows. |
| `Decision Trees/` | `decision_tree.ipynb` | `Algerian_forest_fires_dataset_UPDATE.csv` | Builds interpretable tree-based models using split criteria such as entropy, Gini impurity, or regression error. |
| `Random Forests/` | `random_forest.ipynb` | `abalone.data` | Combines multiple decision trees with bootstrap sampling and random feature subsetting. |
| `Perceptron/` | `perceptron.ipynb` | `data_banknote_authentication.txt` | Trains a single-layer perceptron-style classifier with configurable activation behavior. |
| `Neural Networks/` | `neural_network.ipynb` | `Concrete_Data.csv` | Demonstrates a feedforward neural network for predictive modeling. |
| `Voting Models/` | `voting.ipynb` | `parkinsons_updrs.data` | Combines multiple base estimators with hard voting for classification or weighted averaging for regression. |

## Unsupervised Learning

Unsupervised learning notebooks explore structure in feature data without using target labels.

| Folder | Notebook | Dataset | What it demonstrates |
| --- | --- | --- | --- |
| `PCA/` | `pca.ipynb` | `toxicity_data.csv` | Reduces feature dimensionality by projecting data onto principal components. |
| `K-Means Clustering/` | `kmeans.ipynb` | `wholesale_customers.csv` | Finds compact groups in customer-style feature data using centroid-based clustering. |
| `DBSCAN/` | `dbscan.ipynb` | `tripadvisor_review.csv` | Finds density-based clusters and identifies noise points without specifying the number of clusters in advance. |

## Notebook Pattern

Most notebooks follow the same basic workflow:

1. Load a local dataset.
2. Clean or prepare the feature matrix and target values.
3. Split or scale the data when needed.
4. Train an algorithm from `ml_package`.
5. Evaluate predictions or visualize learned structure.

For algorithm details, start with the `README.md` inside each example folder.
