"""Core package for custom ML implementations."""

# Supervised learning
from .supervised_learning.knn import KNN
from .supervised_learning.linear_regression import LinearRegression
from .supervised_learning.logistic_regression import LogisticRegression
from .supervised_learning.perceptron import Perceptron
from .supervised_learning.neural_network import NeuralNetwork
from .supervised_learning.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor
from .supervised_learning.random_forest import RandomForestClassifier, RandomForestRegressor
from .supervised_learning.voting import HardVotingClassifier, VotingRegressor

# Unsupervised learning
from .unsupervised_learning.pca import PCA
from .unsupervised_learning.kmeans import KMeans
from .unsupervised_learning.dbscan import DBSCAN

# Utilities
from .utils.preprocessing import StandardScaler, MinMaxScaler, train_test_split, train_val_test_split, k_fold_split, randomized_search_cv
from .utils import classification_metrics, regression_metrics, preprocessing

__all__ = [
    # Supervised
    "KNN",
    "LinearRegression",
    "LogisticRegression",
    "Perceptron",
    "NeuralNetwork",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "HardVotingClassifier",
    "VotingRegressor",
    # Unsupervised
    "PCA",
    "KMeans",
    "DBSCAN",
    # Utils
    "StandardScaler",
    "MinMaxScaler",
    "train_test_split",
    "train_val_test_split",
    "k_fold_split",
    "randomized_search_cv",
    "classification_metrics",
    "regression_metrics",
    "preprocessing",
]