""" pca.py """

import numpy as np

class PCA(object):
    """
    Principal Component Analysis (PCA).

    Reduces dimensionality by projecting data onto the directions of
    maximum variance. The data is centered before computing the covariance
    matrix, and principal components are derived from the eigenvectors of
    that matrix.

    PCA is sensitive to feature scale. If features use different units or
    ranges, scale the data before fitting (for example with
    ``StandardScaler`` from ``ml_package.utils``).

    Parameters
    ----------
    n_components: int
        Number of principal components to retain.

    Attributes
    ----------
    n_components_: int
        Number of components to retain, as specified at construction.
    components_: numpy.ndarray, shape (n_features, n_components)
        Top eigenvectors of the covariance matrix, forming the projection
        matrix. Set after ``fit()``.
    mean_: numpy.ndarray, shape (n_features,)
        Column-wise mean of the training data, used for centering. Set
        after ``fit()``.
    explained_variance_: numpy.ndarray, shape (n_components,)
        Eigenvalue associated with each retained principal component,
        representing the variance explained along that direction. Set
        after ``fit()``.
    explained_variance_ratio_: numpy.ndarray, shape (n_components,)
        Fraction of total variance explained by each retained component.
        Set after ``fit()``.

    Methods
    -------
    fit(X)
        Compute principal components from training data.
    transform(X)
        Project data onto the fitted principal components.
    fit_transform(X)
        Fit the model and project the training data.
    """

    def __init__(self, n_components):
        """
        Initialize the PCA model.

        Parameters
        ----------
        n_components: int
            Number of principal components to retain.
        """    
        
        self.n_components_ = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None


    # Fit method
    def fit(self, X):
        """
        Compute principal components from training data.

        Centers the data column-wise, computes the unbiased sample
        covariance matrix, and derives principal components via
        eigendecomposition. The top ``n_components`` eigenvectors
        (sorted by descending eigenvalue) are stored as the projection
        matrix. Sets ``components_``, ``mean_``, ``explained_variance_``,
        and ``explained_variance_ratio_``.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data to fit.
        """    
                
        X = np.array(X)
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # SVD is numerically stable and handles p >> n correctly
        # economy/thin SVD: U is (n_samples × k), S is (k,), Vt is (k × n_features)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Eigenvalues of the covariance matrix = S² / (n_samples - 1)
        eigen_values = (S ** 2) / (X.shape[0] - 1)

        # Eigenvectors are the rows of Vt (right singular vectors)
        self.components_ = Vt[:self.n_components_].T  # shape (n_features, n_components)

        self.explained_variance_ = eigen_values[:self.n_components_]
        total_variance = np.sum(eigen_values)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance


    # Function to project the data on the principal components
    def transform(self, X):
        """
        Project data onto the fitted principal components.

        Centers ``X`` using the mean computed during ``fit()``, then
        projects onto the stored components.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Data to project. Must have the same number of features as
            the training data.

        Returns
        -------
        numpy.ndarray, shape (n_samples, n_components)
            Data projected onto the top ``n_components`` principal components.
        """
    
        X = np.array(X)
        X_centered = X - self.mean_
        return X_centered @ self.components_

    def fit_transform(self, X):
        """
        Fit the model and project the training data.

        Equivalent to calling ``fit(X)`` followed by ``transform(X)``.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        numpy.ndarray, shape (n_samples, n_components)
            Training data projected onto the fitted principal components.
        """   
        
        self.fit(X)
        return self.transform(X)
