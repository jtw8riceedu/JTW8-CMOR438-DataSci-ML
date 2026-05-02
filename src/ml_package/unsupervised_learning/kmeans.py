""" kmeans.py """

import numpy as np

class KMeans:
    """
    K-Means clustering algorithm.

    Partitions data into ``n_clusters`` groups by iteratively assigning
    each sample to its nearest centroid and recomputing centroids as the
    mean of their assigned samples. Convergence is detected when the total
    centroid shift falls below ``tol``. Empty clusters are re-initialized
    to a randomly chosen data point.

    K-Means uses Euclidean distance and is sensitive to feature scale.
    Scale features before fitting when columns use different units or
    ranges (for example with ``StandardScaler`` from ``ml_package.utils``).

    Parameters
    ----------
    n_clusters: int, optional
        Number of clusters to form. Defaults to 3.
    max_iter: int, optional
        Maximum number of assignment/update iterations. Defaults to 100.
    tol: float, optional
        Convergence threshold for the total centroid shift (L2 norm).
        Defaults to 1e-4.
    random_state: int or None, optional
        Seed for NumPy's random number generator, used during centroid
        initialization. Pass an integer for reproducible results.
        Defaults to ``None``.

    Attributes
    ----------
    centroids_: numpy.ndarray, shape (n_clusters, n_features)
        Final centroid positions after fitting.
    labels_: numpy.ndarray, shape (n_samples,)
        Cluster index assigned to each training sample.
    inertia_: float
        Sum of squared distances from each sample to its assigned centroid.

    Methods
    -------
    fit(X)
        Fit the model to data.
    predict(X)
        Assign new samples to the nearest fitted centroid.
    fit_predict(X)
        Fit the model and return cluster labels for the training data.
    """
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=None):
        """
        Initialize the KMeans model.

        Parameters
        ----------
        n_clusters: int, optional
            Number of clusters. Defaults to 3.
        max_iter: int, optional
            Maximum number of iterations. Defaults to 100.
        tol: float, optional
            Convergence tolerance for centroid shift. Defaults to 1e-4.
        random_state: int or None, optional
            Random seed for reproducibility. Defaults to ``None``.
        """
        
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None  # Sum of squared distances

    # Internal method to inialize the centroids
    def _initialize_centroids(self, X):
        """
        Determine initial centroids by sampling randomly from the data.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Input data from which centroids are drawn.

        Returns
        -------
        numpy.ndarray, shape (n_clusters, n_features)
            Randomly selected initial centroid positions.
        """
        
        if self.random_state is not None:
            np.random.seed(self.random_state)

        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]

    def _compute_distances(self, X, centroids):
        """
        Compute the Euclidean distance from each sample to each centroid.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Input data.
        centroids: numpy.ndarray, shape (n_clusters, n_features)
            Current centroid positions.

        Returns
        -------
        numpy.ndarray, shape (n_samples, n_clusters)
            Distance from each sample to each centroid.
        """
        
        # Shape: (n_samples, n_clusters)
        return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

    def _assign_clusters(self, distances):
        """
        Assign each sample to its nearest centroid.

        Parameters
        ----------
        distances: numpy.ndarray, shape (n_samples, n_clusters)
            Distances from each sample to each centroid.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Index of the nearest centroid for each sample.
        """    
        
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        """
        Recompute centroids as the mean of their assigned samples.

        If a cluster is empty (no samples assigned), its centroid is
        re-initialized to a randomly chosen data point to prevent
        degenerate solutions.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Input data.
        labels: numpy.ndarray, shape (n_samples,)
            Current cluster assignments.

        Returns
        -------
        numpy.ndarray, shape (n_clusters, n_features)
            Updated centroid positions.
        """      
        
        centroids = np.zeros((self.n_clusters, X.shape[1]))

        # Select points assigned to each cluster
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]

            # Handle empty clusters
            if len(cluster_points) == 0:
                centroids[k] = X[np.random.randint(0, X.shape[0])]
            else:
                centroids[k] = np.mean(cluster_points, axis=0)

        return centroids
    
    def _compute_inertia(self, X, centroids, labels):
        """
        Compute the total within-cluster sum of squared distances (inertia).

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Input data.
        centroids: numpy.ndarray, shape (n_clusters, n_features)
            Current centroid positions.
        labels: numpy.ndarray, shape (n_samples,)
            Cluster assignment for each sample.

        Returns
        -------
        float
            Sum of squared distances from each sample to its assigned centroid.
        """    
        # Initialize intertia as 0
        inertia = 0.0

        # Loop over each cluster k
        for k in range(self.n_clusters):
            
            # Select points assigned to each cluster
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
            
                # Compute sum of squared distances between points and their centroids
                inertia += np.sum((cluster_points - centroids[k]) ** 2)
        
        return inertia

    # Fit method
    def fit(self, X):
        """
        Fit the K-Means model to data.

        Initializes centroids, then iterates between assigning samples to
        the nearest centroid and updating centroids until convergence or
        ``max_iter`` is reached. Sets ``centroids_``, ``labels_``, and
        ``inertia_`` after fitting.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data.
        """    
        
        X = np.array(X)

        # Initialize centroids
        self.centroids_ = self._initialize_centroids(X)

        for _ in range(self.max_iter):
            # Assign clusters
            distances = self._compute_distances(X, self.centroids_)
            labels = self._assign_clusters(distances)

            # Update centroids
            new_centroids = self._update_centroids(X, labels)

            # Check convergence
            shift = np.linalg.norm(self.centroids_ - new_centroids)
            self.centroids_ = new_centroids

            if shift < self.tol:
                break

        self.labels_ = labels
        self.inertia_ = self._compute_inertia(X, self.centroids_, labels)

    # Predict method
    def predict(self, X):
        """
        Assign new samples to the nearest fitted centroid.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            New input data to cluster.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Index of the nearest centroid for each sample.
        """   
       
        X = np.array(X)
        distances = self._compute_distances(X, self.centroids_)
        return self._assign_clusters(distances)

    
    def fit_predict(self, X):
        """
        Fit the model and return cluster labels for the training data.

        Equivalent to calling ``fit(X)`` followed by returning
        ``self.labels_``.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Cluster label assigned to each training sample.
        """  
      
        self.fit(X)
        return self.labels_

