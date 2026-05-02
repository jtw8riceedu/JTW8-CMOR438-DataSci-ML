""" dbscan.py """

import numpy as np

class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

    Identifies clusters as dense regions of points separated by sparser
    areas, and marks points that do not belong to any dense region as
    noise. Unlike K-Means, the number of clusters does not need to be
    specified in advance, and clusters of arbitrary shape are supported.

    A point is a core point if at least ``min_samples`` other points lie
    within distance ``eps`` of it. Core points and all points reachable
    from them form a cluster. Points not reachable from any core point
    are labeled as noise (-1).

    DBSCAN uses Euclidean distance, so ``eps`` is interpreted in the
    feature space provided to ``fit``. Scale features before fitting when
    columns use different units or ranges (for example with
    ``StandardScaler`` from ``ml_package.utils``).

    Parameters
    ----------
    eps: float, optional
        Maximum distance between two points for one to be considered in
        the neighborhood of the other. Defaults to 0.5.
    min_samples: int, optional
        Minimum number of points required within ``eps`` of a point for
        it to be considered a core point. Defaults to 5.

    Attributes
    ----------
    labels_: numpy.ndarray, shape (n_samples,)
        Cluster label assigned to each sample after fitting. Noise points
        are labeled -1; clusters are labeled 0, 1, 2, ...

    Methods
    -------
    fit(X)
        Fit the model and assign cluster labels to the training data.
    fit_predict(X)
        Fit the model and return the cluster labels.
    """
   
    def __init__(self, eps=0.5, min_samples=5):
        """
        Initialize the DBSCAN model.

        Parameters
        ----------
        eps: float, optional
            Neighborhood radius. Defaults to 0.5.
        min_samples: int, optional
            Minimum number of neighbors for a core point. Defaults to 5.
        """      
        
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None  # -1 = noise

    def _euclidean_distance(self, X, point):
        """
        Compute the Euclidean distance from sample in X to a given point.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Data matrix.
        point: numpy.ndarray, shape (n_features,)
            Reference point.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Euclidean distance from each row of ``X`` to ``point``.
        """     
        
        return np.linalg.norm(X - point, axis=1)

    def _region_query(self, X, idx):
        """
        Find all points within ``eps`` of a given point.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Data matrix.
        idx: int
            Index of the query point in ``X``.

        Returns
        -------
        numpy.ndarray, shape (n_neighbors,)
            Indices of all points (including ``idx`` itself) within
            distance ``eps`` of ``X[idx]``.
        """    
        
        distances = self._euclidean_distance(X, X[idx])

        # Return all points within the specifed radius of a given point
        return np.where(distances <= self.eps)[0]

    # Fit method
    def fit(self, X):
        """
        Fit the DBSCAN model and assign cluster labels.

        Iterates over all unvisited points. Points with fewer than
        ``min_samples`` neighbors within ``eps`` are labeled as noise (-1).
        Core points seed a new cluster, which is then expanded by
        ``_expand_cluster``. Sets ``self.labels_`` upon completion.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data.
        """    
        
        X = np.array(X)
        n_samples = X.shape[0]

        # Initialize all samples as noise
        self.labels_ = -1 * np.ones(n_samples)
        visited = np.zeros(n_samples, dtype=bool)

        cluster_id = 0

        for i in range(n_samples):
            if visited[i]:
                continue

            visited[i] = True
            neighbors = self._region_query(X, i)

            # If there are not enough neighbors, label as noise
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1
            else:
                # Expand cluster
                self._expand_cluster(X, i, neighbors, cluster_id, visited)
                cluster_id += 1

    def _expand_cluster(self, X, idx, neighbors, cluster_id, visited):
        """
        Expand a cluster from a core point by adding all density-reachable points.

        Assigns ``cluster_id`` to the seed point and iteratively adds
        unvisited neighbors to the cluster. If a newly visited neighbor is
        itself a core point, its neighbors are appended to the expansion
        queue. Any point still labeled as noise is reassigned to the current
        cluster (border point).

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Data matrix.
        idx: int
            Index of the core point seeding this cluster.
        neighbors: numpy.ndarray
            Initial set of neighbor indices returned by ``_region_query``.
        cluster_id: int
            Integer label to assign to this cluster.
        visited: numpy.ndarray of bool, shape (n_samples,)
            Boolean array tracking which points have been visited.
        """    
        
        self.labels_[idx] = cluster_id

        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                new_neighbors = self._region_query(X, neighbor_idx)

                # If neighbor is a core point, add its neighbors
                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.concatenate((neighbors, new_neighbors))

            # Assign to cluster if not already assigned
            if self.labels_[neighbor_idx] == -1:
                self.labels_[neighbor_idx] = cluster_id

            i += 1

    def fit_predict(self, X):
        """
        Fit the model and return cluster labels for the training data.

        Equivalent to calling ``fit(X)`` and returning ``self.labels_``.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Cluster labels. Noise points are labeled -1; clusters are
            labeled 0, 1, 2, ...
        """    
        
        self.fit(X)
        return self.labels_
