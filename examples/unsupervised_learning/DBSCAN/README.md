# DBSCAN

This notebook introduces DBSCAN (Density-Based Spatial Clustering of Applications with Noise), an unsupervised clustering algorithm that groups data points based on the density of their neighborhoods. Unlike centroid-based methods, DBSCAN does not require specifying the number of clusters in advance and is capable of discovering clusters of arbitrary shape. The algorithm classifies each point into one of three categories:

* **Core points** have a sufficiently dense neighborhood and form the interior of a cluster
* **Border points** lie within the neighborhood of a core point but are not dense enough to be core points themselves
* **Noise points** (outliers) do not belong to any cluster

A more detailed overview can be found below.

## Overview of the Algorithm

### 1. Define Neighborhood Parameters 

DBSCAN relies on two user-defined parameters: $\varepsilon$ (epsilon) and *min_samples*. The parameter $\varepsilon$ defines the radius of the neighborhood around a point, and *min_samples* is the minimum number of points required within that radius for a point to be considered a core point. Together, these parameters control the density threshold that separates clusters from noise.

The $\varepsilon$-neighborhood of a point $p$ is the set of all points within distance $\varepsilon$ of $p$:

$$
N_\varepsilon(p) = \{ q \in D \mid d(p, q) \leq \varepsilon \}
$$

where $D$ is the dataset and $d(p, q)$ is the distance between points $p$ and $q$. The distance formula used in the DBSCAN class is Euclidean distance:

$$
d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}
$$

A point $p$ is a **core point** if it has at least *MinPts* points within its $\varepsilon$-neighborhood:

$$
|N_\varepsilon(p)| \geq \text{MinPts}
$$


### 2. Identify Density-Reachable Points

Once core points are identified, DBSCAN expands clusters by following chains of density-reachable points. A point $q$ is said to be **directly density-reachable** from a core point $p$ if $q$ lies within the $\varepsilon$-neighborhood of $p$:

$$
q \in N_\varepsilon(p)
$$

More generally, a point $q$ is **density-reachable** from $p$ if there exists a chain of points $p_1, p_2, \ldots, p_n$ where $p_1 = p$, $p_n = q$, and each $p_{i+1}$ is directly density-reachable from $p_i$. Two points are **density-connected** if there exists a third point from which both are density-reachable. Clusters are defined as maximal sets of density-connected points.


### 3. Expand Clusters and Label Points

The algorithm begins by selecting an arbitrary unvisited point. If the point is a core point, a new cluster is formed and expanded by iteratively adding all density-reachable points to the cluster. If a newly added point is itself a core point, its neighborhood is also explored and added to the cluster. This process continues until no more points can be added, at which point the algorithm moves on to the next unvisited point.

Points that are visited but cannot be assigned to any cluster — because they are not density-reachable from any core point — are labeled as noise. Border points are assigned to the cluster of a neighboring core point but do not trigger further expansion themselves.

The algorithm terminates once all points have been visited, yielding a set of clusters $C_1, C_2, \ldots, C_k$ and a set of noise points $N$, where:

$$
C_1 \cup C_2 \cup \cdots \cup C_k \cup N = D
$$