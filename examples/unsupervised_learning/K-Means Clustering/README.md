# K-Means Clustering

This notebook introduces the K-Means clustering algorithm, an unsupervised algorithm that aims to find groups within given data. For this algorithm, we first specify a *K* value, which determines the number of groups created from the feature data. Then, the algorithm randomly assigns each data point to one of the K clusters. Next, we find the center of each cluster. After we determine these centroids, we re-assign each data point to the closest centroid, which is calculated by using the Euclidean distance formula.  

After re-assigning the points, we repeat this procedure, calculating the new centroids and re-assigning points to the closest ones. We repeat this process until the clusters converge, i.e. there is no change in the clusters. The goal of K-Means is to minimize the within-cluster variance, producing compact, well-separated clusters. A more detailed overview can be found below.


## Overview of the Algorithm

### 1. Initialize Centroids

The first step is to select *K* initial centroids. In the KMeans class, centroids are initialized by randomly sampling from the data. 

### 2. Assignment Step

With centroids initialized, each data point is assigned to the cluster whose centroid is closest to it. The distance between a point $x_i$ and centroid $\mu_k$ is measured using the Euclidean distance:

$$
d(x_i, \mu_k) = \sqrt{\sum_{j=1}^{n} (x_{ij} - \mu_{kj})^2}
$$

Each point $x_i$ is assigned to the cluster $c_i$ that minimizes this distance:

$$
c_i = \underset{k \in \{1, \ldots, K\}}{\arg\min} \ d(x_i, \mu_k)
$$

This assignment step partitions the dataset into *K* clusters $C_1, C_2, \ldots, C_K$.


### 3. Update Step

Once all points have been assigned, the centroid of each cluster is recomputed as the mean of all points currently assigned to it:

$$
\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i
$$

where $|C_k|$ is the number of points in cluster $C_k$. Moving the centroid to the mean of its assigned points is what minimizes the within-cluster sum of squared distances, which is the objective function *J* that K-Means seeks to minimize:

$$
J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \| x_i - \mu_k \|^2
$$


### 4. Iterate Until Convergence

The assignment and update steps are repeated until the algorithm converges. Convergence is reached when the centroids or cluster assignments no longer change. The KMeans class runs the algorithm multiple times (with the default being 100) and retains the solution with the lowest final value of $J$.