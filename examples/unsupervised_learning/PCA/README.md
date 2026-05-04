# Principal Component Analysis (PCA)

This notebook introduces Principal Component Analysis (PCA), an unsupervised dimensionality reduction technique that transforms a dataset into a lower-dimensional representation while preserving as much variance as possible. PCA works by identifying the directions in the feature space along which the data varies the most, and projecting the data onto those directions. The key components of the algorithm are:

* **Principal components** are the orthogonal directions of maximum variance in the data, ordered from most to least variance explained
* **The covariance matrix** captures the pairwise relationships between features and is used to identify the principal components
* **The projection** maps the original high-dimensional data onto the lower-dimensional subspace defined by the top principal components

A more detailed overview can be found below.

## Overview of the Algorithm

### 1. Standardize the Data

Before computing principal components, the data must be scaled so that features with larger scales don't dominate. In this package, it is recommended to scale the data using the StandardScaler class. StandardScaler works by taking each data point, subtracting the mean of the feature it belongs to, and dividing by the standard deviation of that feature. In other words, for each feature $j$, each value is transformed as:

$$
x_{ij}' = \frac{x_{ij} - \bar{x}_j}{s_j}
$$

where $\bar{x}_j$ is the mean and $s_j$ is the standard deviation of feature $j$ across all $n$ samples. After standardization, the data matrix $X$ has shape $n \times p$, where $p$ is the number of features.


### 2. Compute the Covariance Matrix

The next step is to compute the covariance matrix of the standardized data, which indicates the direction and relative magnitude with which each pair of features varies together. The covariance matrix $\Sigma$ is a $p \times p$ symmetric matrix defined as:

$$
\Sigma = \frac{1}{n - 1} X^T X
$$

The diagonal entries of $\Sigma$ are the variances of each feature, and the off-diagonal entries are the covariances between pairs of features. Large off-diagonal values indicate that two features are correlated and therefore contain redundant information, which PCA can compress.


### 3. Compute Eigenvectors and Eigenvalues

PCA identifies the principal components by performing an eigendecomposition of the covariance matrix. The eigenvectors and eigenvalues of $\Sigma$ satisfy:

$$
\Sigma v = \lambda v
$$

where $v$ is an eigenvector and $\lambda$ is its corresponding eigenvalue. Each eigenvector defines a direction in the original feature space, and its associated eigenvalue measures the amount of variance in the data explained by that direction. The eigenvectors are orthogonal to one another, meaning the principal components capture independent sources of variation in the data.

The eigenvalues are sorted in descending order, $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_p$, and the corresponding eigenvectors $v_1, v_2, \ldots, v_p$ are the principal components ordered from most to least variance explained. The proportion of total variance explained by the $k$-th principal component is:

$$
\text{Explained Variance Ratio}_k = \frac{\lambda_k}{\sum_{j=1}^{p} \lambda_j}
$$

This quantity is useful for choosing how many components to retain. A common rule of thumb is to keep enough principal components so that about 80% of the total variance is explained. 


### 4. Project the Data

Finally, the original data is projected onto the top $d$ principal components to produce the reduced-dimensional representation. The projection matrix $W$ is formed by stacking the top $d$ eigenvectors as columns:

$$
W = \begin{bmatrix} v_1 & v_2 & \cdots & v_d \end{bmatrix}
$$

The transformed dataset $Z$ is then obtained by projecting the standardized data matrix $X$ onto $W$:

$$
Z = X W
$$

where $Z$ has shape $n \times d$. Each row of $Z$ is the lower-dimensional representation of the corresponding original data point, expressed in terms of the principal component axes. The transformation retains the directions of greatest variance and discards the directions that contribute least, effectively compressing the data while minimizing information loss.

While PCA is very useful for reducing high-dimensional data into more manageable datasets for supervised algorithms to learn on, a downside is that the principal components are no longer interpretable. Each component uses a combination of the features in the data, so they don't have units or any real meaning. However, they are still useful for plotting and finding patterns in the data.   