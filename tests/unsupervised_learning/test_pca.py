import numpy as np

from src.ml_package.unsupervised_learning.pca import PCA


def test_pca_centers_data_without_internal_scaling():
    X = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
    ])
    pca = PCA(n_components=1)

    transformed = pca.fit_transform(X)
    centered = X - np.mean(X, axis=0)
    manual_projection = centered @ pca.components_

    assert transformed.shape == (4, 1)
    np.testing.assert_allclose(transformed, manual_projection)
    np.testing.assert_allclose(pca.mean_, np.mean(X, axis=0))
    assert not hasattr(pca, "std_")
