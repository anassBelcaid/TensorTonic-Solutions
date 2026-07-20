import numpy as np


def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return indices of the k nearest
    neighbors in X_train for each sample in X_test.

    Parameters
    ----------
    X_train : array-like of shape (n_train, n_features)
    X_test : array-like of shape (n_test, n_features)
    k : int

    Returns
    -------
    result : ndarray of shape (n_test, k)
        Indices of the k nearest neighbors. If k > n_train,
        remaining entries are filled with -1.
    """

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    # Handle 1D feature vectors
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    # Squared norms
    train_sq = np.sum(X_train**2, axis=1)
    test_sq = np.sum(X_test**2, axis=1)

    # Vectorized squared Euclidean distances
    distances = test_sq[:, None] + train_sq[None, :] - 2 * X_test @ X_train.T

    # Numerical stability
    distances = np.maximum(distances, 0)

    n_test = X_test.shape[0]
    n_train = X_train.shape[0]

    # Output initialized with -1 padding
    result = np.full((n_test, k), -1, dtype=int)

    num_neighbors = min(k, n_train)

    if num_neighbors > 0:
        nearest = np.argsort(distances, axis=1, kind="stable")[:, :num_neighbors]
        result[:, :num_neighbors] = nearest
    return result

 

