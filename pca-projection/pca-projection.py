import numpy as np


def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    X = np.asarray(X)

    # Center the data
    X = X - np.mean(X, axis=0, keepdims=True)

    # Covariance matrix
    n = X.shape[0]
    cov = (X.T @ X) / n

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort by descending eigenvalue
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx[:k]]

    # Project
    return X @ eigvecs

