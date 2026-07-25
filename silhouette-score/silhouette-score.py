import numpy as np


def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float
    """
    # first let's compute the the distance from each poin to any other point
    X = np.asarray(X)
    labels = np.asarray(labels)
    N = X.shape[0]
    K = len(np.unique(labels))
    S = 0

    # creating the one-hot for gathering
    oneHot = np.zeros((N, K))
    oneHot[np.arange(N), labels] = 1
    counts = oneHot.sum(axis=0)
    for i in range(N):
        label = labels[i]
        Distances = np.sqrt(((X[i][None, :] - X) ** 2).sum(axis=1))
        sum_D = Distances.T @ oneHot
        ai = sum_D[label] / (counts[label] - 1)
        sum_D = sum_D / counts
        bi = min([v for (i, v) in enumerate(sum_D) if i != label])

        S += (bi - ai) / (N * max(ai, bi))

    return S

