import numpy as np


def gini(y):
    y = np.array(y)
    _, counts = np.unique(y, return_counts=True)
    counts = counts / np.sum(counts)
    return 1 - np.sum(counts**2)


def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    N1, N2 = len(y_left), len(y_right)
    N = N1 + N2

    if N == 0:
        return 0.0
    if N1 == 0:
        return gini(y_right)  # ← fixed
    if N2 == 0:
        return gini(y_left)  # ← fixed

    return (N1 / N) * gini(y_left) + (N2 / N) * gini(y_right)



