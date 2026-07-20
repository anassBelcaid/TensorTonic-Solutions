import numpy as np


def gini(p):
    _, counts = np.unique(p, return_counts=True)

    counts = counts / np.sum(counts)

    return 1.0 - np.sum(counts**2)


def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Use the _entropy() helper above.
    """
    y = np.array(y)
    split_mask = np.array(split_mask)

    left = y[split_mask]
    right = y[~split_mask]

    n = len(y)
    n1 = split_mask.sum()
    n2 = n - n1

    return gini(y) - n1 / n * gini(left) - n2 / n * gini(right)


def decision_tree_split(X, y):
    """
    Find the best feature and threshold to split the data.
    """
    best_gain = 0
    best_feature = None
    best_cut = None
    X = np.array(X)
    y = np.array(y)

    D = X.shape[1]

    for j in range(D):
        # computing the possible cuts
        feature = X[:, j]
        values = np.unique(feature)
        cuts = (values[1:] + values[:-1]) / 2.0

        for cut in cuts:
            # computing the mask
            mask = feature < cut
            gain = information_gain(y, mask)

            if gain > best_gain:
                best_gain = gain
                best_feature = j
                best_cut = cut.item()

    return [best_feature, best_cut]

