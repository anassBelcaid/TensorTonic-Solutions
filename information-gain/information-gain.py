import numpy as np


def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0


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

    return _entropy(y) - n1 / n * _entropy(left) - n2 / n * _entropy(right)
