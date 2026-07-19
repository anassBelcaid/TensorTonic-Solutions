import numpy as np


def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    y = np.array(y)
    _, counts = np.unique(y, return_counts=True)
    counts = counts / np.sum(counts, keepdims=True)
    print(counts)

    return -np.sum(counts * np.log2(counts))




