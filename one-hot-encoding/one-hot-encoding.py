import numpy as np


def one_hot(y, num_classes=None):
    """
    Convert integer labels y ∈ {0,...,K-1} into one-hot matrix of shape (N, K).
    """
    # Write code here
    y = np.asarray(y)

    N = len(y)
    if num_classes is None:
        num_classes = y.max() + 1

    encoding = np.zeros((N, num_classes))

    encoding[range(N), y] = 1
    return encoding
