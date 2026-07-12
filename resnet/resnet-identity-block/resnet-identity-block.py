import numpy as np


def relu(x):
    return np.maximum(x, 0)


def identity_block(x, W1, W2):
    """
    Returns: np.ndarray of shape (batch, channels) with identity residual block output
    """
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    # computing the first block
    h = relu(x @ W1.T)

    # computing the second layer
    return relu(h @ W2.T + x).tolist()
