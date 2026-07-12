import numpy as np


def relu(x):
    return np.maximum(x, 0)


def conv_block(x, W1, W2, Ws):
    """
    Returns: np.ndarray with sum of main path output and projected shortcut
    """
    # YOUR CODE HERE
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    Ws = np.array(Ws)

    h = relu(x @ W1)
    z = h @ W2
    s = x @ Ws

    return relu(z + s)


