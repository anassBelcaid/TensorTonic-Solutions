import numpy as np

def relu(x):
    return np.maximum(x, 0)

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns: np.ndarray with bottleneck residual block output (compress, process, expand + skip)
    """
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    W3 = np.array(W3)
    Ws = np.array(Ws)

    h1 = relu(x @ W1)      # compress
    h2 = relu(h1 @ W2)     # process
    out = h2 @ W3          # expand — no relu here

    shortcut = x
    if shortcut.shape != out.shape:
        shortcut = x @ Ws

    return relu(out + shortcut)   # single relu after addition
