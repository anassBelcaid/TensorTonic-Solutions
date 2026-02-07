import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    # Write code here
    x = np.atleast_1d(x)

    return np.where(x >=0, x, alpha * x)