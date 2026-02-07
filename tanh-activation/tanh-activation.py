import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Write code here
    x = np.atleast_1d(x)

    positive, negative = np.exp(x), np.exp(-x)

    return (positive - negative) / (positive + negative)