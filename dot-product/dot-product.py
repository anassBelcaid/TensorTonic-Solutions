import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    # Write code here
    x , y = np.array(x), np.array(y)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError()

    return x.dot(y)
    