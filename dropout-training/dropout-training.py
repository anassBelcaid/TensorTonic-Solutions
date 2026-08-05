import numpy as np


def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.asarray(x)
    if rng is None:
        distribution = np.random.random(size=x.shape)
    else:
        distribution = rng.random(size=x.shape)
    mask = distribution <= p

    # return np.where
    dropout_pattern = np.where(mask, 0, 1.0 / (1 - p))

    return x * dropout_pattern, dropout_pattern

