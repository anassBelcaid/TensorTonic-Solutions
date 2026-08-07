import numpy as np
from math import floor


def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    # Write code here
    X = np.asarray(X)
    H, W = X.shape

    h, w = floor((H - pool_size) / stride) + 1, floor((W - pool_size) / stride) + 1

    # Hint: explore np.lib.stride_tricks.sliding_window_view
    slides = np.lib.stride_tricks.sliding_window_view(X, (pool_size, pool_size))

    return slides[::stride, ::stride].max(axis=(-1, -2)).tolist()


