import numpy as np


def rolling_std(values, window_size):
    """
    Compute the rolling population standard deviation.
    """
    values = np.asarray(values)

    result = []
    for window in np.lib.stride_tricks.sliding_window_view(values, window_size):
        result.append(float(window.std()))

    return result
