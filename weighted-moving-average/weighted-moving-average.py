import numpy as np


def weighted_moving_average(values, weights):
    """
    Compute the weighted moving average using the given weights.
    """
    values = np.asarray(values)
    weights = np.asarray(weights)
    weights = weights / weights.sum()

    result = []
    for window in np.lib.stride_tricks.sliding_window_view(values, len(weights)):
        result.append(float(np.sum(window * weights)))
    return result

