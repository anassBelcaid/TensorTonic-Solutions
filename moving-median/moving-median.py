import numpy as np
def moving_median(values, window_size):
    """
    Compute the rolling median for each window position.
    """
    values = np.asarray(values)

    result = []
    for window in np.lib.stride_tricks.sliding_window_view(values, window_size):
        result.append(float(np.median(window)))

    return result
