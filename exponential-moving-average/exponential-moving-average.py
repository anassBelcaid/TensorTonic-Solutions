import numpy as np


def exponential_moving_average(values, alpha):
    """
    Compute the exponential moving average of the given values.
    """
    # Write code here
    values = np.asarray(values, dtype=np.float64)
    ema = values.copy()

    n = len(values)
    for i in range(1, n):
        ema[i] = (1 - alpha) * ema[i - 1] + alpha * ema[i]

    return ema.tolist()

