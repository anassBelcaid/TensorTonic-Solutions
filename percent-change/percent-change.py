import numpy as np


def percent_change(series):
    """
    Compute the fractional change between consecutive values.
    """

    series = np.asanyarray(series)
    result = np.where(series[:-1] != 0, (series[1:] - series[:-1]) / series[:-1], 0)
    return result.tolist()

