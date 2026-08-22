import numpy as np


def seasonal_average(series, period):
    """
    Compute the average value for each position in the seasonal cycle.
    """
    series = np.asarray(series)

    result = []
    for p in range(period):
        result.append(float(np.mean(series[p::period])))
    return result
