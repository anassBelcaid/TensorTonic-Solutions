import numpy as np


def autocorrelation(series, max_lag):
    """
    Compute the autocorrelation of a time series for lags 0 to max_lag.
    """
    series = np.asarray(series)
    mean = series.mean()
    var = ((series - mean) ** 2).sum()

    result = [1]
    for k in range(1, max_lag + 1):
        if var == 0:
            result.append(0.0)

        else:
            lagged = series[k:]
            cov = ((lagged - mean) * (series[:-k] - mean)).sum()
            result.append(float(cov / var))
    return result

