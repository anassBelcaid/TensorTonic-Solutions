import numpy as np


def lag_features(series, lags):
    """
    Create a lag feature matrix from the time series.
    """
    series = np.asarray(series)
    lags = np.asarray(lags)
    max_lag = lags.max()

    N = len(series) - max_lag
    D = len(lags)

    result = np.zeros((N, D))
    for row in range(max_lag, len(series)):
        for j, lag in enumerate(lags):
            result[row - max_lag][j] = series[row - lag]

    return result.tolist()

