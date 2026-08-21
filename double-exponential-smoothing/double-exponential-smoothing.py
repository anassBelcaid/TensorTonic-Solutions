def double_exponential_smoothing(series, alpha, beta):
    """
    Apply Holt's linear trend method and return the level values.
    """
    # Write code here
    n = len(series)

    l = n * [series[0]]
    trend = series[1] - series[0]

    # initisation
    for i in range(1, n):
        l[i] = alpha * series[i] + (1 - alpha) * (l[i - 1] + trend)
        trend = beta * (l[i] - l[i - 1]) + (1 - beta) * trend
    return l
