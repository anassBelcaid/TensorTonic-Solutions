def simple_moving_average(values, window_size):
    """
    Compute the simple moving average of the given values.
    """
    # Write code here
    n = len(values)
    result = []

    for start in range(n - window_size + 1):
        result.append(sum(values[start : (start + window_size)]) / window_size)
    return result
