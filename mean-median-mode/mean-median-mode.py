import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    # Write code here
    x = np.atleast_1d(x)

    frequencies = Counter(x)
    mode = next(iter(frequencies))

    mean, median = np.mean(x), np.median(x)
    return (mean, median, mode)






