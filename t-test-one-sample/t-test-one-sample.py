import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """

    # convert x to numpy 
    x = np.array(x)

    # getting the size of the sample
    n = len(x)

    # computing sample mean
    x_bar = x.mean()

    # compute the bessel correction standard deviation
    std = np.sqrt(1./(n - 1) * np.sum( (x - x_bar)**2))

    return float((x_bar - mu0) /(std/np.sqrt(n)) )

