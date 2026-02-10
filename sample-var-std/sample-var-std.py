import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    # Write code here
    x = np.array(x)
    n = len(x)
    mean = x.mean()

    var = 1/(n-1) * ((x - mean)**2).sum()

    return (var, np.sqrt(var))