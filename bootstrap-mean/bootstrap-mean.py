import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Returns: (boot_means, lower, upper)
    """
    # Write code here
    n = len(x)
    boot_means = np.array([x[rng.integers(0, n, n)].mean() if rng else x[np.random.randint(0, n, n)].mean() for _ in range(n_bootstrap)])
    err = (1 - ci)/2
    lower, upper = np.quantile(boot_means, err), np.quantile(boot_means,1-err)

    return boot_means, lower, upper







