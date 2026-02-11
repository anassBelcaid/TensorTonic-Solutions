import numpy as np

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    # Write code here
    # the formulat is e^{-alpha}x^k / (k)!
    # ===> applying log ==> (-lambda) + k * log(x) - sum(log(k))

    # getting the sum of logs
    logs = np.array([-lam + a * np.log(lam)-np.sum(np.log(np.arange(1,a+1))) for a in range(0,k+1)])

    # retreiving to exponential
    probs = np.exp(logs)

    return float(probs[-1]), float(probs.sum())


