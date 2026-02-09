import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    # Write code here
    probs = np.array([comb(n, k) * p**k * (1-p)**(n-k) for k in range(n+1)])

    return probs[k].item(), probs[:(k+1)].sum().item()
