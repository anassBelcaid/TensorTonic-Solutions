import numpy as np


def cumulative_returns(returns):
    """
    Compute the cumulative return at each time step.
    """
    # Write code here
    W_t = np.array(returns) + 1
    n = len(W_t)
    for i in range(1, n):
        W_t[i] *= W_t[i - 1]

    W_t -= 1
    return W_t.tolist()
