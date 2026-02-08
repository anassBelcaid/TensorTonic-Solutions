import numpy as np

def percentile(x, q):
    if q == 0:
        return x[0]
    if q == 100:
        return x[-1]

    # computing L
    n = len(x)
    L = (q * (n-1)) / 100

    if L == int(L):
        return x[int(L)]

    r2 =  L - int(L)
    r1 = 1 - r2

    return r1 * x[int(L)] + r2 * x[int(L) + 1]
    
def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    # Write code here
    x.sort()

    return np.array([percentile(x, q1) for q1 in q])


    
