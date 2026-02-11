import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """
    # Write code here
    F = np.array(C)

    # computing the row, col, overall sums
    R, C , N = np.sum(F, axis=1), np.sum(F, axis=0), np.sum(F)


    # computing expected
    expected = R[:, np.newaxis] * C[np.newaxis, :] / N

    # computing the chi test 
    chi2 = np.sum((F - expected)**2 / expected)

    return float(chi2), expected

