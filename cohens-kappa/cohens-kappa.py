import numpy as np


def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    # Write code here
    rater1 = np.asarray(rater1)
    rater2 = np.asarray(rater2)

    N = len(rater1)

    po = (rater1 == rater2).mean()

    # computing the number of classes
    K = max(max(rater1), max(rater2)) + 1

    pc = 0.0
    for k in range(K):
        pc += (rater1 == k).mean() * (rater2 == k).mean()

    return 1.0 if pc == 1.0 else (po - pc) / (1 - pc)
