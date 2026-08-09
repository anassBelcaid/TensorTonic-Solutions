import numpy as np


def _dot(a, b):
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


def lbfgs_direction(grad, s_list, y_list):
    """
    Compute the L-BFGS search direction using the two-loop recursion.
    """

    # first let's transform everything to numpy
    grad = np.asarray(grad)
    s = np.asarray(s_list)
    y = np.asarray(y_list)

    rho = 1.0 / (s * y).sum(axis=1)

    # backward step
    q = grad.copy()
    n = s.shape[0]
    alpha = np.zeros(n)
    for i in reversed(range(n)):
        # computing alpha_i
        alpha[i] = rho[i] * np.dot(s[i], q)

        # update q
        q = q - alpha[i] * y[i]

    # middle step
    gamma = np.dot(s[n - 1], y[n - 1]) / np.dot(y[n - 1], y[n - 1])
    r = gamma * q

    # forward loop
    for i in range(n):
        beta = rho[i] * np.dot(y[i], r)
        r = r + s[i] * (alpha[i] - beta)

    return -r
