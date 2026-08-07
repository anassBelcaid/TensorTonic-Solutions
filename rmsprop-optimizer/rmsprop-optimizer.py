import numpy as np


def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    st = beta * s + (1-beta) g**2
    wt = wt-1 - eta * 1./sqrt(st + eps) * g
    """
    s = np.asarray(s)
    w = np.asarray(w)
    g = np.asarray(g)

    s = beta * s + (1 - beta) * g**2

    w = w - lr * (g / np.sqrt(s + eps))

    return w, s
