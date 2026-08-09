import numpy as np


def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Perform one Nadam update step.
    """
    # Write code here
    w = np.asarray(w)
    m = np.asarray(m)
    v = np.asarray(v)
    grad = np.asarray(grad)

    # update the momentum
    m = beta1 * m + (1 - beta1) * grad

    # update the second momentum
    v = beta2 * v + (1 - beta2) * grad**2

    # netstrov update
    w = w - lr * (beta1 * m + (1 - beta1) * grad) / (np.sqrt(v) + eps)

    return w, m, v
