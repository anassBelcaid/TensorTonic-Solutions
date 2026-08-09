import numpy as np


def adadelta_step(w, grad, E_grad_sq, E_update_sq, rho=0.9, eps=1e-6):
    """
    Perform one AdaDelta update step.
    """
    # Write code here
    w = np.asarray(w)
    grad = np.asarray(grad)
    E_grad_sq = np.asarray(E_grad_sq)
    E_update_sq = np.asarray(E_update_sq)

    # update the expectation for the gradient square
    E_grad_sq = rho * E_grad_sq + (1 - rho) * grad**2

    # compute the parameter update
    deltaW = -np.sqrt(E_update_sq + eps) / np.sqrt(E_grad_sq + eps) * grad

    # update the squared update expected value
    E_update_sq = rho * E_update_sq + (1 - rho) * deltaW**2

    # parameter update
    w = w + deltaW

    return w, E_grad_sq, E_update_sq

