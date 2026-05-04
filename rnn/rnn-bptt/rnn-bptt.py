import numpy as np


def bptt_single_step(
    dh_next: np.ndarray,
    h_t: np.ndarray,
    h_prev: np.ndarray,
    x_t: np.ndarray,
    W_hh: np.ndarray,
) -> tuple:
    """
    Backprop through one RNN time step.
    Returns (dh_prev, dW_hh).
    """
    # YOUR CODE HERE
    # computing thhe derivation
    dtanh = (1 - h_t * h_t) * dh_next

    # computing the derivative in respect dWhh
    dW_hh = dtanh.T @ h_prev

    # computing the gradient in respect to dh_prev
    dh_prev = dtanh @ W_hh

    return dh_prev, dW_hh


