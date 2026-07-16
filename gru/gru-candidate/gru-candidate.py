import numpy as np


def candidate_hidden(
    h_prev: np.ndarray,
    x_t: np.ndarray,
    r_t: np.ndarray,
    W_h: np.ndarray,
    b_h: np.ndarray,
) -> np.ndarray:
    """
    Compute candidate: h_tilde = tanh(W_h @ [r*h, x] + b_h)
    """
    # reseting
    h_prev = r_t * h_prev

    # stacking
    stacked = np.concatenate([h_prev, x_t], axis=-1)

    return np.tanh(stacked @ W_h.T + b_h)
