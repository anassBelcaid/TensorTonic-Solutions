import numpy as np


def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """
    # YOUR CODE HERE
    G = [1]
    # computing the norm
    N = np.linalg.norm(W_hh, ord=2)

    for _ in range(T-1):
        l = G[-1]
        G.append((N * l).item())

    return G

