import numpy as np


def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    # Write code here
    g = np.array(g)

    norm_g = np.sqrt(np.sum(g**2))
    if norm_g == 0 or max_norm <= 0:
        return g

    return np.where(norm_g <= max_norm, g, g * max_norm / norm_g)
