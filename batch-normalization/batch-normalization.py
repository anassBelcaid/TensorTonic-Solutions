import numpy as np


def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).

    (N, D): normalize each feature over the batch axis (axis=0)
    (N, C, H, W): normalize each channel over axes (0,2,3)
    """
    # Write code here
    x = np.asarray(x)
    gamma = np.asarray(gamma)
    beta = np.asarray(beta)
    if x.ndim == 2:
        axis = 0
        expected_shape = (x.shape[1],)
    elif x.ndim == 4:
        axis = (0, 2, 3)
        expected_shape = (x.shape[1],)
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)
    else:
        raise ValueError(f"x must be 2D or 4D, got {x.ndim}D")

    if gamma.size != expected_shape[0] or beta.size != expected_shape[0]:
        raise ValueError(
            f"gamma and beta must each contain {expected_shape[0]} values"
        )

    mean = x.mean(axis=axis, keepdims=True)

    var = x.var(axis=axis, keepdims=True, ddof=0)

    return gamma * ((x - mean) / (np.sqrt(var + eps))) + beta
