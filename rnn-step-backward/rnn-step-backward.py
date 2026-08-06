import numpy as np


def rnn_step_backward(dh, cache):
    """
    Backpropagate through one vanilla-RNN step.

    The forward equation is ``h_t = tanh(W @ x_t + U @ h_prev + b)``.
    ``cache`` must contain ``[x_t, h_prev, h_t, W, U, b]``.

    Args:
        dh: upstream gradient with respect to h_t (shape: H,)
        cache: x_t, h_prev, h_t, W, U, and b from the forward step

    Returns:
        dx_t: gradient wrt input x_t      (shape: D,)
        dh_prev: gradient wrt previous h (shape: H,)
        dW: gradient wrt W               (shape: H x D)
        dU: gradient wrt U               (shape: H x H)
        db: gradient wrt bias            (shape: H,)
    """

    x_t, h_prev, h_t, W, U, b = cache
    x_t = np.asarray(x_t)
    h_prev = np.asarray(h_prev)
    h_t = np.asarray(h_t)
    W = np.asarray(W)
    U = np.asarray(U)
    dh = np.asarray(dh)

    # If a = W @ x_t + U @ h_prev + b and h_t = tanh(a), then
    # dL/da = dL/dh_t * (1 - tanh(a)^2).
    da = dh * (1.0 - h_t**2)

    dx_t = W.T @ da
    dh_prev = U.T @ da
    dW = np.outer(da, x_t)
    dU = np.outer(da, h_prev)
    db = da.copy()

    # Return tuple of 5 gradients: (dx_t, dh_prev, dW, dU, db)
    return dx_t,  dh_prev,  dW,  dU,  db



