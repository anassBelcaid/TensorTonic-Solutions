import numpy as np


def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


# def _as2d(a, feat):
#     """Convert 1D array to 2D and track if conversion happened"""
#     a = np.asarray(a, dtype=float)
#     if a.ndim == 1:
#         return a.reshape(1, feat), True
#     return a, False


def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    # Write code here
    x = np.asarray(x)
    h_prev = np.asarray(h_prev)
    if x.ndim == 1:
        x = x[None, :]
        h_prev = h_prev[None, :]
    # params = {
    #     "Wz": (D, H), "Uz": (H, H), "bz": (H,),  # Update gate
    #     "Wr": (D, H), "Ur": (H, H), "br": (H,),  # Reset gate
    #     "Wh": (D, H), "Uh": (H, H), "bh": (H,)   # Candidate
    # }
    Wz = params["Wz"]
    Uz = params["Uz"]
    bz = params["bz"]

    Wr = params["Wr"]
    Ur = params["Ur"]
    br = params["br"]

    Wh = params["Wh"]
    Uh = params["Uh"]
    bh = params["bh"]

    # update gate
    zt = _sigmoid(x @ Wz + h_prev @ Uz + bz)

    # reset gate
    rt = _sigmoid(x @ Wr + h_prev @ Ur + br)

    # candidate
    h_hat = np.tanh(x @ Wh + (rt * h_prev) @ Uh + bh)

    # new hidden state
    h_new = (1 - zt) * h_prev + zt * h_hat

    return h_new.squeeze()
