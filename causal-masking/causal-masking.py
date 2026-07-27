import numpy as np


def apply_causal_mask(scores, mask_value=-1e9):
    result = np.asarray(scores, dtype=float).copy()

    T = result.shape[-1]
    if result.shape[-2] != T:
        raise ValueError("The final two dimensions must have shape (T, T).")

    future_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    result[..., future_mask] = mask_value

    return result
