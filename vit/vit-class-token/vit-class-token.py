import numpy as np


def prepend_class_token(patches: np.ndarray, cls_token: np.ndarray) -> np.ndarray:
    """
    Returns the float64 sequence with the class token at position zero.
    """
    B, _, D = patches.shape
    broadcasted = np.broadcast_to(cls_token, (B, 1, D))
    return np.concat((broadcasted, patches), axis=1)
