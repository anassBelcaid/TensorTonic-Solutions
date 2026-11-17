import numpy as np

eps = 1e-6


def classification_head(encoder_output: np.ndarray, W_head: np.ndarray) -> np.ndarray:
    """
    Returns float64 class logits with shape (B, C).
    """
    head = encoder_output[:, 0, :]

    # layer norm
    mu = head.mean(axis=-1, keepdims=True)
    var = head.var(axis=-1, keepdims=True)

    z = (head - mu) / np.sqrt(var + eps)

    return z @ W_head

