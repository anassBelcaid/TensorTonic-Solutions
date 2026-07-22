import numpy as np


def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    # Write code here
    a = np.asarray(a)
    b = np.asarray(b)
    y = np.asarray(y)

    if a.ndim == 1:
        a = a[None, :]
        b = b[None, :]
    # let's compute the distances
    d = np.linalg.norm(a - b, axis=1)

    reduction = np.mean if reduction == "mean" else np.sum

    return reduction(y * d**2 + (1 - y) * np.maximum(0, margin - d) ** 2)
