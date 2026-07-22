import numpy as np


def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores, same shape as y_true
    reduction: "mean" or "sum"
    Return: float
    """
    dtype = np.float64
    reduction = np.mean if reduction == "mean" else np.sum
    y_true = np.array(y_true, dtype=dtype)
    y_score = np.array(y_score, dtype=dtype)

    return reduction(np.maximum(margin - y_true * y_score, 0))

