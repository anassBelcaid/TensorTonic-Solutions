import numpy as np


def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask_true = y_true == 1
    mask_pred = y_pred == 1

    tp = np.sum(mask_true & mask_pred)
    tn = np.sum(~mask_true & ~mask_pred)
    fp = np.sum(~mask_true & mask_pred)
    fn = np.sum(mask_true & ~mask_pred)
    return np.mean(y_true == y_pred)
