import numpy as np


def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    # Write code here
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)

    values = (tpr[1:] + tpr[:-1]) / 2.0
    h = fpr[1:] - fpr[:-1]

    return np.sum(values * h)
