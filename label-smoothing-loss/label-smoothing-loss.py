import numpy as np


def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    predictions = np.asarray(predictions)
    K = len(predictions)
    Q = (epsilon / K) * np.ones(K)
    Q[target] += 1 - epsilon

    return -np.sum(Q * np.log(predictions))
