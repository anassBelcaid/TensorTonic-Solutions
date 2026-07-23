import numpy as np


def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    # Write code here
    anchor = np.asarray(anchor)
    positive = np.asarray(positive)
    negative = np.asarray(negative)

    if anchor.ndim == 1:
        anchor = anchor[None, :]
        positive = positive[None, :]
        negative = negative[None, :]

    positive_distance = np.linalg.norm(anchor - positive, axis=1) ** 2
    negative_distance = np.linalg.norm(anchor - negative, axis=1) ** 2

    diff = positive_distance - negative_distance + margin
    return np.mean(np.where(diff > 0, diff, 0))
