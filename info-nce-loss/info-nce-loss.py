import numpy as np


def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    # Write code here
    Z1 = np.asarray(Z1)
    Z2 = np.asarray(Z2)

    S = (Z1 @ Z2.T) / temperature

    S = S - S.max(
        axis=1, keepdims=True
    )  # substraction trick don't know if it still valid like the softmax

    S = np.exp(S)
    S = S / S.sum(axis=1, keepdims=True)

    return -np.mean(np.log(np.diag(S)))



