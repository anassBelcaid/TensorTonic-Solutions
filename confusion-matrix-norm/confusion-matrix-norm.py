import numpy as np


def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize="none"):
    """
    Compute confusion matrix with optional normalization.
    """
    # Write code here
    if num_classes is None:
        num_classes = max(int(max(y_true)), max(y_pred)) + 1

    confusion = np.zeros((num_classes, num_classes))

    for yt, yp in zip(y_true, y_pred):
        confusion[yt][yp] += 1

    if normalize == "true":
        confusion = confusion / confusion.sum(axis=1, keepdims=True)
    if normalize == "pred":
        confusion = confusion / confusion.sum(axis=0, keepdims=True)

    if normalize == "all":
        confusion /= confusion.sum()

    return confusion.tolist()
