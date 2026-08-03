import numpy as np


def expected_calibration_error(y_true, y_pred, n_bins):
    """
    Compute Expected Calibration Error.
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # TODO: How to partition the bins using numpy efficiently
    # without using [0,1] / bins

    bins_edges = np.histogram_bin_edges(y_pred, range=(0, 1), bins=n_bins)
    bin_ids = np.digitize(y_pred, bins_edges[1:-1])
    N = len(y_true)

    ece = 0.0
    for bin in range(n_bins):
        mask = bin_ids == bin

        if not np.any(mask):
            continue
        accuracy = np.mean(y_true[mask])
        confidence = np.mean(y_pred[mask])
        weight = np.mean(mask)
        ece += weight * abs(accuracy - confidence)
    return ece
