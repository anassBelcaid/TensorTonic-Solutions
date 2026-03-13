import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    # computing the log for y_pred
    log_prediction = np.log(y_pred)

    n = len(y_pred)
    return -np.mean(log_prediction[range(n), y_true])


