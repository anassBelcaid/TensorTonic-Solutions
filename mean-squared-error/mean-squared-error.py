import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Write code here
    y_pred, y_true =np.array(y_pred), np.array(y_true)

    if y_pred.shape != y_true.shape:
        return None
    if y_pred.ndim != 1:
        return None
    return float(((y_pred - y_true)**2).mean())
