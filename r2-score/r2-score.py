import numpy as np


def r2_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Constant target
    if np.all(y_true == y_true[0]):
        return 1.0 if np.allclose(y_true, y_pred) else 0.0

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1.0 - ss_res / ss_tot




