import numpy as np


def impute_missing(X, strategy="mean"):
    """
    Fill NaN values (including the string "nan") in each feature column using
    the column mean or median.
    """
    X = np.asarray(X, dtype=float)
    masked_X = np.ma.masked_invalid(X)

    if strategy == "mean":
        reduction = np.ma.mean(masked_X, axis=0, keepdims=True)
    elif strategy == "median":
        reduction = np.ma.median(masked_X, axis=0, keepdims=True)
    else:
        raise ValueError("strategy must be 'mean' or 'median'")

    # A fully missing column has no calculable mean or median. Use the
    # assignment's required fallback of zero for such columns.
    reduction = np.ma.filled(reduction, 0.0)

    return np.where(np.isnan(X), reduction, X)



