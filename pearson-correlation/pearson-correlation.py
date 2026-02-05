import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    # Write code here
    X = np.array(X, dtype =float)

    if X.ndim != 2 or X.shape[0] < 2 :
        return None

    centered = X - X.mean(axis=0)

    cov = (centered.T @ centered) / X.shape[0]

    std = np.std(X, axis = 0)

    std_devs = np.outer(std, std)



    with np.errstate(divide='ignore', invalid='ignore'):
        result = cov / std_devs

    np.fill_diagonal(result,1)
    return result










