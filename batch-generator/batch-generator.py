import numpy as np


def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    """
    Randomly shuffle a dataset and yield mini-batches (X_batch, y_batch).

    Parameters
    ----------
    X, y : array-like
        Samples and their corresponding targets. Their first dimensions must
        have the same length.
    batch_size : int
        Maximum number of samples in each yielded batch.
    rng : numpy.random.Generator, numpy.random.RandomState, int, or None
        Random-number source used for shuffling. An integer is treated as a
        seed. If omitted, a new generator is created.
    drop_last : bool, default=False
        If true, omit an incomplete final batch.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim == 0 or y.ndim == 0:
        raise ValueError("X and y must have a sample dimension")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples")
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(
        batch_size, (int, np.integer)
    ):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    if rng is None or isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(rng)
    if not hasattr(rng, "permutation"):
        raise TypeError("rng must provide a permutation method or be an integer seed")

    indices = rng.permutation(len(X))
    stop = len(X) if not drop_last else len(X) - len(X) % batch_size

    for start in range(0, stop, batch_size):
        batch_indices = indices[start : start + batch_size]
        yield X[batch_indices], y[batch_indices]
