import numpy as np


def stratified_split(X, y, test_size=0.2, rng=None):
    """
    Split features X and labels y into train/test while preserving class proportions.

    Samples are shuffled within each class before splitting. The class-specific
    partitions are then joined without an additional shuffle.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim == 0 or y.ndim != 1:
        raise ValueError("X must have a sample dimension and y must be one-dimensional")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples")
    if isinstance(test_size, (bool, np.bool_)) or not isinstance(
        test_size, (int, float, np.integer, np.floating)
    ):
        raise TypeError("test_size must be a number")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator or None")

    labels = np.unique(y)
    if len(y) == 0:
        raise ValueError("X and y must not be empty")

    train_parts = []
    test_parts = []

    for label in labels:
        class_indices = np.flatnonzero(y == label)
        if rng is not None:
            class_indices = rng.permutation(class_indices)

        class_test_count = int(round(len(class_indices) * float(test_size)))
        test_parts.append(class_indices[:class_test_count])
        train_parts.append(class_indices[class_test_count:])

    # Shuffling decides membership, not the order returned to the caller.
    train_indices = np.sort(np.concatenate(train_parts))
    test_indices = np.sort(np.concatenate(test_parts))

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]



