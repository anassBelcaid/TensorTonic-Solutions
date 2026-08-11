import numpy as np


def target_encoding(categories, targets):
    """
    Replace each category with the mean target value for that category.
    """
    categories = np.asarray(categories)
    targets = np.asarray(targets)

    classes, indices = np.unique(categories, return_inverse=True)
    N, K = len(categories), len(classes)

    # creating the one hot
    one_hot = np.zeros((N, K))
    one_hot[np.arange(N), indices] = 1

    # computing the cardinal of each class
    cardinal = one_hot.sum(axis=0)

    # computing the scores
    scores = one_hot.T @ targets

    # group values
    group_val = scores / cardinal

    return group_val[indices].tolist()
