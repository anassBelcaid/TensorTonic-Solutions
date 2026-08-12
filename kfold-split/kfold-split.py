import numpy as np


def get_split_section(n, k):
    """
    Function to compute the split sections
    """
    rem = n % k

    size = n // k

    result = []
    end = 0
    for _ in range(rem):
        end += size + 1
        result.append(end)

    while end + size < n:
        end += size
        result.append(end)
    return result


def kfold_split(N, k, shuffle=True, rng=None):
    """
    Returns: list of length k with tuples (train_idx, val_idx)
    """
    # Write code here
    # let's check the rng first
    idxs = np.arange(N)
    if shuffle:
        if rng is None:
            np.random.shuffle(idxs)
        else:
            idxs = rng.permutation(idxs)
    splits = np.split(idxs, get_split_section(N, k))
    print(splits)

    folds = []
    for fold in range(k):
        valid = splits[fold]
        train = np.hstack(splits[:fold] + splits[fold + 1 :])
        folds.append((train, valid))

    return folds
