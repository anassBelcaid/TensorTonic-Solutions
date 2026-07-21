import numpy as np


def gaussian_naive_bayes(X_train, y_train, X_test):
    """
    Predict class labels for test samples using Gaussian Naive Bayes.

    Args:
        X_train: (n_train, d)
        y_train: (n_train,)
        X_test: (n_test, d)

    Returns:
        Predicted class labels for each test sample.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    # Binary class prior
    _, py = np.unique(y_train, return_counts=True)

    log_prior = np.log(py / len(y_train))

    D = X_train.shape[1]
    C = len(py)

    # Mean and variance for each feature and class
    means = np.zeros((C, D))
    vars = np.zeros((C, D))

    mask = y_train == 1

    for i in range(D):
        for c in range(C):
            mask = y_train == c
            feature = X_train[mask, i]
            means[c, i] = feature.mean()
            vars[c, i] = feature.var() + 1e-9

    result = np.empty(X_test.shape[0], dtype=int)

    for i, x in enumerate(X_test):
        scores = np.zeros(C)

        for c in range(C):
            scores[c] = (
                log_prior[c]
                - 0.5 * np.sum(np.log(2 * np.pi * vars[c]))
                - 0.5 * np.sum((x - means[c]) ** 2 / vars[c])
            )

        result[i] = np.argmax(scores)

    return result.tolist()
