import numpy as np


def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Compute unnormalized log posteriors for Bernoulli Naive Bayes.

    Args:
        X_train: (n_train, d) binary feature matrix
        y_train: (n_train,) class labels
        X_test:  (n_test, d) binary feature matrix

    Returns:
        scores: (n_test, n_classes) array of unnormalized log posteriors,
                with columns ordered by ascending class labels.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    classes = np.unique(y_train)
    C = len(classes)
    D = X_train.shape[1]

    # Prior probabilities
    log_prior = np.zeros(C)

    # P(x_j = 1 | y = c)
    theta = np.zeros((C, D))

    for i, c in enumerate(classes):
        Xc = X_train[y_train == c]

        # Prior
        log_prior[i] = np.log(len(Xc) / len(X_train))

        # Laplace smoothing
        theta[i] = (Xc.sum(axis=0) + 1) / (len(Xc) + 2)

    log_theta = np.log(theta)
    log_one_minus_theta = np.log(1 - theta)

    # Compute unnormalized log posteriors
    scores = np.zeros((len(X_test), C))

    for i in range(C):
        scores[:, i] = (
            log_prior[i] + X_test @ log_theta[i] + (1 - X_test) @ log_one_minus_theta[i]
        )

    return scores
