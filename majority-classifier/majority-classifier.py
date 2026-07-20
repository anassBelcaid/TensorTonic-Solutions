import numpy as np


def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # Write code here
    labels, counts = np.unique(y_train, return_counts=True)
    m = len(X_test)

    # getting the unique labels with the highest frequency
    highest_frequency = [L[0] for L in zip(labels, counts) if L[1] == np.max(counts)]
    vote = highest_frequency[0].item()

    return m * [vote]

