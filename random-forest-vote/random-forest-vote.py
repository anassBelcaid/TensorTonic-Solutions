import numpy as np


def _major_vote(votes):
    votes = np.array(votes)
    labels, counts = np.unique(votes, return_counts=True)
    max_count = counts.max().item()

    majors_labels = [L[0] for L in zip(labels, counts) if L[1] == max_count]
    majors_labels.sort()
    return majors_labels[0].item()


def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    predictions = np.atleast_2d(predictions)
    result = []
    for j in range(predictions.shape[1]):
        result.append(_major_vote(predictions[:, j]))
    return result

