import numpy as np


def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    recommended = set(recommended[:k])
    relevant = set(relevant)

    precision = len(recommended & relevant) / k
    recall = len(recommended & relevant) / len(relevant)

    return [precision, recall]
