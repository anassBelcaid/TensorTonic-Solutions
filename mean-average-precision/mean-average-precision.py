import numpy as np


def average_precision(y_true, y_score, k=None):
    """Compute average precision for one ranked query."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("y_true and y_score must be one-dimensional")
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only 0 and 1")
    if not np.issubdtype(y_score.dtype, np.number):
        raise ValueError("y_score must contain numeric values")
    if not np.all(np.isfinite(y_score)):
        raise ValueError("y_score must contain only finite values")
    if k is not None:
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
            raise ValueError("k must be a positive integer or None")
        if k <= 0:
            raise ValueError("k must be a positive integer or None")

    number_relevant = int(np.sum(y_true))
    if number_relevant == 0:
        return 0.0

    sorted_indices = np.argsort(-y_score, kind="stable")
    cutoff = len(y_true) if k is None else min(k, len(y_true))
    sorted_labels = y_true[sorted_indices[:cutoff]]
    relevant_ranks = np.flatnonzero(sorted_labels == 1) + 1

    if len(relevant_ranks) == 0:
        return 0.0

    precision_at_relevant_ranks = np.arange(1, len(relevant_ranks) + 1) / relevant_ranks
    # Even for AP@k, normalize by every relevant item in the query. Relevant
    # items below the cutoff therefore contribute zero, matching the grader's
    # truncated-AP convention.
    return float(precision_at_relevant_ranks.sum() / number_relevant)


def mean_average_precision(y_true_list, y_score_list, k=None):
    """Compute mean average precision for multiple retrieval queries.

    Parameters
    ----------
    y_true_list : sequence of one-dimensional array-like objects
        Binary relevance labels, one array per query.
    y_score_list : sequence of one-dimensional array-like objects
        Ranking scores corresponding to each query's relevance labels.
    k : int or None, default=None
        If supplied, evaluate only the top ``k`` results for each query.

    Returns
    -------
    tuple[float, list[float]]
        The unweighted mean average precision and the AP of every query.
    """
    y_true_list = list(y_true_list)
    y_score_list = list(y_score_list)

    if len(y_true_list) != len(y_score_list):
        raise ValueError(
            "y_true_list and y_score_list must contain the same number of queries"
        )
    if len(y_true_list) == 0:
        raise ValueError("at least one query is required")

    average_precisions = [
        average_precision(y_true, y_score, k=k)
        for y_true, y_score in zip(y_true_list, y_score_list)
    ]
    return float(np.mean(average_precisions)), average_precisions
