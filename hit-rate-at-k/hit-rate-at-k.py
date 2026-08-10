def good_match(recommentation, ground_truth):
    return any([r in ground_truth for r in recommentation])


def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """
    # Write code here
    # cutting
    recommendations = [set(rec[:k]) for rec in recommendations]
    ground_truth = [set(gt) for gt in ground_truth]

    # number of user
    n = len(ground_truth)
    s = 0
    for i in range(n):
        s += 1 if good_match(recommendations[i], ground_truth[i]) else 0
    return s / n
