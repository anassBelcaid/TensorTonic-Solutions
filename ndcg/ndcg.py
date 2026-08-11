import math


def dcg(relevance):
    """
    compute the discounted cumultaive gain
    """
    return sum([(2**r - 1) / math.log(i + 2) for i, r in enumerate(relevance)])


def ndcg(relevance_scores, k):
    """
    Compute NDCG@k.
    """
    # Write code here
    gain = dcg(relevance_scores[:k])
    k = min(k, len(relevance_scores))

    # now we sort to get the idea gain
    relevance_scores.sort(reverse=True)
    ideal = dcg(relevance_scores[:k])
    if ideal == 0:
        return 0.0
    return gain / ideal



