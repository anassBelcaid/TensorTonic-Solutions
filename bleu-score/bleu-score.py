from collections import Counter
from math import log, exp


def generate_n_grams(tokens, n):
    """
    function to generate a hashmap containing the n-grams counts
    """

    if n <= 0:
        raise ValueError("n must be a positive integer")

    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def modified_precision(candidate, reference, n):
    """
    function to compute the modified precision between candidate and reference
    """
    # generate the n_grames
    candidate_grams = generate_n_grams(candidate, n)
    reference_grams = generate_n_grams(reference, n)

    numerator, denominator = 0, 0
    for key, candidate_count in candidate_grams.items():
        reference_count = reference_grams[key]
        numerator += min(candidate_count, reference_count)
        denominator += candidate_count

    if denominator == 0:
        return 0.0

    return numerator / denominator


def brievty_penalty(candidate, reference):
    c = len(candidate)
    r = len(reference)

    if c == 0:
        return 0

    if c > r:
        return 1

    return exp(1 - r / c)


def bleu_score(candidate, reference, max_n):
    """
    Compute the BLEU score for a candidate translation.
    """
    if not candidate:
        return 0.0

    # Write code here
    P = 0
    for n in range(1, max_n + 1):
        precision = modified_precision(candidate, reference, n)
        if precision == 0:
            return 0.0
        P += log(precision) / max_n
    P = exp(P)
    BP = brievty_penalty(candidate, reference)

    return BP * P
