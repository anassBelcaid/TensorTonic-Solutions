import math


def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    # Write code here
    N = len(prob_distributions)

    P = 0.0
    for distribution, target in zip(prob_distributions, actual_tokens):
        P -= math.log(distribution[target])
    P /= N

    return math.exp(P)

