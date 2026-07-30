import numpy as np


def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here
    n = len(vocab)
    result = np.zeros(n, dtype=int)
    vocab = {token: i for (i, token) in enumerate(vocab)}

    np.add.at(result, [vocab[token] for token in tokens if token in vocab], 1)
    return result


