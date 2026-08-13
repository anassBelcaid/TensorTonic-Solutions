import numpy as np


def build_token_mappings(tokens):
    """Create mappings for converting tokens to indices and back."""
    vocabulary = list(dict.fromkeys(tokens))
    token_to_idx = {token: idx for idx, token in enumerate(vocabulary)}
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}
    return token_to_idx, idx_to_token


def bigram_probabilities(tokens):
    """
    Returns: (counts, probs)
      counts: dict mapping (w1, w2) -> integer count
      probs: dict mapping (w1, w2) -> float P(w2 | w1) with add-1 smoothing
    """
    token_to_idx, idx_to_token = build_token_mappings(tokens)
    vocabulary_size = len(token_to_idx)

    if vocabulary_size == 0:
        return {}, {}

    # Perform the bigram calculations with indices.
    count_matrix = np.zeros((vocabulary_size, vocabulary_size), dtype=int)
    token_indices = [token_to_idx[token] for token in tokens]

    if len(token_indices) >= 2:
        for a, b in np.lib.stride_tricks.sliding_window_view(token_indices, 2):
            count_matrix[a, b] += 1

    # Add-1 smoothing and normalization over possible following tokens.
    probability_matrix = (count_matrix + 1) / (
        count_matrix.sum(axis=1, keepdims=True) + vocabulary_size
    )

    # Convert the matrix indices back to the token-pair keys requested above.
    counts = {
        (idx_to_token[i], idx_to_token[j]): int(count_matrix[i, j])
        for i in range(vocabulary_size)
        for j in range(vocabulary_size)
        if count_matrix[i, j] > 0
    }
    probs = {
        (idx_to_token[i], idx_to_token[j]): float(probability_matrix[i, j])
        for i in range(vocabulary_size)
        for j in range(vocabulary_size)
    }

    return counts, probs
