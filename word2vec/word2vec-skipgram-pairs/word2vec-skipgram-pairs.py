import torch


def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    """
    Returns int64 torch.Tensor of shape (num_pairs, 2).
    """
    # YOUR CODE HERE
    pairs = []
    N = len(token_ids)
    for i in range(0, N):
        l = max(0, i - window)
        for j in range(l, i):
            pairs.append((token_ids[i], token_ids[j]))
        r = min(N, i + window + 1)
        for j in range(i + 1, r):
            pairs.append((token_ids[i], token_ids[j]))
    if len(pairs) == 0:
        return torch.empty((0, 2), dtype = torch.int64)
    return torch.tensor(pairs, dtype = torch.int64)






