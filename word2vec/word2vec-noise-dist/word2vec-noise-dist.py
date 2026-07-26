import torch


def noise_distribution(counts: torch.Tensor, alpha: float = 0.75) -> torch.Tensor:
    """
    Returns torch.Tensor of shape (vocab_size,), a probability distribution that sums to 1.
    """
    # YOUR CODE HERE
    # raising counts to the power of alpha
    counts = torch.tensor(counts, dtype=torch.float64)

    counts = counts**alpha

    counts = counts / torch.sum(counts)

    return counts

