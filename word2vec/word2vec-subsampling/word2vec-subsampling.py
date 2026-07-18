import torch


def subsample_keep_probs(counts: torch.Tensor, t: float = 1e-5) -> torch.Tensor:
    """
    Returns torch.Tensor of shape (vocab_size,) with the keep-probability for each word.
    """
    # computing the probability
    f_w = counts / counts.sum()

    return torch.clamp(torch.sqrt(t / f_w), 0, 1)



