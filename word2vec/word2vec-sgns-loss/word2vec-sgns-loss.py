import torch
import torch.nn.functional as F


def sgns_loss(
    center_vec: torch.Tensor, pos_vec: torch.Tensor, neg_vecs: torch.Tensor
) -> torch.Tensor:
    """
    Returns a scalar torch.Tensor: the SGNS loss.
    """
    center_vec = torch.tensor(center_vec)
    pos_vec = torch.tensor(pos_vec)
    neg_vecs = torch.tensor(neg_vecs)

    pos_dot = pos_vec @ center_vec
    neg_dot = neg_vecs @ center_vec

    return F.softplus(-pos_dot) + F.softplus(neg_dot).sum()
