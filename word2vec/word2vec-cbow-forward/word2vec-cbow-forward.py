import torch
import torch.nn.functional as F


def cbow_forward(
    context_ids: torch.Tensor, target_id: int, W_in: torch.Tensor, W_out: torch.Tensor
) -> torch.Tensor:
    """
    Returns a scalar torch.Tensor: the CBOW cross-entropy loss for predicting target_id from the averaged context.
    """
    # taking the mean of the embeding of the BOW ( Bag Of Words)
    h = W_in[context_ids].mean(dim=0)

    # computing the logits
    logits = W_out @ h

    return -torch.log(F.softmax(logits, dim=0)[target_id])

