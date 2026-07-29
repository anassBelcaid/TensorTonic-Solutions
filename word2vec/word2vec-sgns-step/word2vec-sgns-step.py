import torch


def sgns_sgd_step(
    W_in: torch.Tensor,
    W_out: torch.Tensor,
    center_id: int,
    pos_id: int,
    neg_ids: torch.Tensor,
    lr: float,
) -> tuple:
    """
    Returns tuple (W_in_updated, W_out_updated), each the same shape as the inputs, after one SGNS SGD step.
    """

    # let's compute the center vector
    vc = W_in[center_id].clone()

    # taking the negative vectors
    U_p = W_out[pos_id].clone()
    U_neg = W_out[neg_ids].clone()

    # computing the scores
    sp = U_p @ vc
    sigmoid_sp = torch.sigmoid(sp)

    # computing the negative scores
    sn = U_neg @ vc
    sigmoid_sn = torch.sigmoid(sn)

    # computing W_in gradient
    grad_in = (sigmoid_sp - 1) * U_p + torch.sum(sigmoid_sn[:, None] * U_neg, dim=0)

    grad_pos = (sigmoid_sp - 1) * vc
    grad_neg = sigmoid_sn[:, None] * vc

    # Work on copies and disable autograd tracking for this manual SGD update.
    with torch.no_grad():
        W_in_updated = W_in.clone()
        W_out_updated = W_out.clone()
        W_in_updated[center_id] -= lr * grad_in
        W_out_updated[pos_id] -= lr * grad_pos

        # Accumulate updates when the same negative ID occurs more than once.
        W_out_updated.index_add_(0, neg_ids, -lr * grad_neg)

    return W_in_updated, W_out_updated



