import math

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    d_model = Q.shape[-1]

    S = Q @ K.transpose(-1, -2)
    S_scaled = S / math.sqrt(d_model)

    return F.softmax(S_scaled, dim=-1) @ V

