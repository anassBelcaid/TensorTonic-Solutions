import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    dimension is (batch, seq_len, d_v)
    """
    # Your code here
    
    # compute the score matrix
    d = Q.shape[2]

    # computing the scaled matrix
    scaled_scores = Q @ K.transpose(-2, -1)  / math.sqrt(d)

    # computing softmax
    attn = F.softmax(scaled_scores, dim = -1)
    return attn @ V

