import torch
import torch.nn.functional as F


def composite_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, growth_rate, H, W): BN, ReLU, then a 3x3 same-padding convolution.
    """
    # YOUR CODE HERE
    # BN Step
    dtype = torch.float64

    x = torch.tensor(x, dtype=dtype)
    bn_gamma = torch.tensor(bn_gamma, dtype=dtype)
    bn_beta = torch.tensor(bn_beta, dtype=dtype)
    bn_mean = torch.tensor(bn_mean, dtype=dtype)
    bn_var = torch.tensor(bn_var, dtype=dtype)
    conv_weight = torch.tensor(conv_weight, dtype=dtype)

    bn = (
        bn_gamma[None, :, None, None]
        * (
            (x - bn_mean[None, :, None, None])
            / torch.sqrt(bn_var[None, :, None, None] + eps)
        )
        + bn_beta[None, :, None, None]
    )
    rl = F.relu(bn)

    return F.conv2d(rl, conv_weight, padding=1, bias=None, stride=1)
