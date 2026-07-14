import torch
import torch.nn.functional as F


def batch_norm(x, gamma, beta, mean, var, eps=1e-5):
    return (
        gamma[None, :, None, None]
        * ((x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + eps))
        + beta[None, :, None, None]
    )


def dense_block(x, layers, growth_rate, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, C + L*growth_rate, H, W).
    """
    # YOUR CODE HERE
    dtype = torch.float64
    x = torch.tensor(x, dtype=dtype)

    # moving through the layers
    for layer in layers:
        bn_gamma = torch.tensor(layer["bn_gamma"], dtype=dtype)
        bn_beta = torch.tensor(layer["bn_beta"], dtype=dtype)
        bn_mean = torch.tensor(layer["bn_mean"], dtype=dtype)
        bn_var = torch.tensor(layer["bn_var"], dtype=dtype)
        conv_weight = torch.tensor(layer["conv_weight"], dtype=dtype)

        # apply the conv ---> relu ---> bn block
        rl = F.relu(batch_norm(x, bn_gamma, bn_beta, bn_mean, bn_var, eps))

        o = F.conv2d(rl, conv_weight, padding=1, bias=None)

        # Stacking along channels
        x = torch.cat([x, o], dim=1)
    return x


