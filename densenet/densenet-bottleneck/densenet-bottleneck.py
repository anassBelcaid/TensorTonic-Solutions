import torch
import torch.nn.functional as F


def batch_norm(x, gamma, beta, mean, var, eps=1e-5):
    return (
        gamma[None, :, None, None]
        * ((x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + eps))
        + beta[None, :, None, None]
    )


def bottleneck_layer(
    x,
    bn1_gamma,
    bn1_beta,
    bn1_mean,
    bn1_var,
    conv1_weight,
    bn2_gamma,
    bn2_beta,
    bn2_mean,
    bn2_var,
    conv2_weight,
    eps=1e-5,
):
    """
    Returns torch.Tensor of shape (N, growth_rate, H, W) after the two-stage bottleneck composite.
    """
    dtype = torch.float64

    x = torch.as_tensor(x, dtype=dtype)
    bn1_gamma = torch.as_tensor(bn1_gamma, dtype=dtype)
    bn1_beta = torch.as_tensor(bn1_beta, dtype=dtype)
    bn1_mean = torch.as_tensor(bn1_mean, dtype=dtype)
    bn1_var = torch.as_tensor(bn1_var, dtype=dtype)
    conv1_weight = torch.as_tensor(conv1_weight, dtype=dtype)

    bn2_gamma = torch.as_tensor(bn2_gamma, dtype=dtype)
    bn2_beta = torch.as_tensor(bn2_beta, dtype=dtype)
    bn2_mean = torch.as_tensor(bn2_mean, dtype=dtype)
    bn2_var = torch.as_tensor(bn2_var, dtype=dtype)
    conv2_weight = torch.as_tensor(conv2_weight, dtype=dtype)

    # Rest of your implementation...
    # computing the first block
    bn1 = batch_norm(x, bn1_gamma, bn1_beta, bn1_mean, bn1_var, eps)
    rl1 = F.relu(bn1)
    y1 = F.conv2d(rl1, conv1_weight, padding=0, bias=None)

    # computing the second block
    bn2 = batch_norm(y1, bn2_gamma, bn2_beta, bn2_mean, bn2_var, eps)
    rl2 = F.relu(bn2)
    return F.conv2d(rl2, conv2_weight, padding=1, bias=None)

