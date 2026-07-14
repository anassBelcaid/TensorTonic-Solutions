import torch
import torch.nn.functional as F


def batch_norm(x, gamma, beta, mean, var, eps=1e-5):
    return (
        gamma[None, :, None, None]
        * ((x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + eps))
        + beta[None, :, None, None]
    )


def transition_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, out_channels, H//2, W//2) after BN-ReLU-1x1Conv then 2x2 average pooling.
    """
    # YOUR CODE HERE
    dtype = torch.float64
    x = torch.tensor(x)
    bn_gamma = torch.tensor(bn_gamma)
    bn_beta = torch.tensor(bn_beta)
    bn_mean = torch.tensor(bn_mean)
    bn_var = torch.tensor(bn_var)
    conv_weight = torch.tensor(conv_weight)

    # Bn layer
    x = batch_norm(x, bn_gamma, bn_beta, bn_mean, bn_var, eps)

    # Relu
    x = F.relu(x)

    # conv 1x1
    x = F.conv2d(x, conv_weight, bias=None)

    # Averating pooling
    x = F.avg_pool2d(x, 2)

    return x
