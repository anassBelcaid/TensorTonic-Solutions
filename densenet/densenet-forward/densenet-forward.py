import torch
import torch.nn.functional as F


def batch_norm(x, gamma, beta, mean, var, eps=1e-5):
    return (
        gamma[None, :, None, None]
        * ((x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + eps))
        + beta[None, :, None, None]
    )


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


def transition_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, out_channels, H//2, W//2) after BN-ReLU-1x1Conv then 2x2 average pooling.
    """
    # YOUR CODE HERE
    dtype = torch.float64
    x = torch.tensor(x, dtype=dtype)
    bn_gamma = torch.tensor(bn_gamma, dtype=dtype)
    bn_beta = torch.tensor(bn_beta, dtype=dtype)
    bn_mean = torch.tensor(bn_mean, dtype=dtype)
    bn_var = torch.tensor(bn_var, dtype=dtype)
    conv_weight = torch.tensor(conv_weight, dtype=dtype)

    # Bn layer
    x = batch_norm(x, bn_gamma, bn_beta, bn_mean, bn_var, eps)

    # Relu
    x = F.relu(x)

    # conv 1x1
    x = F.conv2d(x, conv_weight, bias=None)

    # Averating pooling
    x = F.avg_pool2d(x, 2)

    return x


def densenet_forward(x, weights, growth_rate, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, num_classes) with class logits.
    """
    dtype = torch.float64
    x = torch.tensor(x, dtype=dtype)

    # Step block (3, 3) to get the desired initial channels
    stem_weight = torch.tensor(weights["stem_conv"], dtype=dtype)
    x = F.conv2d(x, stem_weight, padding=1)

    # Step 2 : block ---> transition
    blocks = weights["blocks"]
    transitions = weights["transitions"]

    for block, transition in zip(blocks, transitions):
        print(transition)
        x = dense_block(x, block, growth_rate, eps)
        x = transition_layer(x, **transition, eps=eps)

    # final block which don't need a transition
    x = dense_block(x, blocks[-1], growth_rate, eps)

    # 'fc_bias', 'fc_weight',  'final_bn_var', 'final_bn_beta', 'final_bn_mean'
    # BN layer
    # BN layer
    final_bn_gamma = torch.tensor(weights["final_bn_gamma"], dtype=dtype)
    final_bn_beta = torch.tensor(weights["final_bn_beta"], dtype=dtype)
    final_bn_mean = torch.tensor(weights["final_bn_mean"], dtype=dtype)
    final_bn_var = torch.tensor(weights["final_bn_var"], dtype=dtype)

    x = batch_norm(
        x,
        final_bn_gamma,
        final_bn_beta,
        final_bn_mean,
        final_bn_var,
        eps,
    )
    # relu
    x = F.relu(x)
    # average global poooling
    x = torch.mean(x, dim=(2, 3))

    # Final FC layer
    W = torch.tensor(weights["fc_weight"], dtype=dtype)
    b = torch.tensor(weights["fc_bias"], dtype=dtype)

    return x @ W.T + b

