import numpy as np


def vgg_conv_block(x: np.ndarray, weights: list, biases: list) -> np.ndarray:
    """Apply convolution weights followed by ReLU."""
    out = x.copy()
    for W, b in zip(weights, biases):
        out = np.maximum(0, out @ W + b)
    return out


def maxpool_2x2(x):
    B, H, W, C = x.shape
    return x.reshape(B, H // 2, 2, W // 2, 2, C).max(axis=(2, 4))


def vgg_features(
    x: np.ndarray, config: list, conv_weights: list, conv_biases: list
) -> np.ndarray:
    """
    Returns: np.ndarray feature tensor after applying conv layers and max pooling
    """
    out = np.asarray(x)
    conv_index = 0

    for layer in config:
        if isinstance(layer, int):
            out = vgg_conv_block(
                out,
                [conv_weights[conv_index]],
                [conv_biases[conv_index]],
            )
            conv_index += 1
        else:
            out = maxpool_2x2(out)

    return out
