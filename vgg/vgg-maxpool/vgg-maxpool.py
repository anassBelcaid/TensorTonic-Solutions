import numpy as np


def vgg_maxpool(x: np.ndarray) -> np.ndarray:
    """
    Implement VGG-style max pooling (2x2, stride 2).
    """
    # Your implementation here
    x = np.asarray(x)
    B, H, W, C = x.shape

    result = np.zeros((B, H // 2, W // 2, C))

    for i in range(H // 2):
        for j in range(W // 2):
            result[:, i, j, :] = np.max(
                x[:, 2 * i : 2 * (i + 1), 2 * j : 2 * (j + 1)], axis=(1, 2)
            )
    return result
