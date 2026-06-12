import numpy as np


def layer_norm(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """
    Returns: Normalized array of same shape as x
    """
    # Your code here
    mean = np.mean(x, axis=-1, keepdims=True)
    sigma = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    print(mean, sigma)

    centered = (x - mean) / np.sqrt(sigma + eps)

    return gamma * centered + beta
