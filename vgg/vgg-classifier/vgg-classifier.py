import numpy as np


def vgg_classifier(
    features: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    W3: np.ndarray,
    b3: np.ndarray,
) -> np.ndarray:
    """
    Returns: np.ndarray of shape (B, num_classes) with classification logits
    """
    # flattening the features
    N = features.shape[0]
    features = features.reshape(N, -1)

    # layer 1
    layer1 = np.maximum(features @ W1 + b1, 0)

    # layer 2
    layer2 = np.maximum(layer1 @ W2 + b2, 0)

    return layer2 @ W3 + b3
