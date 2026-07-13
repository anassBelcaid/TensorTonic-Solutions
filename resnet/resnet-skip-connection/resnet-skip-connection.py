import numpy as np


def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITH skip connections.
    Gradient at layer l = sum of paths through network
    """
    # YOUR CODE HERE
    x = np.array(x).astype("float")
    gradients_F = np.array(gradients_F)

    n = len(x)
    I = np.eye(n)

    for J in gradients_F:
        x = x @ (J + I)
    return x


def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITHOUT skip connections.
    """
    # YOUR CODE HERE
    x = np.array(x).astype("float")
    gradients_F = np.array(gradients_F)
    n = len(x)
    for J in gradients_F:
        x = x @ J

    return x



