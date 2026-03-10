import numpy as np

def local_response_normalization(x: np.ndarray, k: float = 2, n: int = 5,
                                  alpha: float = 1e-4, beta: float = 0.75) -> np.ndarray:
    """Apply Local Response Normalization across channels."""
    # YOUR CODE HERE
    b = x.copy()


    num_channels = x.shape[3]

    for c in range(num_channels):
        lower = max(0, c - num_channels//2)
        upper = min(num_channels-1, c + num_channels//2) + 1
        b [:,:,:, c] /= k + alpha  * ((x[:,:,:,lower:upper]**2).sum(axis=3))**beta


    return b
