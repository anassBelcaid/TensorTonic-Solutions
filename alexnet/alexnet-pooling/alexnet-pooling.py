import numpy as np

def max_pool2d(x: np.ndarray, kernel_size: int = 3, stride: int = 2) -> np.ndarray:
    """Apply 2D max pooling (shape simulation)."""
    # YOUR CODE HERE
    batch_size = x.shape[0]
    H_in = x.shape[1]
    num_channels = x.shape[3]

    H_out = int(np.floor( (H_in - kernel_size)/stride)) + 1
    return np.zeros((batch_size, H_out, H_out, num_channels))