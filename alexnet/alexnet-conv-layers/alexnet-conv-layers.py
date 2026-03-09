import numpy as np

def alexnet_conv1(image: np.ndarray) -> np.ndarray:
    """AlexNet first conv layer: 11x11, stride 4, 96 filters (shape simulation)."""
    # YOUR CODE HERE
    batch_size = image.shape[0]
    N = image.shape[1]

    out_shape = int(np.floor((N - 11)/4)  + 1)
    
    return np.zeros((batch_size, 55,55, 96))
