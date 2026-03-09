import numpy as np

def dropout(x: np.ndarray, p: float = 0.5, training: bool = True) -> np.ndarray:
    """Apply dropout to input."""
    # YOUR CODE HERE
    if training is False:
        return x
    else:
        mask = np.random.random(len(x)) >= p

        return np.where(mask, x * (1./p), 0)
