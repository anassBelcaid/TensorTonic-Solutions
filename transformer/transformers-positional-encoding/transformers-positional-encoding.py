import numpy as np


def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    PE = np.zeros((seq_length, d_model))

    pos = np.arange(seq_length)
    for i in range(0, d_model, 2):
        PE[:, i] = np.sin(pos / 10000 ** (i / d_model))
    for i in range(1, d_model, 2):
        PE[:, i] = np.cos(pos / 10000 ** ((i - 1) / d_model))

    return PE

