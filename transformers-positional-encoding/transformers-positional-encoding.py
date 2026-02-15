import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    encoding = np.zeros((seq_length, d_model))

    # Creating the i encoding
    indices = np.arange(d_model)

    rows = np.arange(seq_length)

    # computing the denominator
    denominator = np.exp(indices[::2]/ d_model * np.log(10000))

    # computing even values 
    encoding[:, ::2] = np.sin(rows[:, np.newaxis] * denominator[np.newaxis, :] )

    # computing odd values
    encoding[:, 1::2] = np.cos(rows[:, np.newaxis] * denominator[np.newaxis, :] )

    return encoding


