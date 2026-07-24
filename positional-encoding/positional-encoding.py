import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    encoding = np.empty((seq_len, d_model))

    pos = np.arange(seq_len)[:, None]

    even = np.arange(0, d_model, 2)
    div = base ** (even / d_model)

    encoding[:, 0::2] = np.sin(pos / div)
    encoding[:, 1::2] = np.cos(pos / div[: d_model // 2])

    return encoding
