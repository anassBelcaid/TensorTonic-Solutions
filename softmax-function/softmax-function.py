import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.array(x)
    if x.size == 0:
        return None
    reduced_logits = x - x.max(axis=-1, keepdims=True)

    #taking exponential
    exp_logits = np.exp(reduced_logits)

    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)