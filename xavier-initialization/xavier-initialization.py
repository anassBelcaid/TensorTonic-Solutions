import numpy as np
def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    W = np.asarray(W)

    L = np.sqrt(6 / (fan_in+fan_out) )
    return (2 * W * L - L).tolist()
    
