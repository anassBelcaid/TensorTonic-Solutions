import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x, p = np.array(x), np.array(p)
    if not np.isclose(p.sum(), 1):
        raise ValueError("probabilities don't sum to one")
    if x.size != p.size:
        return None

    return np.sum(p * x)
