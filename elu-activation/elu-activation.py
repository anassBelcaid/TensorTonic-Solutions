import math
def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    # Write code here
    return [a  if a > 0 else alpha *(math.exp(a) - 1) for a in x]