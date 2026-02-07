import math
lam = 1.0507009873554804934193349852946
alpha = 1.6732632423543772848170429916717
def selu(x):
    """
    Apply SELU activation to each element.
    """
    # Write code here

    return [lam * a if a > 0 else lam * alpha *(math.exp(a) -1) for a in x]