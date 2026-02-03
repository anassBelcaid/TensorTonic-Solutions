import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # Write code here
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return None
    result = 0

    for i in range(A.shape[0]):
        result += A[i][i]
    
    return result
