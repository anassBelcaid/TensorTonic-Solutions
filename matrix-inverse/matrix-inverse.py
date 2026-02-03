
import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    # Validate input is 2D ans square
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return None

    # check for singular matrix
    if np.isclose(np.linalg.det(A), 1E-10):
        return None

    return np.linalg.inv(A)
