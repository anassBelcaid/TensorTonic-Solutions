import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    # Write code here
    matrix = np.array(matrix)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    if matrix.shape[1] == 0:
        return 0

    U, _ = np.linalg.eig(matrix)
    np.lexsort(U)
    return U


