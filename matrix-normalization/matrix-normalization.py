import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    try:
        matrix = np.array(matrix, dtype=float)
        # Must be 2D and non-empty
        if matrix.ndim != 2 or matrix.size == 0:
            return None
        # Axis validation
        if axis not in (None, 0, 1):
            return None
        
        # axis=None → flatten
        if axis is None:
            flat = matrix.ravel()
            if norm_type == 'l1':
                norm = np.sum(np.abs(flat))
            elif norm_type == 'l2':
                norm = np.sqrt(np.sum(flat ** 2))
            elif norm_type == 'max':
                norm = np.max(np.abs(flat))
            else:
                return None
            if norm == 0:
                return None
            return matrix / norm
        
        # axis = 0 or 1
        if norm_type == 'l1':
            norm = np.sum(np.abs(matrix), axis=axis, keepdims=True)
        elif norm_type == 'l2':
            norm = np.sqrt(np.sum(matrix ** 2, axis=axis, keepdims=True))
        elif norm_type == 'max':
            norm = np.max(np.abs(matrix), axis=axis, keepdims=True)
        else:
            return None
        
        # Handle zero norms: replace with 1 to avoid division by zero
        # This keeps zero vectors as zero
        norm = np.where(norm == 0, 1, norm)
        return matrix / norm
        
    except Exception:
        return None
