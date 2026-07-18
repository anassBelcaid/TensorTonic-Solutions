def interpolate_missing(j, left, right):
    """
    Function to compute the missing value at j given
    (left = (left_indice,left_value) and right = (right_indice, right_value))
    """

    l_idx, l_value = left
    r_idx, r_value = right
    return l_value + ((j - l_idx) / (r_idx - l_idx)) * (r_value - l_value)


def linear_interpolation(values):
    """
    Fill missing (None) values using linear interpolation.
    """
    # Write code here
    n = len(values)
    result = values.copy()

    missing = []  # indices of missing values that need to be filled
    left = (0, values[0])

    for j in range(1, n):
        value = values[j]
        if value is None:
            missing.append(j)
        else:
            # need to update the in between values
            right = (j, value)
            for i in missing:
                result[i] = interpolate_missing(i, left, right)
            missing.clear()
            left = right

    return result
