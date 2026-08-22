import numpy as np


def edit_distance(s1, s2):
    """
    Compute the minimum edit distance between two strings.
    """
    n1 = len(s1)
    n2 = len(s2)

    dp = np.zeros((n1 + 1, n2 + 1))

    # initialisation
    dp[0] = np.arange(n2 + 1)
    dp[:, 0] = np.arange(n1 + 1)

    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            diagonal = dp[i - 1][j - 1] + 1
            if s1[i - 1] == s2[j - 1]:
                diagonal -= 1
            dp[i, j] = min(diagonal, 1 + dp[i - 1][j], 1 + dp[i][j - 1])

    return int(dp[n1, n2])
