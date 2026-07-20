import numpy as np


def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    points = np.array(points)
    centroids = np.array(centroids)
    points_sq = np.sum(points**2, axis=1)
    centroids_sq = np.sum(centroids**2, axis=1)
    p_dot_c = np.dot(points, centroids.T)
    distances = points_sq[:, None] + centroids_sq[None, :] - 2 * p_dot_c

    return np.argsort(distances, axis=1)[:, 0].tolist()
