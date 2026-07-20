import numpy as np


def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    points = np.array(points)
    assignments = np.array(assignments)
    N, D = points.shape

    centroids = np.zeros((k, D))

    for center in range(k):
        mask = assignments == center
        centroids[center] = np.mean(points[mask], axis=0)
    return centroids.tolist()
