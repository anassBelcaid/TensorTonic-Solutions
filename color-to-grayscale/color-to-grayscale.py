def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    # get the dimension
    N, M = len(image), len(image[0])

    result = []

    for i in range(0,N):
        result.append([])
        for j in range(M):
            R, G, B = image[i][j]
            result[i].append(0.299 *R + 0.587 * G + 0.114 * B)

    return result