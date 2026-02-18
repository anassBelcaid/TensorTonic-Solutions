def image_histogram(image):
    """
    Compute the intensity histogram of a grayscale image.
    """
    # Write code here
    result = [0] * 256

    for row in image:
        for v in row:
            result[v] += 1;

    return result