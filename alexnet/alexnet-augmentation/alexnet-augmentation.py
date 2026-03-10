import numpy as np

def random_crop(image: np.ndarray, crop_size: int = 224) -> np.ndarray:
    """Extract a random crop from the image."""
    # YOUR CODE HERE
    n  = image.shape[0]
    i, j = np.random.choice(range(n - crop_size), size = 2 )

    return image[i:i+crop_size, j : j + crop_size]



def random_horizontal_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Randomly flip image horizontally."""
    # YOUR CODE HERE
    return image if np.random.random() <= p else image[::-1,:]

