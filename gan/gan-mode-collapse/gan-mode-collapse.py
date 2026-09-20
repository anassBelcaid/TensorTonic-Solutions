import numpy as np


def detect_mode_collapse(generated_samples: np.ndarray, threshold: float = 0.1) -> dict:
    """
    Returns diversity_score and is_collapsed in a dictionary.
    """
    S = float(np.mean(np.std(generated_samples, axis=0, ddof=0)))
    return {"diversity_score": S, "is_collapsed": S < threshold}
