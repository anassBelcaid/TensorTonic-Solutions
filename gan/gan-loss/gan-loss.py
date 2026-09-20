import numpy as np


def gan_losses(real_probs: np.ndarray, fake_probs: np.ndarray) -> dict:
    """
    Returns discriminator_loss and generator_loss as Python floats.
    """
    # N = len(real_probs) + len(fake_probs)
    EPS = 1e-8
    real_probs = np.clip(real_probs, EPS, 1 - EPS)
    fake_probs = np.clip(fake_probs, EPS, 1 - EPS)
    N = len(real_probs)
    dis_loss = -1 / N * (np.log(real_probs).sum() + np.log(1 - fake_probs).sum())

    # generator loss
    gen_loss = -1.0 / N * (np.log(fake_probs).sum())

    return {"discriminator_loss": float(dis_loss), "generator_loss": float(gen_loss)}
