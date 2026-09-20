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


# Explanation: The discriminator loss rewards high real probabilities and low fake probabilities, while the non-saturating generator loss rewards high fake probabilities.

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def train_discriminator_step(
    real_data: np.ndarray,
    fake_data: np.ndarray,
    D_W: np.ndarray,
    learning_rate: float,
) -> dict:
    """
    Returns updated discriminator weights and the pre-update loss.
    """
    real_data = np.asarray(real_data, dtype=float)
    fake_data = np.asarray(fake_data, dtype=float)
    D_W = np.asarray(D_W, dtype=float)

    N = len(real_data)
    real_prob = sigmoid(real_data @ D_W)
    fake_prob = sigmoid(fake_data @ D_W)
    grad_dis = 1.0 / N * (
        real_data.T @ (1.0 - real_prob) - fake_data.T @ fake_prob
    )
    discriminator_loss = gan_losses(real_prob, fake_prob)["discriminator_loss"]

    return {
        "new_discriminator_weights": D_W + learning_rate * grad_dis,
        "discriminator_loss": discriminator_loss,
    }


