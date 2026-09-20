import numpy as np


def generator(z: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Generate a batch of samples using an affine layer followed by tanh."""
    return np.tanh(z @ W + b)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply the logistic sigmoid element-wise."""
    return 1.0 / (1.0 + np.exp(-x))


def discriminator(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Return the discriminator probability for every sample in a batch."""
    return sigmoid(x @ W)


def gan_losses(real_probs: np.ndarray, fake_probs: np.ndarray) -> dict:
    """Return the discriminator and non-saturating generator losses."""
    eps = 1e-8
    real_probs = np.clip(real_probs, eps, 1.0 - eps)
    fake_probs = np.clip(fake_probs, eps, 1.0 - eps)
    batch_size = len(real_probs)

    discriminator_loss = -(
        np.log(real_probs).sum() + np.log(1.0 - fake_probs).sum()
    ) / batch_size
    generator_loss = -np.log(fake_probs).sum() / batch_size

    return {
        "discriminator_loss": float(discriminator_loss),
        "generator_loss": float(generator_loss),
    }


def gan_forward(
    z: np.ndarray,
    real_data: np.ndarray,
    G_W: np.ndarray,
    G_b: np.ndarray,
    D_W: np.ndarray,
) -> dict:
    """
    Returns generated samples, probabilities, and both GAN losses.
    """
    z = np.asarray(z, dtype=float)
    real_data = np.asarray(real_data, dtype=float)
    G_W = np.asarray(G_W, dtype=float)
    G_b = np.asarray(G_b, dtype=float)
    D_W = np.asarray(D_W, dtype=float)

    generated_samples = generator(z, G_W, G_b)
    real_probabilities = discriminator(real_data, D_W)
    fake_probabilities = discriminator(generated_samples, D_W)
    losses = gan_losses(real_probabilities, fake_probabilities)

    return {
        "generated_samples": generated_samples,
        "real_probabilities": real_probabilities,
        "fake_probabilities": fake_probabilities,
        **losses,
    }


z = np.array([[1, -0.5], [0.3, 0.8]])
real_data = np.array([[0.7, -0.2, 0.4], [0.1, 0.9, -0.3]])
G_W = np.array([[0.3, -0.1, 0.5], [0.2, 0.4, -0.3]])
G_b = np.array([0.1, 0, -0.1])
D_W = np.array([[0.5], [-0.2], [0.3]])

Output: {
    "generated_samples": [
        [0.291313, -0.291313, 0.50052],
        [0.336376, 0.282135, -0.187746],
    ],
    "real_probabilities": [[0.624806], [0.445221]],
    "fake_probabilities": [[0.587605], [0.513856]],
    "discriminator_loss": 1.443261,
    "generator_loss": 0.598756,
}

# Explanation: The generator produces a fake batch, the shared discriminator scores both batches, and the two losses use those exact scores.
