import math

import numpy as np


def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def layer_norm(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """
    Returns: Normalized array of same shape as x
    """
    # Your code here
    mean = np.mean(x, axis=-1, keepdims=True)
    sigma = np.mean((x - mean) ** 2, axis=-1, keepdims=True)

    centered = (x - mean) / np.sqrt(sigma + eps)

    return gamma * centered + beta


def multi_head_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
    W_o: np.ndarray,
    num_heads: int,
) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    Q = np.dot(Q, W_q)
    K = np.dot(K, W_k)
    V = np.dot(V, W_v)

    # now we compute the attension value for each head
    d_head = Q.shape[-1] // num_heads

    heads = []

    for i in range(num_heads):
        L = i * d_head
        R = (i + 1) * d_head
        Q_head = Q[:, :, L:R]
        K_head = K[:, :, i * d_head : (i + 1) * d_head]
        V_head = V[:, :, i * d_head : (i + 1) * d_head]

        S = Q_head @ K_head.transpose(0, 2, 1)
        Scaled = S / math.sqrt(d_head)

        head = softmax(Scaled) @ V_head
        heads.append(head)
    return np.concatenate(heads, axis=2) @ W_o


def feed_forward(
    x: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray
) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    # Your code here
    h1 = np.maximum(x @ W1 + b1, 0)

    return h1 @ W2 + b2


def encoder_block(
    x: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
    W_o: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    gamma1: np.ndarray,
    beta1: np.ndarray,
    gamma2: np.ndarray,
    beta2: np.ndarray,
    num_heads: int,
) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # computing the MHA
    mha = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)

    # add and normalize
    sub1 = layer_norm(x + mha, gamma1, beta1)

    # sublayer 2
    fnn = feed_forward(sub1, W1, b1, W2, b2)

    output = layer_norm(sub1 + fnn, gamma2, beta2)
    return output
