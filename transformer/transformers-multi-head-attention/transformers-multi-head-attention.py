import math

import numpy as np


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


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
    # Your code here
    # projecting the Q, K , V
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
