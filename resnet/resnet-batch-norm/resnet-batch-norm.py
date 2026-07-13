import numpy as np

EPS = 0.00001


def relu(x):
    return np.maximum(x, 0)


def conv(x, W):
    return x @ W


def bn(x, gamma, beta):
    mu = x.mean(axis=0)
    var = x.var(axis=0)
    return gamma * ((x - mu) / np.sqrt(var + EPS)) + beta


def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns: np.ndarray of same shape as input with batch-normalized and skip-connected output
    """
    # YOUR CODE HERE
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    gamma1 = np.array(gamma1)
    beta1 = np.array(beta1)
    gamma2 = np.array(gamma2)
    beta2 = np.array(beta2)

    if mode == "post":
        y = relu(bn(conv(relu(bn(conv(x, W1), gamma1, beta1)), W2), gamma2, beta2) + x)
    else:
        y = conv(relu(bn(conv(relu(bn(x, gamma1, beta1)), W1), gamma2, beta2)), W2) + x
    return {"output": y, "mode": mode}
