import numpy as np


def relu(x):
    return np.maximum(x, 0)


def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    x = np.array(x)
    conv1 = np.array(conv1)
    W1_b1 = np.array(W1_b1)
    W2_b1 = np.array(W2_b1)
    W1_b2 = np.array(W1_b2)
    W2_b2 = np.array(W2_b2)
    Ws_b2 = np.array(Ws_b2)
    fc = np.array(fc)

    # Stem: conv1 + ReLU
    c = relu(x @ conv1)

    # Block 1: identity skip (dims match, 2 -> 2)
    b1 = relu(relu(c @ W1_b1) @ W2_b1 + c)

    # Block 2: projection skip (dims change, 2 -> 3)
    shortcut2 = b1 @ Ws_b2
    b2 = relu(relu(b1 @ W1_b2) @ W2_b2 + shortcut2)

    # FC head: 3 -> num_classes
    return b2 @ fc
