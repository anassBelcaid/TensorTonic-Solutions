import numpy as np


def rnn_cell(
    x_t: np.ndarray,
    h_prev: np.ndarray,
    W_xh: np.ndarray,
    W_hh: np.ndarray,
    b_h: np.ndarray,
) -> np.ndarray:
    """
    Single RNN cell forward pass.
    """
    # YOUR CODE HERE
    return np.tanh(np.dot(x_t, W_xh.T) + np.dot(h_prev, W_hh.T) + b_h)


def rnn_forward(
    X: np.ndarray, h_0: np.ndarray, W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray
) -> tuple:
    """
    Forward pass through entire sequence.
    """
    # YOUR CODE HERE
    B, T, D = X.shape

    # initialize the hidden state
    h = h_0

    # initilaie the space for hiden state
    hidden_size = h_0.shape[1]
    output = np.zeros((B, T, hidden_size))

    for t in range(T):
        xt = X[:, t, :]
        # updating the hidden state
        h = rnn_cell(xt, h, W_xh, W_hh, b_h)

        # saving in output
        output[:, t, :] = h

    return output, h

