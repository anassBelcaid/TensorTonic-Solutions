import numpy as np


def lstm_cell(
    x_t: np.ndarray,
    h_prev: np.ndarray,
    C_prev: np.ndarray,
    W_f: np.ndarray,
    W_i: np.ndarray,
    W_c: np.ndarray,
    W_o: np.ndarray,
    b_f: np.ndarray,
    b_i: np.ndarray,
    b_c: np.ndarray,
    b_o: np.ndarray,
) -> tuple:
    """Complete LSTM cell forward pass."""
    # YOUR CODE HERE
    # compute the stacked vector
    s = np.concatenate((h_prev, x_t), axis=-1)

    # computing the forget get
    f_t = sigmoid(s @ W_f.T + b_f)

    # computing the input gate
    i_t = sigmoid(s @ W_i.T + b_i)

    # computing the candidate memory
    c_tilde = np.tanh(s @ W_c.T + b_c)

    # computing the output
    o_t = sigmoid(s @ W_o.T + b_o)

    # computing the new cell state
    c_t = f_t * C_prev + i_t * c_tilde

    # updating the new hidden state
    h_t = o_t * np.tanh(c_t)

    return h_t, c_t


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class LSTM:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_f = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_i = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_c = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_o = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_f = np.zeros(hidden_dim)
        self.b_i = np.zeros(hidden_dim)
        self.b_c = np.zeros(hidden_dim)
        self.b_o = np.zeros(hidden_dim)
        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(
            2.0 / (hidden_dim + output_dim)
        )
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> tuple:
        """
        Forward pass.
        X shape: (batch, seq_len, input_dim)
        Returns (yt, h_last, C_last)
          yt     shape: (batch, seq_len, output_dim)
          h_last shape: (hidden_dim,)
          C_last shape: (hidden_dim,)
        """
        B, T, D = X.shape
        yt = np.zeros((B, T, self.W_y.shape[0]))

        # initializing the hidden state to zero
        h = np.zeros((B, self.hidden_dim))

        # initializing the cell state to zero
        C = np.zeros((B, self.hidden_dim))

        # unrolling the sequences
        for i in range(T):
            # extacting x_t
            x_t = X[:, i, :]

            # calling the lstm forward
            h, C = lstm_cell(
                x_t,
                h,
                C,
                self.W_f,
                self.W_i,
                self.W_c,
                self.W_o,
                self.b_f,
                self.b_i,
                self.b_c,
                self.b_o,
            )

            # computing the output
            yt[:, i, :] = h @ self.W_y.T + self.b_y

        return yt, h, C
