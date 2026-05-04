import numpy as np


class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim

        # Xavier initialization
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(
            2.0 / (input_dim + hidden_dim)
        )
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(
            2.0 / (2 * hidden_dim)
        )
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(
            2.0 / (hidden_dim + output_dim)
        )
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray, h_0: np.ndarray = None) -> tuple:
        """
        Forward pass through entire sequence.
        Returns (y_seq, h_final).
        """
        # YOUR CODE HERE
        B, T, D = X.shape

        # Creating the output
        y_t = np.zeros((B, T, self.W_hy.shape[0]))
        h = h_0

        # looop over the sequence of time
        for t in range(T):
            x = X[:, t, :]

            # computing the new hidden state
            h = np.tanh(x @ self.W_xh.T + h @ self.W_hh.T + self.b_h)

            # computing the output
            y_t[:, t, :] = h @ self.W_hy.T + self.b_y

        return y_t, h


