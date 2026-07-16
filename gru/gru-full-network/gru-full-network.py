import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def reset_gate(
    h_prev: np.ndarray, x_t: np.ndarray, W_r: np.ndarray, b_r: np.ndarray
) -> np.ndarray:
    """
    Compute reset gate: r_t = sigmoid(W_r @ [h, x] + b_r)
    """
    # YOUR CODE HERE
    stacked = np.concatenate([h_prev, x_t], axis=-1)
    return sigmoid(stacked @ W_r.T + b_r)


def update_gate(
    h_prev: np.ndarray, x_t: np.ndarray, W_z: np.ndarray, b_z: np.ndarray
) -> np.ndarray:
    """
    Compute update gate: z_t = sigmoid(W_z @ [h, x] + b_z)
    """
    # YOUR CODE HERE
    stacked = np.concatenate([h_prev, x_t], axis=-1)

    return sigmoid(stacked @ W_z.T + b_z)


def candidate_hidden(
    h_prev: np.ndarray,
    x_t: np.ndarray,
    r_t: np.ndarray,
    W_h: np.ndarray,
    b_h: np.ndarray,
) -> np.ndarray:
    """
    Compute candidate: h_tilde = tanh(W_h @ [r*h, x] + b_h)
    """
    # reseting
    h_prev = r_t * h_prev

    # stacking
    stacked = np.concatenate([h_prev, x_t], axis=-1)

    return np.tanh(stacked @ W_h.T + b_h)


def hidden_update(
    h_prev: np.ndarray, h_tilde: np.ndarray, z_t: np.ndarray
) -> np.ndarray:
    """
    Compute final state: h_t = z*h_prev + (1-z)*h_tilde
    """
    return z_t * h_prev + (1 - z_t) * h_tilde


def gru_cell(
    x_t: np.ndarray,
    h_prev: np.ndarray,
    W_r: np.ndarray,
    W_z: np.ndarray,
    W_h: np.ndarray,
    b_r: np.ndarray,
    b_z: np.ndarray,
    b_h: np.ndarray,
) -> np.ndarray:
    """
    Complete GRU cell forward pass.
    """
    # computing reset cell
    r_t = reset_gate(h_prev, x_t, W_r, b_r)

    # computing the update gate
    z_t = update_gate(h_prev, x_t, W_z, b_z)

    # candidate cell
    h_tilde = candidate_hidden(h_prev, x_t, r_t, W_h, b_h)

    # combing
    h = hidden_update(h_prev, h_tilde, z_t)

    return h


class GRU:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        self.W_r = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_z = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_h = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_r = np.zeros(hidden_dim)
        self.b_z = np.zeros(hidden_dim)
        self.b_h = np.zeros(hidden_dim)

        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(
            2.0 / (hidden_dim + output_dim)
        )
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> tuple:
        """
        Forward pass. Returns (y, h_last).
        """
        # YOUR CODE HERE
        N, T = X.shape[:2]
        O = self.W_y.shape[0]

        y = np.zeros((N, T, O))
        h = np.zeros((N, self.W_r.shape[0]))

        # processing the sequence (0..T)
        for i in range(T):
            h = gru_cell(
                X[:, i, :],
                h,
                self.W_r,
                self.W_z,
                self.W_h,
                self.b_r,
                self.b_z,
                self.b_h,
            )

            y[:, i, :] = h @ self.W_y.T + self.b_y
        return y, h
