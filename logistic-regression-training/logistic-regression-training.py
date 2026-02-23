import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here

    X = np.array(X)
    y = np.array(y)
    
    K = X.shape[1]


    # initialization of the weights
    W, b = np.zeros(K), 0


    # Training
    for i in range(steps):
        # computing the output
        o = X @ W + b  # size is (n, 1)

        # applying the non linearity
        p = _sigmoid(o)

        # Computing the binary cross loss
        L = - (y * np.log(p)  + (1-y) * np.log(1-p)).mean()

        # computing the gradient of the loss 
        w_grad = (X.T @ (p - y)).mean() # K, n * n, K

        b_grad = (p - y).mean()

        # upading the parameters
        W -= lr * w_grad
        b -= lr * b_grad



    return W, b

