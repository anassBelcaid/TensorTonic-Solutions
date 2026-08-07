def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    # the function is a^2 x + bx + c
    """
    # Write code here
    for _ in range(steps):
        grad = 2 * a * x0 + b

        x0 -= lr * grad
    return x0
