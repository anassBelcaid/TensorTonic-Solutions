import numpy as np


def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.

    x: (C_in, H, W) or (N, C_in, H, W)
    W: (C_out, C_in, K_h, K_w)
    b: (C_out,)
    """
    x = np.asarray(x)
    W = np.asarray(W)
    b = np.asarray(b)

    unbatched = x.ndim == 3
    if unbatched:
        x = x[np.newaxis, ...]

    if x.ndim != 4 or W.ndim != 4 or b.ndim != 1:
        raise ValueError("Expected x to be 3D or 4D, W to be 4D, and b to be 1D")

    batch_size, in_channels, input_height, input_width = x.shape
    out_channels, kernel_channels, kernel_height, kernel_width = W.shape

    if kernel_channels != in_channels:
        raise ValueError("The input and kernel channel counts must match")
    if b.shape != (out_channels,):
        raise ValueError("There must be one bias value per output channel")
    if kernel_height > input_height or kernel_width > input_width:
        raise ValueError("The kernel cannot be larger than the input")

    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1
    output = np.empty(
        (batch_size, out_channels, output_height, output_width),
        dtype=np.result_type(x, W, b),
    )

    for row in range(output_height):
        for column in range(output_width):
            window = x[
                :,
                :,
                row : row + kernel_height,
                column : column + kernel_width,
            ]
            output[:, :, row, column] = np.sum(
                window[:, np.newaxis, :, :, :] * W[np.newaxis, :, :, :, :],
                axis=(2, 3, 4),
            ) + b

    return output[0] if unbatched else output

