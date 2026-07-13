import math

import torch


def densenet_channel_counts(
    stem_channels: int, growth_rate: int, block_layers, compression: float
) -> torch.Tensor:
    """
    Returns a 1D int64 torch.Tensor of channel counts at each stage.
    """
    # YOUR CODE HERE
    result = [stem_channels]
    for b in block_layers:
        l = result[-1]
        # expansion
        result.append(l + b * growth_rate)

        # compression
        l = result[-1]
        result.append(math.floor(compression * l))
    # no compression after the last block
    result.pop()
    return torch.tensor(result)
    

