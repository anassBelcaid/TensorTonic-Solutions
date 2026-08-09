from math import cos, pi


def cosine_annealing_schedule(base_lr, min_lr, total_steps, current_step):
    """
    Compute the learning rate using cosine annealing.
    """
    # Write code here
    return min_lr + 0.5 * (base_lr - min_lr) * (
        1 + cos(pi * current_step / total_steps)
    )
