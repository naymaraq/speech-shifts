import math


def _cosine_annealing(initial_lr, step, max_steps, min_lr):
    mult = 0.5 * (1 + math.cos(math.pi * step / max_steps))
    out_lr = (initial_lr - min_lr) * mult + min_lr
    return out_lr


def _linear_warmup_with_cosine_annealing(max_lr, warmup_steps, step, decay_steps, min_lr):

    assert max_lr > min_lr
    # Use linear warmup for the initial part.
    if warmup_steps > 0 and step <= warmup_steps:
        return max_lr * float(step) / float(warmup_steps)

    # For any steps larger than `decay_steps`, use `min_lr`.
    if step > warmup_steps + decay_steps:
        return min_lr

    # If we are done with the warmup period, use the decay style.
    num_steps_ = step - warmup_steps
    decay_steps_ = decay_steps
    decay_ratio = float(num_steps_) / float(decay_steps_)
    assert decay_ratio >= 0.0
    assert decay_ratio <= 1.0
    delta_lr = max_lr - min_lr

    coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)

    return min_lr + coeff * delta_lr