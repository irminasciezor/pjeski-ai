from math import cos, pi


def lr_cosine(base_lr, epoch, epochs):
    if epochs <= 0:
        return base_lr
    return base_lr * 0.5 * (1 + cos(pi * epoch / epochs))


def lr_cosine_czas(base_lr, t_elapsed, t_max, eta_min=0.0):
    # to samo co cosine, ale po czasie a nie po numerze epoki
    if t_max <= 0:
        return base_lr
    postep = min(max(t_elapsed / t_max, 0.0), 1.0)
    return eta_min + (base_lr - eta_min) * 0.5 * (1 + cos(pi * postep))
