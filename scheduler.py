from math import cos, pi


def lr_cosine(base_lr, epoch, epochs):
    return base_lr * 0.5 * (1 + cos(pi * epoch / epochs))
