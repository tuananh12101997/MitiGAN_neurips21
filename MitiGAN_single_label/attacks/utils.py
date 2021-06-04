import torch


def clamp_input(inputs, normalizer):
    min = normalizer(torch.zeros_like(inputs) - 1.0)
    max = normalizer(torch.ones_like(inputs))
    return torch.max(torch.min(inputs, max), min)
