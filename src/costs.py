import torch


def quadratic_cost(x, y):
    return 0.5 * torch.sum((x - y) ** 2, dim=-1)


def l2_squared_cost(x, y):
    return torch.sum((x - y) ** 2, dim=-1)


def l2_cost(x, y):
    return torch.sum((x - y) ** 2, dim=-1).sqrt()
