import torch
from torch import linalg


def get_optimal_plan(cost_matrix, a_probs, b_probs, entropy_eps, max_iter=300):
    K = -cost_matrix / entropy_eps
    u = torch.zeros_like(a_probs, device=a_probs.device)
    v = torch.zeros_like(b_probs, device=b_probs.device)
    loga = a_probs.log()
    logb = b_probs.log()

    for _ in range(max_iter):
        v = logb - torch.logsumexp(K + u[:, None], 0)
        u = loga - torch.logsumexp(K + v[None, :], 1)

    return torch.exp(K + u[:, None] + v[None, :])


def matrix_square_root(matrix):
    eig_values, eig_vectors = linalg.eigh(matrix)
    return eig_vectors @ torch.diag(eig_values.sqrt()) @ eig_vectors.T


def build_cost_matrix(x, y, cost_fn, expand_dims=True):
    if expand_dims:
        return cost_fn(x[:, None, :], y[None, ...])

    n, m = len(x), len(y)
    cost_matrix = torch.zeros((n, m), device=x.device)

    for i, x_object in enumerate(x):
        cost_matrix[i] = cost_fn(x_object, y)

    return cost_matrix
