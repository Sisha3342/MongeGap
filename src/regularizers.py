import math
import torch
from torch.func import jvp, vjp
from .utils import build_cost_matrix, get_optimal_plan


def entropic_regularizer(
    X_samples, Y_samples, cost_fn, epsilon=None, expand_dims=True, log_epsilon=1e-12
):
    cost_matrix = build_cost_matrix(X_samples, Y_samples, cost_fn, expand_dims)

    with torch.no_grad():
        epsilon = epsilon or 0.01 * torch.mean(cost_matrix)
        x_probs = torch.ones(len(X_samples), device=X_samples.device) / len(X_samples)
        y_probs = torch.ones(len(Y_samples), device=X_samples.device) / len(Y_samples)
        P = get_optimal_plan(cost_matrix, x_probs, y_probs, epsilon)

    entropy = torch.sum(P * torch.log(P + log_epsilon))
    return torch.sum(P * cost_matrix) + epsilon * entropy


def conservativity_regularizer(model, X_samples, hutchinson_count=None):
    n, d = X_samples.shape
    hutchinson_count = hutchinson_count or int(math.ceil(0.2 * d))
    hutchinson_vectors = torch.randn((hutchinson_count, d), device=X_samples.device)
    vjp_function = vjp(model, X_samples)[1]
    loss = 0

    for hutchinson_vector in hutchinson_vectors:
        hutchinson_vector = hutchinson_vector.expand(n, -1)
        diff = (
            jvp(model, (X_samples,), (hutchinson_vector,))[1]
            - vjp_function(hutchinson_vector)[0]
        )
        loss += torch.sum(diff**2)

    return loss / n / hutchinson_count
