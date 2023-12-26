import torch
from .regularizers import entropic_regularizer


def unexplained_variance(benchmark, model, batch_size=8192):
    X = benchmark.input_sampler.sample(batch_size)
    X.requires_grad_(True)
    Y = benchmark.map_fwd(X, nograd=True)

    with torch.no_grad():
        T_X = model(X)

    uvp = 100 * ((Y - T_X) ** 2).sum(dim=-1).mean() / benchmark.output_sampler.var
    return uvp.item()


def sinkhorn_distance(Y, T_X, cost_fn, epsilon=0.1):
    reg1 = entropic_regularizer(T_X, Y, cost_fn, epsilon, expand_dims=False).item()
    reg2 = entropic_regularizer(T_X, T_X, cost_fn, epsilon, expand_dims=False).item()
    reg3 = entropic_regularizer(Y, Y, cost_fn, epsilon, expand_dims=False).item()
    return reg1 - 0.5 * (reg2 + reg3)
