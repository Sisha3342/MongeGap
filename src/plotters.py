import matplotlib.pyplot as plt
from matplotlib import collections as mc


def plot_losses(losses):
    ncols = len(losses)
    fig, axes = plt.subplots(1, ncols, figsize=(9, 4), dpi=100)

    if ncols == 1:
        axes = [axes]

    for i, (title, values) in enumerate(losses.items()):
        axes[i].plot(values)
        axes[i].set_title(title, fontsize=12)

    fig.tight_layout()
    plt.show(fig)
    plt.close(fig)


def plot_mapping(X, Y, T_X, connections_batch_size):
    fig, axes = plt.subplots(1, 3, figsize=(9, 4), dpi=100, sharex=True, sharey=True)

    axes[0].scatter(
        X[:connections_batch_size, 0],
        X[:connections_batch_size, 1],
        c="darkseagreen",
        edgecolors="black",
        zorder=2,
        label=r"Input $x\sim\mathbb{P}$",
    )
    axes[0].scatter(
        T_X[:connections_batch_size, 0],
        T_X[:connections_batch_size, 1],
        c="tomato",
        edgecolors="black",
        zorder=3,
        label=r"Mapped $y=T(x)$",
    )
    lines = list(zip(X[:connections_batch_size], T_X[:connections_batch_size]))
    lc = mc.LineCollection(lines, linewidths=0.5, color="black")
    axes[0].add_collection(lc)
    axes[0].legend(loc="upper right")

    axes[1].scatter(
        T_X[:, 0], T_X[:, 1], c="tomato", edgecolors="black", label=r"Mapped $T(x)$"
    )
    axes[1].legend(loc="upper right")

    axes[2].scatter(
        Y[:, 0], Y[:, 1], c="wheat", edgecolors="black", label=r"Target $\mathbb{Q}$"
    )
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    plt.show(fig)
    plt.close(fig)


def plot_results(X, Y, T_X, losses, emb_X=None, emb_Y=None, connections_batch_size=64):
    dim = X.shape[-1]

    if dim <= 2 or emb_X is None or emb_Y is None:  # Use PCA when dim > 2
        X = X.cpu().detach().numpy()
        Y = Y.cpu().detach().numpy()
        T_X = T_X.cpu().detach().numpy()
    else:
        X = emb_X.transform(X.cpu().detach().numpy())
        Y = emb_Y.transform(Y.cpu().detach().numpy())
        T_X = emb_Y.transform(T_X.cpu().detach().numpy())

    plot_mapping(X, Y, T_X, connections_batch_size)
    plot_losses(losses)
