import torch
from torch import nn, linalg
from .utils import matrix_square_root


class MLP(nn.Module):
    def __init__(self, dim_size, hid_sizes=[128, 64, 64]):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim_size, hid_sizes[0]),
            nn.GELU(),
        )

        for hid_in, hid_out in zip(hid_sizes, hid_sizes[1:]):
            self.mlp.append(nn.Linear(hid_in, hid_out))
            self.mlp.append(nn.GELU())

        self.mlp.append(nn.Linear(hid_sizes[-1], dim_size))

        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, X):
        return self.mlp(X)


class MLPGeneric(MLP):
    def forward(self, X):
        return X + self.mlp(X)


class MLPQuadratic(MLP):
    def __init__(self, dim_size, hid_sizes=[128, 64, 64]):
        super().__init__(dim_size, hid_sizes)

        self.affine_weight = nn.Parameter(
            torch.zeros((dim_size, dim_size)), requires_grad=False
        )
        self.affine_bias = nn.Parameter(torch.zeros(dim_size), requires_grad=False)

    def gaussian_init(self, X_samples, Y_samples):
        x_mu = torch.mean(X_samples, dim=0)
        x_sigma_sq = matrix_square_root(torch.cov(X_samples.T))
        x_sigma_sq_inv = linalg.inv(x_sigma_sq)

        y_mu = torch.mean(Y_samples, dim=0)
        y_sigma = torch.cov(Y_samples.T)

        linear = (
            x_sigma_sq_inv
            @ matrix_square_root(x_sigma_sq @ y_sigma @ x_sigma_sq)
            @ x_sigma_sq_inv
        )
        bias = -linear @ x_mu + y_mu

        dim_size = self.affine_bias.shape[0]
        self.affine_weight.copy_(torch.eye(dim_size).to(X_samples.device) - linear)
        self.affine_bias.copy_(-bias)

    def identity_init(self):
        self.affine_weight.copy_(torch.zeros_like(self.affine_weight))
        self.affine_bias.copy_(torch.zeros_like(self.affine_bias))

    def forward(self, X):
        mlp_output = X @ self.affine_weight.T + self.affine_bias + self.mlp(X)
        return X - mlp_output
