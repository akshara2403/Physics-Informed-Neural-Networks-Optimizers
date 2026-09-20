"""PINN architecture (roadmap-original, unchanged in spirit)."""
import torch
import torch.nn as nn


class PINN(nn.Module):
    """Fully connected PINN mapping (x, t) -> u(x, t).

    Inputs are rescaled to [-1, 1] using registered buffers ``lb``/``ub``,
    exactly as in the roadmap reference script -- this normalization
    materially stabilizes training and is kept unchanged.
    """

    def __init__(self, lb, ub, layers=(2, 32, 32, 32, 1), dtype=torch.float32):
        super().__init__()
        self.register_buffer("lb", torch.tensor(lb, dtype=dtype).reshape(1, -1))
        self.register_buffer("ub", torch.tensor(ub, dtype=dtype).reshape(1, -1))
        self.layer_sizes = tuple(layers)
        self.layers = nn.ModuleList(
            nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)
        )
        self.activation = torch.tanh
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        z = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for layer in self.layers[:-1]:
            z = self.activation(layer(z))
        return self.layers[-1](z)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
