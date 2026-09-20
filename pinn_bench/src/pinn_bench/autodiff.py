"""Autograd helpers shared by every problem's PDE residual."""
import torch


def gradients(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """d(y)/d(x) via autograd, keeping the graph for higher-order derivatives."""
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def to_tensor(arr, device, dtype=torch.float32, requires_grad=False):
    return torch.tensor(arr, dtype=dtype, device=device, requires_grad=requires_grad)
