import pytest
import torch

from pinn_bench.optimizers.shampoo import StableShampoo
from pinn_bench.optimizers.soap import SOAP
from pinn_bench.optimizers.muon import Muon, _newton_schulz5


@pytest.mark.parametrize("optimizer_cls,kwargs", [
    (StableShampoo, dict(lr=0.05, update_freq=5)),
    (SOAP, dict(lr=0.01, update_freq=5)),
    (Muon, dict(lr=0.02, adam_lr=0.01)),
])
def test_optimizer_reduces_loss_on_toy_regression(optimizer_cls, kwargs):
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)
    )
    X = torch.randn(64, 4)
    true_W = torch.randn(4, 1)
    y = X @ true_W + 0.01 * torch.randn(64, 1)

    optimizer = optimizer_cls(model.parameters(), **kwargs)
    losses = []
    for _ in range(60):
        optimizer.zero_grad()
        pred = model(X)
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0] * 0.5, f"loss did not decrease enough: {losses[0]} -> {losses[-1]}"


def test_newton_schulz5_bounds_singular_values():
    """The quintic Newton-Schulz map used by Muon does NOT converge to exact
    orthogonality as steps -> infinity -- its scalar map s -> a*s+b*s^3+c*s^5
    has fixed points near 0.87 and 1.26 (not 1) and is chaotic outside a
    narrow basin, which we verified numerically before writing this test
    (running more iterations does not monotonically improve orthogonality).
    Its actual, documented guarantee (Jordan et al. 2024) is that after the
    default 5 steps it keeps singular values within a bounded range of 1,
    which is all Muon's momentum-orthogonalization needs -- so that is what
    this test checks, rather than asserting exact orthonormality.
    """
    torch.manual_seed(0)
    G = torch.randn(8, 5)
    ortho = _newton_schulz5(G, steps=5)
    sing_vals = torch.linalg.svdvals(ortho)
    assert torch.all(sing_vals > 0.3)
    assert torch.all(sing_vals < 2.0)
