"""Training diagnostics.

grad_norm_value and loss_oscillation_index are roadmap-original.
gradient_conflict_cosine and hutchinson_trace_estimate are new additions
for this project, aimed at *explaining* optimizer behavior rather than
just scoring it: a negative PDE-vs-data gradient cosine tells you the two
loss terms are fighting each other, and the Hutchinson trace gives a cheap
proxy for how sharply curved the loss landscape is at that point, both of
which are exactly the kind of ill-conditioning that motivates trying a
preconditioned optimizer like Shampoo/SOAP/Muon in the first place.
"""
import numpy as np
import torch


def grad_norm_value(model) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.detach().norm().item() ** 2
    return total_norm ** 0.5


def loss_oscillation_index(losses) -> float:
    if len(losses) < 3:
        return 0.0
    diffs = np.diff(np.array(losses))
    sign_changes = np.sum(np.sign(diffs[1:]) != np.sign(diffs[:-1]))
    return float(sign_changes / max(1, len(diffs) - 1))


def gradient_conflict_cosine(model, pde_loss, data_loss) -> float:
    """cos(angle) between grad(pde_loss) and grad(data_loss) w.r.t. model params.

    +1: the two objectives pull in the same direction (easy).
    -1: they directly oppose each other (a classic PINN failure mode --
        driving down the PDE residual pushes the boundary/initial fit
        worse, and vice versa).
     0: unrelated directions.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    grad_pde = torch.autograd.grad(pde_loss, params, retain_graph=True, allow_unused=True)
    grad_data = torch.autograd.grad(data_loss, params, retain_graph=True, allow_unused=True)

    # `allow_unused=True` returns None for a parameter a loss doesn't
    # functionally depend on -- e.g. the PINN's final-layer bias vanishes
    # under the spatial/temporal differentiation inside the PDE residual, so
    # grad_pde is None there even though grad_data is not. Zero-fill rather
    # than drop those entries so both flattened vectors stay aligned
    # per-parameter (dropping them independently silently shifted the two
    # vectors out of alignment and produced a shape-mismatch crash the first
    # time this was actually run).
    flat_pde = torch.cat([
        (g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1))
        for g, p in zip(grad_pde, params)
    ])
    flat_data = torch.cat([
        (g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1))
        for g, p in zip(grad_data, params)
    ])

    denom = flat_pde.norm() * flat_data.norm()
    if denom.item() < 1e-12:
        return 0.0
    cos = torch.dot(flat_pde, flat_data) / denom
    return float(cos.detach().cpu().item())


def hutchinson_trace_estimate(model, loss, num_samples: int = 5) -> float:
    """Stochastic estimate of tr(Hessian) of `loss` w.r.t. model parameters,
    via Hutchinson's estimator with Rademacher probe vectors:
        tr(H) ~= E[z^T H z],  z_i in {-1, +1} iid.

    Used here purely as a lightweight scalar curvature proxy logged
    alongside the loss curves -- larger magnitude generally indicates a
    sharper local landscape, which is where preconditioned optimizers are
    expected to help most and plain SGD/Adam-like steps struggle most.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    flat_grad = torch.cat([g.reshape(-1) for g in grads])

    estimates = []
    for _ in range(num_samples):
        z = torch.randint(0, 2, flat_grad.shape, device=flat_grad.device, dtype=flat_grad.dtype)
        z = z * 2 - 1  # Rademacher: {0,1} -> {-1,+1}
        gz = torch.dot(flat_grad, z)
        hz = torch.autograd.grad(gz, params, retain_graph=True, allow_unused=True)
        flat_hz = torch.cat([
            (h.reshape(-1) if h is not None else torch.zeros_like(p).reshape(-1))
            for h, p in zip(hz, params)
        ])
        estimates.append(torch.dot(z, flat_hz).detach().cpu().item())
    return float(np.mean(estimates))
