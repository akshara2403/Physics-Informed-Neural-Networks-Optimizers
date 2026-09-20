"""Muon: a from-scratch re-implementation of the Newton-Schulz-orthogonalized
momentum optimizer (Jordan et al., 2024, "Muon: MomentUm Orthogonalized by
Newton-schulz"). Independent reimplementation for this project.

Idea: for >=2D weight matrices, take an SGD-momentum buffer and replace it
with its nearest semi-orthogonal matrix (via a quintic Newton-Schulz
iteration, avoiding an explicit SVD), which empirically decorrelates the
update across output directions. 1D parameters (biases, and the PINN's
scalar-ish layers) are not matrices in any meaningful sense, so they fall
back to plain Adam, following the recommendation in the original Muon
write-up to pair Muon (for matrices) with Adam (for everything else).
"""
import torch


def _newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton-Schulz iteration approximating G (G^T G)^{-1/2}."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + eps)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-2, momentum=0.95, weight_decay=0.0,
                 ns_steps=5, adam_lr=1e-3, adam_betas=(0.9, 0.999), adam_eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                         ns_steps=ns_steps, adam_lr=adam_lr, adam_betas=adam_betas,
                         adam_eps=adam_eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            adam_lr = group["adam_lr"]
            beta1, beta2 = group["adam_betas"]
            adam_eps = group["adam_eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if not torch.isfinite(g).all():
                    continue
                if weight_decay != 0.0:
                    g = g.add(p, alpha=weight_decay)

                state = self.state[p]

                if g.dim() >= 2:
                    mat = g.reshape(g.shape[0], -1)
                    if "muon_buf" not in state:
                        state["muon_buf"] = torch.zeros_like(mat)
                    buf = state["muon_buf"]
                    buf.mul_(momentum).add_(mat)
                    ortho = _newton_schulz5(buf, steps=ns_steps)
                    scale = max(1.0, mat.shape[0] / mat.shape[1]) ** 0.5
                    update = (ortho * scale).reshape(g.shape)
                    if not torch.isfinite(update).all():
                        continue
                    p.add_(update, alpha=-lr)
                else:
                    if "m" not in state:
                        state["m"] = torch.zeros_like(p)
                        state["v"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    t = state["step"]
                    state["m"].mul_(beta1).add_(g, alpha=1 - beta1)
                    state["v"].mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    m_hat = state["m"] / (1 - beta1 ** t)
                    v_hat = state["v"] / (1 - beta2 ** t)
                    update = m_hat / (v_hat.sqrt() + adam_eps)
                    if not torch.isfinite(update).all():
                        continue
                    p.add_(update, alpha=-adam_lr)

        return loss
