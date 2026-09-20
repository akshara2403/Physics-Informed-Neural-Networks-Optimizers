"""SOAP: a from-scratch, simplified re-implementation inspired by
"SOAP: Improving and Stabilizing Shampoo using Adam" (Vyas, Morwani et al.,
2024). This is an independent reimplementation for this project, not a
port of the authors' code, and is deliberately simplified (no
Kronecker-factored second-moment coupling across the full optimizer
state) -- treat it as "Shampoo-basis-rotated Adam," which is the core idea
the name describes.

Mechanism: periodically compute the eigenbases of the same G@G.T / G.T@G
accumulators used by StableShampoo (via eigh, for the same numerical-
stability reasons), rotate the gradient into that eigenbasis, run ordinary
Adam bookkeeping (first/second moment EMA) in the rotated space, then
rotate the update back. The eigenbasis captures the same curvature
information Shampoo's preconditioner does, while Adam's per-coordinate
normalization in that basis is cheaper and better-conditioned than
Shampoo's matrix inverse-root at every step.
"""
import torch


class SOAP(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, update_freq=10, ema_decay=0.95):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                         update_freq=update_freq, ema_decay=ema_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            update_freq = group["update_freq"]
            ema_decay = group["ema_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if not torch.isfinite(g).all():
                    continue
                if weight_decay != 0.0:
                    g = g.add(p, alpha=weight_decay)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0

                if g.dim() >= 2:
                    mat = g.reshape(g.shape[0], -1)
                    if "L" not in state:
                        m0, n0 = mat.shape
                        state["L"] = torch.zeros(m0, m0, device=p.device, dtype=p.dtype)
                        state["R"] = torch.zeros(n0, n0, device=p.device, dtype=p.dtype)
                        state["QL"] = torch.eye(m0, device=p.device, dtype=p.dtype)
                        state["QR"] = torch.eye(n0, device=p.device, dtype=p.dtype)
                        state["m"] = torch.zeros(m0, n0, device=p.device, dtype=p.dtype)
                        state["v"] = torch.zeros(m0, n0, device=p.device, dtype=p.dtype)

                    state["L"].mul_(ema_decay).add_(mat @ mat.T, alpha=1 - ema_decay)
                    state["R"].mul_(ema_decay).add_(mat.T @ mat, alpha=1 - ema_decay)

                    if state["step"] % update_freq == 0:
                        _, state["QL"] = torch.linalg.eigh(
                            state["L"] + eps * torch.eye(state["L"].shape[0], device=p.device, dtype=p.dtype)
                        )
                        _, state["QR"] = torch.linalg.eigh(
                            state["R"] + eps * torch.eye(state["R"].shape[0], device=p.device, dtype=p.dtype)
                        )

                    QL, QR = state["QL"], state["QR"]
                    g_rot = QL.T @ mat @ QR

                    state["step"] += 1
                    t = state["step"]
                    state["m"].mul_(beta1).add_(g_rot, alpha=1 - beta1)
                    state["v"].mul_(beta2).addcmul_(g_rot, g_rot, value=1 - beta2)
                    m_hat = state["m"] / (1 - beta1 ** t)
                    v_hat = state["v"] / (1 - beta2 ** t)
                    update_rot = m_hat / (v_hat.sqrt() + eps)
                    update = (QL @ update_rot @ QR.T).reshape(g.shape)
                else:
                    if "m" not in state:
                        state["m"] = torch.zeros_like(p)
                        state["v"] = torch.zeros_like(p)
                    state["step"] += 1
                    t = state["step"]
                    state["m"].mul_(beta1).add_(g, alpha=1 - beta1)
                    state["v"].mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    m_hat = state["m"] / (1 - beta1 ** t)
                    v_hat = state["v"] / (1 - beta2 ** t)
                    update = m_hat / (v_hat.sqrt() + eps)

                if not torch.isfinite(update).all():
                    continue
                p.add_(update, alpha=-lr)

        return loss
