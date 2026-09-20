"""StableShampoo: a from-scratch, numerically stabilized re-implementation
of Shampoo-style matrix preconditioning (Gupta, Koren & Singer 2018).

Motivation (see README "Motivation" section for the full story): the
original roadmap pilot run used the third-party `torch-optimizer` Shampoo
implementation, which failed on 3 of 4 real pilot runs with

    linalg.svd: The algorithm failed to converge because the input matrix
    is ill-conditioned or has too many repeated singular values (error code: 14)

This implementation avoids SVD entirely. Left/right preconditioners are
symmetric positive semi-definite by construction (G @ G.T and G.T @ G), so
their inverse fourth roots are computed via `torch.linalg.eigh` (which is
both cheaper and numerically better-behaved than SVD for this case),
combined with an explicit eigenvalue floor and gradient clipping as a
second line of defense.

This is a simplified, per-parameter ("block-diagonal") Shampoo: each >=2D
parameter tensor is treated as a single matrix (reshaping if needed) with
its own pair of preconditioners; 1D parameters (mostly biases) fall back to
an Adagrad-style diagonal preconditioner, since a matrix Shampoo update is
not meaningful for a vector.
"""
import torch


def _matrix_inverse_root(mat: torch.Tensor, power: float, eps: float) -> torch.Tensor:
    """Symmetric PSD matrix to the given (typically negative fractional) power via eigh."""
    n = mat.shape[0]
    mat = mat + eps * torch.eye(n, device=mat.device, dtype=mat.dtype)
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = eigvals.clamp_min(eps)
    return (eigvecs * eigvals.pow(power)) @ eigvecs.T


class StableShampoo(torch.optim.Optimizer):
    """
    Numerical-stability note (found by actually running this optimizer, not
    just by inspection): with the eigenvalue floor as the *only* safeguard,
    an early L/R accumulator that is still near rank-deficient (e.g. after a
    single gradient) has some eigenvalues within numerical noise of zero.
    Raising such a near-zero eigenvalue to the -1/4 power blows it up by a
    factor of eps^{-1/4}, and that huge preconditioner gets applied to
    *every subsequent gradient* until the next `update_freq` refresh -- in
    an early test run on a toy regression this alone was enough to diverge
    a loss from ~4.7 to >7000 within 10 steps. Two additional safeguards
    fix this: (1) a `precond_clip` trust-region bound that rescales the
    *preconditioned* update back down whenever it is disproportionately
    larger than the raw gradient it came from, and (2) an accumulation
    warmup (`min_accum_steps`) that keeps the preconditioner at identity
    until L/R have accumulated enough curvature information to be reliably
    inverted. Both are on by default.
    """

    def __init__(self, params, lr=1e-2, momentum=0.9, weight_decay=0.0,
                 epsilon=1e-4, update_freq=10, max_grad_norm=10.0,
                 precond_clip=5.0, min_accum_steps=5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                         epsilon=epsilon, update_freq=update_freq,
                         max_grad_norm=max_grad_norm, precond_clip=precond_clip,
                         min_accum_steps=min_accum_steps)
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
            eps = group["epsilon"]
            update_freq = group["update_freq"]
            max_gn = group["max_grad_norm"]
            precond_clip = group["precond_clip"]
            min_accum_steps = group["min_accum_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if not torch.isfinite(g).all():
                    continue  # safety net: skip a corrupted update rather than crash
                gn = g.norm()
                if gn > max_gn:
                    g = g * (max_gn / (gn + 1e-12))
                if weight_decay != 0.0:
                    g = g.add(p, alpha=weight_decay)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(p)

                if g.dim() >= 2:
                    mat = g.reshape(g.shape[0], -1)
                    if "L" not in state:
                        state["L"] = torch.zeros(mat.shape[0], mat.shape[0], device=p.device, dtype=p.dtype)
                        state["R"] = torch.zeros(mat.shape[1], mat.shape[1], device=p.device, dtype=p.dtype)
                        state["L_inv_root"] = torch.eye(mat.shape[0], device=p.device, dtype=p.dtype)
                        state["R_inv_root"] = torch.eye(mat.shape[1], device=p.device, dtype=p.dtype)
                    state["L"].add_(mat @ mat.T)
                    state["R"].add_(mat.T @ mat)
                    if state["step"] >= min_accum_steps and state["step"] % update_freq == 0:
                        state["L_inv_root"] = _matrix_inverse_root(state["L"], -0.25, eps)
                        state["R_inv_root"] = _matrix_inverse_root(state["R"], -0.25, eps)
                    if state["step"] < min_accum_steps:
                        precond = mat  # warmup: behave like plain (momentum) SGD until L/R are informative
                    else:
                        precond = state["L_inv_root"] @ mat @ state["R_inv_root"]
                    precond = precond.reshape(g.shape)
                    # Trust-region clip: never let preconditioning amplify the
                    # step more than `precond_clip`x beyond the raw gradient.
                    raw_norm = g.norm()
                    precond_norm = precond.norm()
                    if precond_norm > precond_clip * raw_norm + 1e-12:
                        precond = precond * (precond_clip * raw_norm / (precond_norm + 1e-12))
                else:
                    if "diag_accum" not in state:
                        state["diag_accum"] = torch.zeros_like(p)
                    state["diag_accum"].add_(g * g)
                    precond = g / (state["diag_accum"].sqrt() + eps)

                if not torch.isfinite(precond).all():
                    state["step"] += 1
                    continue

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(precond)
                p.add_(buf, alpha=-lr)
                state["step"] += 1

        return loss
