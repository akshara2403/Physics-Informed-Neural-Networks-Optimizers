"""Evaluation metrics.

evaluate_model is roadmap-original (relative L2 solution gap + PDE residual
MSE on held-out points). runtime_normalized_error is new: it penalizes an
optimizer for buying a small gap with a large wall-clock cost, which a raw
accuracy comparison hides.
"""
import numpy as np
import torch

from .autodiff import gradients, to_tensor


def evaluate_model(model, problem, problem_name, device, dtype=torch.float32):
    model.eval()
    X_test = to_tensor(problem["X_test"], device, dtype, requires_grad=False)
    y_test = to_tensor(problem["y_test"], device, dtype, requires_grad=False)

    with torch.no_grad():
        y_pred = model(X_test)
        rel_l2 = torch.linalg.norm(y_test - y_pred) / torch.linalg.norm(y_test)

    from .problems import PROBLEMS
    problem_module = PROBLEMS[problem_name]

    n_eval = min(2000, problem["X_f"].shape[0])
    idx = np.random.choice(problem["X_f"].shape[0], size=n_eval, replace=False)
    X_res = to_tensor(problem["X_f"][idx], device, dtype, requires_grad=True)
    f = problem_module.residual(model, X_res, problem, gradients)
    residual_mse = torch.mean(f ** 2).detach().cpu().item()
    model.train()
    return rel_l2.detach().cpu().item(), residual_mse, y_pred.detach().cpu().numpy()


def runtime_normalized_error(relative_gap: float, runtime_seconds: float) -> float:
    """gap * sqrt(runtime): a simple cost-aware accuracy score.

    Two optimizers reaching the same solution gap are not equally good if
    one takes 10x longer. Using sqrt(runtime) rather than raw runtime keeps
    the penalty from completely swamping the accuracy term for benchmarks
    where all optimizers finish within a similar order of magnitude of
    wall-clock time (as is the case on the small CPU-only runs in this
    repository); the exact exponent is a modeling choice, not a physical law.
    """
    if not np.isfinite(relative_gap) or not np.isfinite(runtime_seconds) or runtime_seconds < 0:
        return float("nan")
    return float(relative_gap * np.sqrt(runtime_seconds))
