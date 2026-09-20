"""Per-run training loop. Structurally follows the roadmap script's
train_one_run/run_adam_like_phase/run_lbfgs_phase, extended with the new
diagnostics (gradient-conflict cosine, Hutchinson trace) logged alongside
the roadmap-original loss/grad-norm history.
"""
import random
import time

import numpy as np
import pandas as pd
import torch

from .autodiff import gradients, to_tensor
from .diagnostics import (
    grad_norm_value,
    loss_oscillation_index,
    gradient_conflict_cosine,
    hutchinson_trace_estimate,
)
from .evaluate import evaluate_model, runtime_normalized_error
from .models import PINN
from .optimizers import build_first_order_optimizer
from .problems import PROBLEMS


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _check_finite(loss: torch.Tensor):
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss encountered.")


def make_problem_tensors(problem, device, dtype=torch.float32):
    return {
        "problem": problem,
        "X_f": to_tensor(problem["X_f"], device, dtype, requires_grad=True),
        "X_ic": to_tensor(problem["X_ic"], device, dtype, requires_grad=False),
        "y_ic": to_tensor(problem["y_ic"], device, dtype, requires_grad=False),
        "X_bc": to_tensor(problem["X_bc"], device, dtype, requires_grad=False),
        "y_bc": to_tensor(problem["y_bc"], device, dtype, requires_grad=False),
    }


def compute_losses(model, problem_tensors, problem_name):
    X_f, X_ic, y_ic = problem_tensors["X_f"], problem_tensors["X_ic"], problem_tensors["y_ic"]
    X_bc, y_bc = problem_tensors["X_bc"], problem_tensors["y_bc"]
    problem = problem_tensors["problem"]

    residual_fn = PROBLEMS[problem_name].residual
    f = residual_fn(model, X_f, problem, gradients)
    pde_loss = torch.mean(f ** 2)
    ic_loss = torch.mean((model(X_ic) - y_ic) ** 2)
    bc_loss = torch.mean((model(X_bc) - y_bc) ** 2)
    data_loss = ic_loss + bc_loss
    total_loss = pde_loss + data_loss
    return total_loss, pde_loss, ic_loss, bc_loss, data_loss


def _log_row(history_records, phase, step, total_loss, pde_loss, ic_loss, bc_loss, gnorm,
             grad_conflict=None, hutchinson=None):
    history_records.append({
        "phase": phase,
        "step": step,
        "total_loss": total_loss.detach().cpu().item(),
        "pde_loss": pde_loss.detach().cpu().item(),
        "ic_loss": ic_loss.detach().cpu().item(),
        "bc_loss": bc_loss.detach().cpu().item(),
        "grad_norm": gnorm,
        "grad_conflict_cosine": grad_conflict,
        "hutchinson_trace": hutchinson,
    })


def run_adam_like_phase(model, problem_tensors, problem_name, optimizer, steps,
                         history_records, phase_name, log_every=50,
                         diagnostics_every=250, hutchinson_samples=5):
    for step in range(steps):
        optimizer.zero_grad()
        total_loss, pde_loss, ic_loss, bc_loss, data_loss = compute_losses(
            model, problem_tensors, problem_name
        )
        _check_finite(total_loss)

        want_diagnostics = (step % diagnostics_every == 0)
        grad_conflict = None
        hutchinson = None
        if want_diagnostics:
            grad_conflict = gradient_conflict_cosine(model, pde_loss, data_loss)
            hutchinson = hutchinson_trace_estimate(model, total_loss, num_samples=hutchinson_samples)

        total_loss.backward()
        gnorm = grad_norm_value(model)
        optimizer.step()

        if step % log_every == 0 or step == steps - 1:
            _log_row(history_records, phase_name, step, total_loss, pde_loss, ic_loss, bc_loss,
                      gnorm, grad_conflict, hutchinson)


def run_lbfgs_phase(model, problem_tensors, problem_name, max_iter, history_records, phase_name,
                     diagnostics_every=10):
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=max_iter, max_eval=max_iter,
        history_size=50, line_search_fn="strong_wolfe",
    )
    call_count = [0]

    def closure():
        optimizer.zero_grad()
        total_loss, pde_loss, ic_loss, bc_loss, data_loss = compute_losses(
            model, problem_tensors, problem_name
        )
        _check_finite(total_loss)
        want_diagnostics = (call_count[0] % diagnostics_every == 0)
        grad_conflict, hutchinson = None, None
        if want_diagnostics:
            grad_conflict = gradient_conflict_cosine(model, pde_loss, data_loss)
            hutchinson = hutchinson_trace_estimate(model, total_loss, num_samples=5)
        total_loss.backward()
        if call_count[0] % diagnostics_every == 0:
            _log_row(history_records, phase_name, call_count[0], total_loss, pde_loss, ic_loss,
                      bc_loss, grad_norm_value(model), grad_conflict, hutchinson)
        call_count[0] += 1
        return total_loss

    optimizer.step(closure)


def train_one_run(problem, problem_name, optimizer_name, seed, steps_cfg, hparams, device,
                   dtype=torch.float32):
    set_seed(seed)
    model = PINN(problem["lb"], problem["ub"], dtype=dtype).to(device)
    problem_tensors = make_problem_tensors(problem, device, dtype)
    history_records = []
    start_time = time.time()

    optimizer_hparams = (hparams or {}).get(optimizer_name, {})

    if optimizer_name == "LBFGS":
        run_lbfgs_phase(model, problem_tensors, problem_name, steps_cfg["lbfgs_max_iter"],
                         history_records, "LBFGS")
    elif optimizer_name == "Adam_LBFGS":
        optimizer = build_first_order_optimizer("Adam", model.parameters(), optimizer_hparams)
        run_adam_like_phase(model, problem_tensors, problem_name, optimizer,
                             steps_cfg["hybrid_adam_steps"], history_records, "Adam_pretrain")
        run_lbfgs_phase(model, problem_tensors, problem_name, steps_cfg["lbfgs_max_iter"],
                         history_records, "LBFGS_refine")
    else:
        optimizer = build_first_order_optimizer(optimizer_name, model.parameters(), optimizer_hparams)
        steps = steps_cfg["shampoo_steps"] if optimizer_name in ("Shampoo", "SOAP", "Muon") \
            else steps_cfg["adam_steps"]
        run_adam_like_phase(model, problem_tensors, problem_name, optimizer, steps,
                             history_records, optimizer_name)

    runtime = time.time() - start_time
    rel_l2, residual_mse, y_pred = evaluate_model(model, problem, problem_name, device, dtype)
    final_total, final_pde, final_ic, final_bc, _ = compute_losses(model, problem_tensors, problem_name)

    losses = [r["total_loss"] for r in history_records]
    failure_flag = "ok"
    if not np.isfinite(rel_l2):
        failure_flag = "nonfinite_error"
    elif rel_l2 > 5.0:
        failure_flag = "large_error"

    result = {
        "problem": problem_name,
        "optimizer": optimizer_name,
        "seed": seed,
        "relative_solution_gap": rel_l2,
        "pde_residual_mse": residual_mse,
        "final_total_loss": final_total.detach().cpu().item(),
        "final_pde_loss": final_pde.detach().cpu().item(),
        "final_ic_loss": final_ic.detach().cpu().item(),
        "final_bc_loss": final_bc.detach().cpu().item(),
        "runtime_seconds": runtime,
        "runtime_normalized_error": runtime_normalized_error(rel_l2, runtime),
        "loss_oscillation_index": loss_oscillation_index(losses),
        "num_logged_points": len(history_records),
        "num_parameters": model.num_parameters(),
        "failure_flag": failure_flag,
        "failure_message": "",
    }
    history = pd.DataFrame(history_records)
    return result, history, y_pred
