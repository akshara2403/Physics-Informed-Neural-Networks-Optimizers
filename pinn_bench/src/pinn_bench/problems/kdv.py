"""Korteweg-de Vries (KdV) one-soliton benchmark.

This benchmark is NOT part of the original roadmap. It is included here to
stress-test the optimizers on a PDE whose residual requires a third-order
spatial derivative (u_xxx), which none of the roadmap's three benchmarks
exercise -- Shampoo/SOAP/Muon-style preconditioning is most often motivated
by ill-conditioned, high-order-derivative loss landscapes, so this is a more
discriminating test bed for the very optimizers this project studies.

PDE:  u_t + 6*u*u_x + u_xxx = 0,   x in [-10, 10], t in [0, 1]

Exact one-soliton solution (verified symbolically in tests/test_problems.py
with SymPy, and numerically by residual evaluation on a dense grid):
    u(x, t) = (c/2) * sech^2( sqrt(c)/2 * (x - c*t - x0) )
for any wave speed c > 0 and phase offset x0.
"""
import numpy as np

NAME = "kdv"

_C = 4.0     # soliton speed
_X0 = -4.0   # initial phase offset, chosen so the soliton stays in-domain over t in [0,1]


def exact_solution(x, t, c=_C, x0=_X0):
    arg = np.sqrt(c) / 2.0 * (x - c * t - x0)
    return (c / 2.0) / np.cosh(arg) ** 2


def make_data(seed, n_f=1500, n_ic=150, n_bc=150, **_):
    rng = np.random.default_rng(seed)
    lb = np.array([-10.0, 0.0])
    ub = np.array([10.0, 1.0])

    x_f = rng.uniform(lb[0], ub[0], size=(n_f, 1))
    t_f = rng.uniform(lb[1], ub[1], size=(n_f, 1))
    X_f = np.hstack([x_f, t_f])

    x_i = rng.uniform(lb[0], ub[0], size=(n_ic, 1))
    t_i = np.zeros_like(x_i)
    X_ic = np.hstack([x_i, t_i])
    y_ic = exact_solution(x_i, t_i)

    t_b = rng.uniform(lb[1], ub[1], size=(n_bc // 2, 1))
    X_bc = np.vstack([
        np.hstack([np.full_like(t_b, lb[0]), t_b]),
        np.hstack([np.full_like(t_b, ub[0]), t_b]),
    ])
    y_bc = exact_solution(X_bc[:, 0:1], X_bc[:, 1:2])

    x = np.linspace(lb[0], ub[0], 101)
    t = np.linspace(lb[1], ub[1], 101)
    Xg, Tg = np.meshgrid(x, t)
    X_test = np.hstack([Xg.reshape(-1, 1), Tg.reshape(-1, 1)])
    y_test = exact_solution(X_test[:, 0:1], X_test[:, 1:2])

    return {
        "name": NAME, "c": _C, "x0": _X0, "lb": lb, "ub": ub,
        "X_f": X_f, "X_ic": X_ic, "y_ic": y_ic,
        "X_bc": X_bc, "y_bc": y_bc,
        "X_test": X_test, "y_test": y_test,
        "grid_shape": Xg.shape, "x_grid": x, "t_grid": t,
    }


def residual(model, X_f, problem, gradients):
    u = model(X_f)
    grad_u = gradients(u, X_f)
    u_x, u_t = grad_u[:, 0:1], grad_u[:, 1:2]
    grad_u_x = gradients(u_x, X_f)
    u_xx = grad_u_x[:, 0:1]
    u_xxx = gradients(u_xx, X_f)[:, 0:1]
    return u_t + 6.0 * u * u_x + u_xxx
