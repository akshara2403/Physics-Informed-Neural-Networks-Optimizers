import numpy as np
import pytest
import sympy as sp
import torch

from pinn_bench.autodiff import gradients, to_tensor
from pinn_bench.problems import heat, burgers, allen_cahn, kdv


def test_kdv_soliton_solves_pde_symbolically():
    x, t, c, x0 = sp.symbols("x t c x0", real=True, positive=False)
    u = (c / 2) / sp.cosh(sp.sqrt(c) / 2 * (x - c * t - x0)) ** 2
    residual = sp.diff(u, t) + 6 * u * sp.diff(u, x) + sp.diff(u, x, 3)
    residual_at_params = residual.subs({c: kdv._C, x0: kdv._X0})
    # Check the symbolic residual vanishes at several (x, t) points.
    for xv in (-3.0, 0.0, 2.5):
        for tv in (0.0, 0.5, 1.0):
            val = float(residual_at_params.subs({x: xv, t: tv}))
            assert abs(val) < 1e-6


def test_kdv_soliton_solves_pde_numerically():
    xs = np.linspace(-10, 10, 25)
    ts = np.linspace(0, 1, 5)
    Xg, Tg = np.meshgrid(xs, ts)
    X = np.hstack([Xg.reshape(-1, 1), Tg.reshape(-1, 1)])
    X_t = to_tensor(X, "cpu", torch.float64, requires_grad=True)

    def exact_u(Xin):
        x_, t_ = Xin[:, 0:1], Xin[:, 1:2]
        c, x0 = kdv._C, kdv._X0
        arg = torch.sqrt(torch.tensor(c, dtype=Xin.dtype)) / 2.0 * (x_ - c * t_ - x0)
        return (c / 2.0) / torch.cosh(arg) ** 2

    u = exact_u(X_t)
    grad_u = gradients(u, X_t)
    u_x, u_t = grad_u[:, 0:1], grad_u[:, 1:2]
    u_xx = gradients(u_x, X_t)[:, 0:1]
    u_xxx = gradients(u_xx, X_t)[:, 0:1]
    residual = u_t + 6.0 * u * u_x + u_xxx
    assert torch.max(torch.abs(residual)).item() < 1e-4


def test_heat_exact_solution_solves_pde_numerically():
    xs = np.linspace(0.05, 0.95, 15)
    ts = np.linspace(0.05, 0.95, 15)
    Xg, Tg = np.meshgrid(xs, ts)
    X = np.hstack([Xg.reshape(-1, 1), Tg.reshape(-1, 1)])
    X_t = to_tensor(X, "cpu", torch.float64, requires_grad=True)

    alpha = 0.1
    x_, t_ = X_t[:, 0:1], X_t[:, 1:2]
    u = torch.exp(-alpha * np.pi**2 * t_) * torch.sin(np.pi * x_)
    grad_u = gradients(u, X_t)
    u_x, u_t = grad_u[:, 0:1], grad_u[:, 1:2]
    u_xx = gradients(u_x, X_t)[:, 0:1]
    residual = u_t - alpha * u_xx
    assert torch.max(torch.abs(residual)).item() < 1e-6


def test_allen_cahn_reference_shapes():
    x_ref, t_ref, Exact = allen_cahn._generate_reference(nx=32, nt=21)
    assert x_ref.shape == (32,)
    assert t_ref.shape == (21,)
    assert Exact.shape == (21, 32)
    assert np.all(np.isfinite(Exact))


def test_kdv_make_data_shapes():
    problem = kdv.make_data(seed=0, n_f=50, n_ic=10, n_bc=10)
    assert problem["X_f"].shape == (50, 2)
    assert problem["X_ic"].shape == (10, 2)
    assert problem["X_bc"].shape[1] == 2
    assert problem["y_test"].shape[0] == problem["X_test"].shape[0]
