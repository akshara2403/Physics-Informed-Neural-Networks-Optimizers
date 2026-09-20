"""Allen-Cahn equation benchmark (roadmap-original), a stiff reaction-
diffusion PDE with a documented finite-difference reference solution.

u_t = d*u_xx + 5*(u - u^3),  x in [-1,1) periodic, t in [0,1]
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

NAME = "allen_cahn"


def _generate_reference(nx=128, nt=101, d=1e-3):
    x = np.linspace(-1, 1, nx, endpoint=False)
    t_eval = np.linspace(0, 1, nt)
    dx = x[1] - x[0]
    u0 = (x**2) * np.cos(np.pi * x)

    def rhs(t, u):
        u_xx = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2
        return d * u_xx + 5.0 * (u - u**3)

    sol = solve_ivp(rhs, (0.0, 1.0), u0, t_eval=t_eval, method="BDF", rtol=1e-7, atol=1e-8)
    if not sol.success:
        raise RuntimeError("Allen-Cahn reference solver failed: " + sol.message)
    return x, t_eval, sol.y.T


def make_data(seed, n_f=1500, n_ic=150, n_bc=150, **_):
    rng = np.random.default_rng(seed)
    d = 1e-3
    lb = np.array([-1.0, 0.0])
    ub = np.array([1.0, 1.0])

    x_ref, t_ref, Exact = _generate_reference(d=d)
    Xg, Tg = np.meshgrid(x_ref, t_ref)
    X_test = np.hstack([Xg.reshape(-1, 1), Tg.reshape(-1, 1)])
    y_test = Exact.reshape(-1, 1)

    x_f = rng.uniform(-1, 1, size=(n_f, 1))
    t_f = rng.uniform(0, 1, size=(n_f, 1))
    X_f = np.hstack([x_f, t_f])

    x_i = rng.uniform(-1, 1, size=(n_ic, 1))
    t_i = np.zeros_like(x_i)
    X_ic = np.hstack([x_i, t_i])
    y_ic = (x_i**2) * np.cos(np.pi * x_i)

    left_boundary_over_time = Exact[:, 0]
    boundary_interp = interp1d(t_ref, left_boundary_over_time, kind="cubic", fill_value="extrapolate")
    t_b = rng.uniform(0, 1, size=(n_bc // 2, 1))
    X_bc = np.vstack([
        np.hstack([-np.ones_like(t_b), t_b]),
        np.hstack([np.ones_like(t_b), t_b]),
    ])
    yb = boundary_interp(t_b.flatten()).reshape(-1, 1)
    y_bc = np.vstack([yb, yb])

    X_f = np.vstack([X_f, X_ic, X_bc])

    return {
        "name": NAME, "d": d, "lb": lb, "ub": ub,
        "X_f": X_f, "X_ic": X_ic, "y_ic": y_ic,
        "X_bc": X_bc, "y_bc": y_bc,
        "X_test": X_test, "y_test": y_test,
        "grid_shape": Xg.shape, "x_grid": x_ref, "t_grid": t_ref,
    }


def residual(model, X_f, problem, gradients):
    u = model(X_f)
    grad_u = gradients(u, X_f)
    u_x, u_t = grad_u[:, 0:1], grad_u[:, 1:2]
    u_xx = gradients(u_x, X_f)[:, 0:1]
    return u_t - problem["d"] * u_xx - 5.0 * (u - u**3)
