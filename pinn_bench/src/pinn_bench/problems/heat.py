"""1D heat equation benchmark (roadmap-original).

u_t = alpha * u_xx,  x in [0,1], t in [0,1]
Exact solution: u(x,t) = exp(-alpha * pi^2 * t) * sin(pi * x)
"""
import numpy as np

NAME = "heat"


def exact_solution(x, t, alpha=0.1):
    return np.exp(-alpha * np.pi**2 * t) * np.sin(np.pi * x)


def make_data(seed, n_f=1500, n_ic=150, n_bc=150, **_):
    rng = np.random.default_rng(seed)
    alpha = 0.1
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])

    x_f = rng.uniform(0, 1, size=(n_f, 1))
    t_f = rng.uniform(0, 1, size=(n_f, 1))
    X_f = np.hstack([x_f, t_f])

    x_i = rng.uniform(0, 1, size=(n_ic, 1))
    t_i = np.zeros_like(x_i)
    X_ic = np.hstack([x_i, t_i])
    y_ic = exact_solution(x_i, t_i, alpha)

    t_b = rng.uniform(0, 1, size=(n_bc // 2, 1))
    X_bc = np.vstack([
        np.hstack([np.zeros_like(t_b), t_b]),
        np.hstack([np.ones_like(t_b), t_b]),
    ])
    y_bc = exact_solution(X_bc[:, 0:1], X_bc[:, 1:2], alpha)

    x = np.linspace(0, 1, 101)
    t = np.linspace(0, 1, 101)
    Xg, Tg = np.meshgrid(x, t)
    X_test = np.hstack([Xg.reshape(-1, 1), Tg.reshape(-1, 1)])
    y_test = exact_solution(X_test[:, 0:1], X_test[:, 1:2], alpha)

    return {
        "name": NAME, "alpha": alpha, "lb": lb, "ub": ub,
        "X_f": X_f, "X_ic": X_ic, "y_ic": y_ic,
        "X_bc": X_bc, "y_bc": y_bc,
        "X_test": X_test, "y_test": y_test,
        "grid_shape": Xg.shape, "x_grid": x, "t_grid": t,
    }


def residual(model, X_f, problem, gradients):
    u = model(X_f)
    grad_u = gradients(u, X_f)
    u_x, u_t = grad_u[:, 0:1], grad_u[:, 1:2]
    u_xx = gradients(u_x, X_f)[:, 0:1]
    return u_t - problem["alpha"] * u_xx
