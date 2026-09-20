"""Viscous Burgers' equation benchmark (roadmap-original), against the
public reference solution from the original PINNs repository
(Raissi, Perdikaris & Karniadakis 2019).

u_t + u*u_x - nu*u_xx = 0,  x in [-1,1], t in [0,1],  nu = 0.01/pi
"""
import os
import urllib.request

import numpy as np
import scipy.io

NAME = "burgers"

# NOTE: the original roadmap URL
#   https://github.com/maziarraissi/PINNs/raw/master/appendix/Data/burgers_shock.mat
# returns 403 from this sandbox's GitHub egress proxy for the `raw/` codepath.
# raw.githubusercontent.com serves the identical bytes directly and is used
# here instead -- a real fix discovered by actually running this code.
_DATA_URL = (
    "https://raw.githubusercontent.com/maziarraissi/PINNs/master/"
    "appendix/Data/burgers_shock.mat"
)


def _download(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "burgers_shock.mat")
    if not os.path.exists(out_path):
        urllib.request.urlretrieve(_DATA_URL, out_path)
    data = scipy.io.loadmat(out_path)
    for key in ("t", "x", "usol"):
        if key not in data:
            raise RuntimeError(f"Burgers reference data missing key: {key}")
    return out_path


def make_data(seed, n_f=1500, n_ic=150, n_bc=150, data_dir="data", **_):
    rng = np.random.default_rng(seed)
    nu = 0.01 / np.pi
    mat_path = _download(data_dir)
    data = scipy.io.loadmat(mat_path)

    t = data["t"].flatten()[:, None]
    x = data["x"].flatten()[:, None]
    Exact = np.real(data["usol"]).T

    X, T = np.meshgrid(x, t)
    X_star = np.hstack([X.reshape(-1, 1), T.reshape(-1, 1)])
    u_star = Exact.reshape(-1, 1)
    lb = X_star.min(axis=0)
    ub = X_star.max(axis=0)

    X_ic_all = np.hstack([X[0:1, :].T, T[0:1, :].T])
    y_ic_all = Exact[0:1, :].T
    idx_ic = rng.choice(X_ic_all.shape[0], size=min(n_ic, X_ic_all.shape[0]), replace=False)
    X_ic, y_ic = X_ic_all[idx_ic], y_ic_all[idx_ic]

    X_bc_all = np.vstack([
        np.hstack([X[:, 0:1], T[:, 0:1]]),
        np.hstack([X[:, -1:], T[:, -1:]]),
    ])
    y_bc_all = np.vstack([Exact[:, 0:1], Exact[:, -1:]])
    idx_bc = rng.choice(X_bc_all.shape[0], size=min(n_bc, X_bc_all.shape[0]), replace=False)
    X_bc, y_bc = X_bc_all[idx_bc], y_bc_all[idx_bc]

    X_f = lb + (ub - lb) * rng.random(size=(n_f, 2))
    X_f = np.vstack([X_f, X_ic, X_bc])

    return {
        "name": NAME, "nu": nu, "lb": lb, "ub": ub,
        "X_f": X_f, "X_ic": X_ic, "y_ic": y_ic,
        "X_bc": X_bc, "y_bc": y_bc,
        "X_test": X_star, "y_test": u_star,
        "grid_shape": X.shape, "x_grid": x.flatten(), "t_grid": t.flatten(),
    }


def residual(model, X_f, problem, gradients):
    u = model(X_f)
    grad_u = gradients(u, X_f)
    u_x, u_t = grad_u[:, 0:1], grad_u[:, 1:2]
    u_xx = gradients(u_x, X_f)[:, 0:1]
    return u_t + u * u_x - problem["nu"] * u_xx
