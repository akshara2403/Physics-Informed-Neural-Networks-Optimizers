import torch

from .shampoo import StableShampoo
from .soap import SOAP
from .muon import Muon

# Optimizers that appear in the original roadmap pilot script.
ROADMAP_OPTIMIZERS = ("Adam", "LBFGS", "Adam_LBFGS")
# Matrix-preconditioned optimizers this project studies. "Shampoo" here is
# StableShampoo (see optimizers/shampoo.py docstring for why it replaces the
# third-party torch-optimizer Shampoo used in the original pilot).
PRECONDITIONED_OPTIMIZERS = ("Shampoo", "SOAP", "Muon")
ALL_OPTIMIZERS = ROADMAP_OPTIMIZERS + PRECONDITIONED_OPTIMIZERS


def build_first_order_optimizer(name: str, parameters, hparams: dict):
    """Build the *first-order* (Adam-like) optimizer for a given name.

    LBFGS is handled separately in training.py since it needs a closure.
    "Adam_LBFGS" uses this factory for its Adam pretrain phase.
    """
    hparams = hparams or {}
    if name in ("Adam", "Adam_LBFGS"):
        return torch.optim.Adam(parameters, lr=hparams.get("lr", 1e-3))
    if name == "Shampoo":
        return StableShampoo(
            parameters,
            lr=hparams.get("lr", 1e-2),
            momentum=hparams.get("momentum", 0.9),
            epsilon=hparams.get("epsilon", 1e-6),
            update_freq=hparams.get("update_freq", 10),
        )
    if name == "SOAP":
        return SOAP(
            parameters,
            lr=hparams.get("lr", 3e-3),
            update_freq=hparams.get("update_freq", 10),
        )
    if name == "Muon":
        return Muon(
            parameters,
            lr=hparams.get("lr", 2e-2),
            momentum=hparams.get("momentum", 0.95),
            adam_lr=hparams.get("adam_lr", 1e-3),
        )
    raise ValueError(f"Unknown or non-first-order optimizer name: {name}")
