"""YAML-driven run configuration, replacing the roadmap's hardcoded
QUICK_MODE boolean with an inspectable, versionable config file plus CLI
overrides.
"""
import dataclasses
from typing import Optional

import yaml

DEFAULTS = {
    "seeds": [0, 1],
    "problems": ["heat", "burgers"],
    "optimizers": ["Adam", "LBFGS", "Adam_LBFGS", "Shampoo", "SOAP", "Muon"],
    "adam_steps": 1500,
    "shampoo_steps": 1500,
    "hybrid_adam_steps": 1000,
    "lbfgs_max_iter": 300,
    "n_f": 1500,
    "n_ic": 150,
    "n_bc": 150,
    "results_dir": "results",
    "hparams": {},
}


@dataclasses.dataclass
class RunConfig:
    seeds: list
    problems: list
    optimizers: list
    adam_steps: int
    shampoo_steps: int
    hybrid_adam_steps: int
    lbfgs_max_iter: int
    n_f: int
    n_ic: int
    n_bc: int
    results_dir: str
    hparams: dict

    @property
    def steps_cfg(self) -> dict:
        return {
            "adam_steps": self.adam_steps,
            "shampoo_steps": self.shampoo_steps,
            "hybrid_adam_steps": self.hybrid_adam_steps,
            "lbfgs_max_iter": self.lbfgs_max_iter,
        }

    def problem_kwargs(self) -> dict:
        return {"n_f": self.n_f, "n_ic": self.n_ic, "n_bc": self.n_bc}


def load_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> RunConfig:
    cfg = dict(DEFAULTS)
    if path is not None:
        with open(path) as f:
            file_cfg = yaml.safe_load(f) or {}
        cfg.update(file_cfg)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return RunConfig(**cfg)
