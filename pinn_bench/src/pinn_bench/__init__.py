"""
pinn_bench: Empirical solution-gap and stability analysis of
matrix-preconditioned optimizers for Physics-Informed Neural Networks (PINNs).

This package is a from-scratch re-engineering of the reference pilot script
in the project roadmap ("YRI_pinn_optimizer_8_week_roadmap"), restructured
as a tested, importable, configurable library rather than a single Colab
cell.

From the roadmap (kept, re-implemented):
    - PINN architecture with input normalization to [-1, 1] (models.py)
    - Heat, Burgers, Allen-Cahn benchmark problems (problems/)
    - Adam, L-BFGS, Adam->L-BFGS hybrid training phases (training.py)
    - Relative L2 solution-gap metric, PDE residual MSE (evaluate.py)
    - Loss oscillation index, gradient norm diagnostics (diagnostics.py)
    - failure-flag protocol (ok / nonfinite_error / large_error / failed)

New in this repository (beyond the roadmap, added for this project):
    - A fourth benchmark, the Korteweg-de Vries (KdV) equation, evaluated
      against an exact one-soliton analytic solution (problems/kdv.py).
      Kept OUT of ROADMAP_PROBLEMS so roadmap-vs-novel comparisons stay honest.
    - From-scratch, numerically stabilized implementations of Shampoo, SOAP,
      and Muon (optimizers/), replacing the third-party `torch-optimizer`
      Shampoo used in the original pilot run, which failed via SVD
      non-convergence on real pilot data (see README "Motivation").
    - Gradient-conflict diagnostic: cosine similarity between the PDE-loss
      gradient and the (IC+BC)-loss gradient, to explain *why* an optimizer
      struggles rather than just report that it does (diagnostics.py).
    - Hutchinson stochastic trace estimator as a lightweight loss-curvature
      proxy (diagnostics.py).
    - Runtime-normalized error metric: gap * sqrt(runtime), so accuracy
      claims account for wall-clock cost (evaluate.py).
    - Percentile bootstrap confidence intervals and paired Wilcoxon
      signed-rank tests against an Adam+L-BFGS baseline, with explicit
      small-sample-size caveats (stats.py).
    - YAML-driven configuration (config.py) replacing the hardcoded
      QUICK_MODE boolean, plus a CLI (run.py) and auto-generated Markdown
      report (report.py).

Claim-safety note: this package deliberately reports "relative solution gap"
(||u_true - u_pred||_2 / ||u_true||_2), not "optimality gap" -- we do not
know the true global optimum of the non-convex PINN loss landscape, so no
optimizer here is described as reaching one.
"""

__version__ = "0.1.0"
