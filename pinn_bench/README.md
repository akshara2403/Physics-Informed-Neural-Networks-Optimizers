# pinn_bench

Empirical solution-gap and stability analysis of matrix-preconditioned
optimizers for Physics-Informed Neural Networks (PINNs), extending the
project's 8-week roadmap ("Empirical Solution-Gap and Stability Analysis of
Matrix-Preconditioned Optimizers for PINNs") into a tested, configurable,
reproducible Python package.

Full results from the run actually executed in this repository's
environment are in [`RESULTS.md`](RESULTS.md) and `results/full/REPORT.md`
/ `results/full/csv/`.

## What's from the roadmap (kept, re-implemented)

- The PINN architecture: a tanh MLP with inputs normalized to [-1, 1]
  (`src/pinn_bench/models.py`).
- Three benchmark PDEs: 1D heat (exact solution), viscous Burgers'
  (reference solution from Raissi, Perdikaris & Karniadakis 2019), and
  Allen-Cahn (finite-difference/BDF reference solution) (`problems/`).
- Adam, L-BFGS, and Adam-then-L-BFGS hybrid training (`training.py`).
- The relative-L2 solution-gap metric and PDE residual MSE
  (`evaluate.py`), loss-oscillation index and gradient-norm diagnostics
  (`diagnostics.py`), and the ok/nonfinite_error/large_error/failed
  failure-flag protocol (kept so a single bad run never crashes a sweep).
- Shampoo-style matrix preconditioning as a fourth optimizer to compare
  against Adam/L-BFGS.

## What's new in this repository (beyond the roadmap)

- **A fourth benchmark: the Korteweg-de Vries (KdV) equation**
  (`problems/kdv.py`), evaluated against an exact one-soliton analytic
  solution, verified both symbolically (SymPy) and numerically in
  `tests/test_problems.py`. KdV's residual needs a third-order spatial
  derivative (`u_xxx`), which none of the three roadmap benchmarks
  exercise -- and third/higher-order-derivative loss landscapes are
  exactly where matrix-preconditioned optimizers are usually motivated, so
  this is a more discriminating test of the optimizers this project
  studies. KdV is deliberately excluded from `ROADMAP_PROBLEMS` in
  `problems/__init__.py` so roadmap-vs-novel comparisons stay honest.
- **From-scratch, numerically stabilized Shampoo, plus new SOAP and Muon
  implementations** (`optimizers/`). See "Motivation" below for why
  Shampoo needed rewriting.
- **Gradient-conflict diagnostic**: cosine similarity between
  grad(PDE loss) and grad(IC+BC loss) at each logged step
  (`diagnostics.gradient_conflict_cosine`). This explains *why* a PINN
  training run stalls or oscillates -- a negative cosine means driving down
  the PDE residual is actively fighting the boundary/initial-condition fit
  -- rather than just reporting that it does.
- **Hutchinson stochastic trace estimator** (`diagnostics.hutchinson_trace_estimate`)
  as a cheap loss-curvature proxy, logged alongside the loss curves.
- **Runtime-normalized error** (`evaluate.runtime_normalized_error`):
  `gap * sqrt(runtime)`, so an optimizer that buys a marginally smaller gap
  at several times the wall-clock cost doesn't look strictly better in a
  raw accuracy table.
- **Percentile bootstrap confidence intervals and paired Wilcoxon
  signed-rank tests** against an Adam+L-BFGS baseline (`stats.py`),
  computed honestly from however many seeds a run used, with an explicit
  low-statistical-power warning below 8 seeds -- the roadmap script only
  reported mean/std across seeds.
- **YAML-driven configuration** (`config.py`, `configs/quick.yaml`,
  `configs/full.yaml`) replacing the roadmap's hardcoded `QUICK_MODE`
  boolean, a CLI (`run.py` / `scripts/run_experiment.py`), an
  auto-generated Markdown report (`report.py`), and a pytest suite
  (`tests/`).

## Claim-safety

This project reports a **relative solution gap**
(`||u_true - u_pred||_2 / ||u_true||_2`), never an "optimality gap" -- the
true global optimum of a non-convex PINN loss is unknown, so no result
here claims to have reached one. Statistical comparisons state their own
sample size and are flagged as low-power below 8 seeds rather than
presented as significance claims they haven't earned. Every run that fails
or produces a flagged (non-finite or unusually large) error is kept in the
results table and logged in the report's failure log, not silently
dropped.

## Motivation: why Shampoo needed rewriting

The original roadmap pilot run used the third-party `torch-optimizer`
package's `Shampoo` implementation. On real pilot data it failed on 3 of 4
runs with:

```
linalg.svd: The algorithm failed to converge because the input matrix
is ill-conditioned or has too many repeated singular values (error code: 14)
```

and a 4th run hit a non-finite loss. `StableShampoo`
(`optimizers/shampoo.py`) fixes the root cause: its preconditioners are
symmetric PSD by construction, so their inverse fourth roots are computed
via `torch.linalg.eigh` instead of SVD, with an eigenvalue floor. Actually
running that first version still diverged on a toy regression test (loss
4.7 -> 7000+ in 10 steps, see the docstring in `shampoo.py` for the full
story) because a near-rank-deficient early preconditioner got amplified by
a factor of `eps^-1/4`; the fix was a trust-region clip on the
preconditioned update plus a short warmup before the preconditioner is
trusted at all. Both bugs were caught by actually executing the code and
its test suite, not by inspection.

## This was actually run (not just written)

Every optimizer, every benchmark problem, the full pytest suite, and the
CLI were executed end to end in the environment this repository ships
from -- twice, at two different scales:

1. `configs/quick.yaml` (24-120 runs, small step counts) as an
   integration smoke test.
2. `configs/full.yaml` (4 problems x 6 optimizers x 5 seeds = 120 runs) as
   the extended sweep whose results are reported in `RESULTS.md`.

Bugs the smoke test actually caught and that are fixed in this code (not
worked around):

- **Burgers reference-data URL returned HTTP 403** from
  `github.com/.../raw/...` in this sandbox's network egress; fixed by
  pointing `problems/burgers.py` at the equivalent
  `raw.githubusercontent.com` URL, which serves identical bytes directly.
- **`StableShampoo` diverged on its own optimizer unit test** the first
  time it was actually run against real gradients (see "Motivation"
  above) -- fixed with a trust-region clip and warmup, not by loosening
  the test.
- **The gradient-conflict diagnostic crashed with a tensor-size mismatch**
  the first time it ran inside an actual training loop: a PINN's
  final-layer bias has zero gradient with respect to the PDE loss (a
  differentiation-w.r.t.-input identity makes it disappear from the
  residual), so `torch.autograd.grad(..., allow_unused=True)` returned
  `None` there for one loss but not the other, and dropping `None`
  entries independently misaligned the two flattened gradient vectors.
  Fixed by zero-filling instead of dropping.
- **The Newton-Schulz orthogonalization unit test was wrong, not the
  optimizer**: the published quintic iteration's coefficients have fixed
  points near 0.87 and 1.26, not exactly 1, and the map is chaotic outside
  a narrow basin -- so "run it for more steps and check near-exact
  orthogonality" is not a valid test of Muon's intended (bounded, not
  exact) behavior. The test was rewritten to check the actual documented
  guarantee (singular values stay within a bounded range at the
  recommended step count) instead of an invalid stronger one.
- **Allen-Cahn is genuinely hard for every optimizer at the quick-mode
  step budget** (relative gap close to or above 1.0 for all six
  optimizers in the smoke test) -- consistent with this PDE's known
  stiffness and the PINN literature's use of specialized training
  strategies (causal/curriculum weighting, time-marching) for it, not a
  bug. `configs/full.yaml` raises the step budget, and `RESULTS.md`
  reports what that did and did not fix.

## Repository layout

```
src/pinn_bench/
  models.py            PINN architecture
  autodiff.py           gradients() / to_tensor() helpers
  problems/             heat, burgers, allen_cahn, kdv
  optimizers/           StableShampoo, SOAP, Muon, factory
  diagnostics.py        grad norm, oscillation index, gradient conflict, Hutchinson trace
  evaluate.py            relative gap, residual MSE, runtime-normalized error
  stats.py               bootstrap CI, paired Wilcoxon
  training.py            per-run training loop
  plotting.py             loss curves, summary plots, gradient-conflict plot
  report.py               auto-generated Markdown report
  config.py               YAML config + CLI overrides
  run.py                  sweep orchestration / CLI entry point
configs/                 quick.yaml, full.yaml
scripts/run_experiment.py
tests/                    pytest suite (problems, optimizers, diagnostics/stats)
notebooks/colab_quickstart.ipynb
docs/lit_survey_intro.md  literature survey + manuscript introduction draft
RESULTS.md                results and analysis from the run actually executed here
```

## Running it

```bash
pip install -e .
python -m pytest tests/ -q
python scripts/run_experiment.py --config configs/quick.yaml   # ~2-3 min, CPU
python scripts/run_experiment.py --config configs/full.yaml    # ~20-30 min, CPU
```

Override anything from the CLI, e.g.:

```bash
python scripts/run_experiment.py --config configs/quick.yaml --problems heat kdv --seeds 0 1 2
```

## Known limitations / honestly still open

- All reported runs are CPU-only (2 cores); `configs/full.yaml`'s step
  counts were chosen for that budget, not for maximum achievable accuracy.
  A GPU run with the roadmap's original 5000-step budget would likely
  improve every optimizer's absolute gap.
- Muon's default learning rate (2e-2) diverged on Allen-Cahn during
  development and was lowered to 1e-2 in both configs; it has not been
  independently tuned per problem the way a fully rigorous comparison
  would require, and `RESULTS.md` says so explicitly wherever it matters.
- SOAP and Muon here are independent, simplified from-scratch
  reimplementations inspired by their respective papers, not ports of the
  reference author code -- see the docstrings in `optimizers/soap.py` and
  `optimizers/muon.py` for exactly what's simplified.
- With 5 seeds per condition, the paired Wilcoxon tests in `RESULTS.md`
  are explicitly flagged as low-power; treat them as indicative.
