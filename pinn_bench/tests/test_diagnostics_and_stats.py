import numpy as np
import torch

from pinn_bench.diagnostics import gradient_conflict_cosine, hutchinson_trace_estimate
from pinn_bench.stats import bootstrap_ci, paired_wilcoxon


def test_gradient_conflict_cosine_opposing_losses():
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    model = torch.nn.ParameterList([w])
    model.parameters = lambda: iter([w])  # minimal stand-in with .parameters()

    loss_a = (w[0] - 5.0) ** 2  # wants w[0] to increase from 1
    loss_b = (w[0] + 5.0) ** 2  # wants w[0] to decrease from 1
    cos = gradient_conflict_cosine(model, loss_a, loss_b)
    assert cos < -0.9  # nearly perfectly opposing gradients in this 1D-active case


def test_gradient_conflict_cosine_aligned_losses():
    w = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    model = torch.nn.ParameterList([w])
    model.parameters = lambda: iter([w])

    loss_a = (w[0] - 5.0) ** 2
    loss_b = 2.0 * (w[0] - 5.0) ** 2  # same direction, different scale
    cos = gradient_conflict_cosine(model, loss_a, loss_b)
    assert cos > 0.9


def test_hutchinson_trace_matches_exact_trace_of_diagonal_quadratic():
    torch.manual_seed(0)
    diag = torch.tensor([2.0, 4.0, 6.0, 8.0])
    w = torch.nn.Parameter(torch.zeros(4))
    model = torch.nn.ParameterList([w])
    model.parameters = lambda: iter([w])

    loss = 0.5 * torch.sum(diag * w ** 2)  # Hessian = diag(diag), true trace = sum(diag)
    estimate = hutchinson_trace_estimate(model, loss, num_samples=500)
    true_trace = diag.sum().item()
    assert abs(estimate - true_trace) / true_trace < 0.15  # stochastic estimator, allow slack


def test_bootstrap_ci_reasonable_on_known_distribution():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=1.0, scale=0.1, size=30)
    ci = bootstrap_ci(values, num_resamples=1000, seed=1)
    assert ci["low"] < 1.0 < ci["high"]
    assert ci["n"] == 30


def test_bootstrap_ci_handles_empty_and_singleton():
    assert np.isnan(bootstrap_ci([])["mean"])
    single = bootstrap_ci([3.0])
    assert single["mean"] == 3.0 and single["low"] == 3.0 and single["high"] == 3.0


def test_paired_wilcoxon_detects_consistent_difference():
    baseline = [1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.03]
    other = [0.5, 0.55, 0.45, 0.52, 0.48, 0.51, 0.49, 0.53]  # consistently smaller
    result = paired_wilcoxon(baseline, other)
    assert result is not None
    assert result["p_value"] < 0.05
    assert result["median_diff"] < 0


def test_paired_wilcoxon_none_on_too_few_points():
    assert paired_wilcoxon([1.0], [2.0]) is None
