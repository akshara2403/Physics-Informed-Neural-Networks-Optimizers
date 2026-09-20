"""Statistical helpers for honest cross-seed comparisons.

Both functions are new additions for this project -- the roadmap pilot
script reported only mean/std across seeds. With as few as 2-5 seeds per
condition, mean/std alone can be misleading, so this module adds a
percentile bootstrap confidence interval and a paired Wilcoxon
signed-rank test against a fixed baseline optimizer.

CAVEAT (stated here and repeated in the auto-generated report): with n <
~8 seeds, a Wilcoxon test has very low statistical power, and a bootstrap
CI computed from that few points is itself noisy. These numbers should be
read as "a rough, honestly-computed indication," not as a rigorously
powered significance claim.
"""
import numpy as np
from scipy import stats as scipy_stats


def bootstrap_ci(values, num_resamples=2000, ci=0.95, seed=0):
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(values) == 0:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan"), "n": 0}
    if len(values) == 1:
        return {"mean": float(values[0]), "low": float(values[0]), "high": float(values[0]), "n": 1}

    rng = np.random.default_rng(seed)
    resample_means = np.empty(num_resamples)
    n = len(values)
    for i in range(num_resamples):
        sample = rng.choice(values, size=n, replace=True)
        resample_means[i] = sample.mean()

    alpha = 1.0 - ci
    low = np.percentile(resample_means, 100 * alpha / 2)
    high = np.percentile(resample_means, 100 * (1 - alpha / 2))
    return {"mean": float(values.mean()), "low": float(low), "high": float(high), "n": n}


def paired_wilcoxon(baseline_values, other_values):
    """Paired Wilcoxon signed-rank test, matched by index (i.e. by seed).

    Returns None if there are fewer than 2 finite, non-tied paired
    observations (the test is undefined / uninformative below that).
    """
    baseline_values = np.asarray(baseline_values, dtype=float)
    other_values = np.asarray(other_values, dtype=float)
    n = min(len(baseline_values), len(other_values))
    baseline_values, other_values = baseline_values[:n], other_values[:n]

    mask = np.isfinite(baseline_values) & np.isfinite(other_values)
    baseline_values, other_values = baseline_values[mask], other_values[mask]
    diffs = other_values - baseline_values

    if len(diffs) < 2 or np.all(diffs == 0):
        return None
    try:
        stat, p_value = scipy_stats.wilcoxon(diffs)
    except ValueError:
        return None
    return {
        "n_pairs": int(len(diffs)),
        "statistic": float(stat),
        "p_value": float(p_value),
        "median_diff": float(np.median(diffs)),
        "low_power_warning": len(diffs) < 8,
    }
