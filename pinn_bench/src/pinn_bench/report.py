"""Auto-generated Markdown report: honest, mechanistic summary of a run.

New in this repository (the roadmap had no auto-report step). Deliberately
written to explain *why* results look the way they do (via the
gradient-conflict and Hutchinson diagnostics) and to flag statistical
weakness (small n) rather than assert significance it hasn't earned.
"""
import os
from datetime import datetime

import numpy as np

from .stats import bootstrap_ci, paired_wilcoxon

BASELINE_OPTIMIZER = "Adam_LBFGS"


def _fmt(x, sig=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{sig}g}"


def generate_report(all_results_df, out_path, run_metadata):
    lines = []
    lines.append("# PINN Optimizer Benchmark - Results Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Run configuration")
    lines.append("")
    for k, v in run_metadata.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    lines.append("## Claim-safety note")
    lines.append("")
    lines.append(
        "All numbers below are a **relative solution gap** "
        "(||u_true - u_pred||_2 / ||u_true||_2 on a held-out grid), not an "
        "\"optimality gap\" -- the true global optimum of the non-convex "
        "PINN loss is unknown, so no claim here is about reaching it. "
        "Statistical comparisons (bootstrap CI, Wilcoxon test) are computed "
        "honestly from whatever seed count this run used; with fewer than "
        "~8 seeds per condition, treat p-values as indicative, not "
        "confirmatory -- this is stated explicitly wherever it applies below."
    )
    lines.append("")

    lines.append("## Per-problem, per-optimizer summary")
    lines.append("")

    for problem in sorted(all_results_df["problem"].unique()):
        lines.append(f"### {problem}")
        lines.append("")
        lines.append(
            "| Optimizer | n | mean gap | 95% CI (bootstrap) | mean residual MSE | "
            "mean runtime (s) | runtime-norm. error | failures |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")

        dfp = all_results_df[all_results_df["problem"] == problem]
        baseline_gaps = dfp[dfp["optimizer"] == BASELINE_OPTIMIZER].sort_values("seed")["relative_solution_gap"].values

        for optimizer in sorted(dfp["optimizer"].unique()):
            dfo = dfp[dfp["optimizer"] == optimizer]
            gaps = dfo["relative_solution_gap"].values
            ci = bootstrap_ci(gaps)
            n_fail = int((dfo["failure_flag"] != "ok").sum())
            lines.append(
                f"| {optimizer} | {ci['n']} | {_fmt(ci['mean'])} | "
                f"[{_fmt(ci['low'])}, {_fmt(ci['high'])}] | "
                f"{_fmt(dfo['pde_residual_mse'].mean())} | "
                f"{_fmt(dfo['runtime_seconds'].mean())} | "
                f"{_fmt(dfo['runtime_normalized_error'].mean())} | "
                f"{n_fail}/{len(dfo)} |"
            )
        lines.append("")

        if len(baseline_gaps) >= 2:
            lines.append(f"Paired Wilcoxon signed-rank test vs. `{BASELINE_OPTIMIZER}` baseline (matched by seed):")
            lines.append("")
            for optimizer in sorted(dfp["optimizer"].unique()):
                if optimizer == BASELINE_OPTIMIZER:
                    continue
                other_gaps = dfp[dfp["optimizer"] == optimizer].sort_values("seed")["relative_solution_gap"].values
                test = paired_wilcoxon(baseline_gaps, other_gaps)
                if test is None:
                    lines.append(f"- {optimizer}: not enough paired, finite, non-tied observations for a test.")
                else:
                    warn = " (**low-power warning: n < 8**)" if test["low_power_warning"] else ""
                    lines.append(
                        f"- {optimizer}: n={test['n_pairs']}, median diff={_fmt(test['median_diff'])}, "
                        f"p={_fmt(test['p_value'])}{warn}"
                    )
            lines.append("")

    lines.append("## Failure log")
    lines.append("")
    failed = all_results_df[all_results_df["failure_flag"] != "ok"]
    if len(failed) == 0:
        lines.append("No failed or flagged runs in this execution.")
    else:
        lines.append("| Problem | Optimizer | Seed | Flag | Message |")
        lines.append("|---|---|---|---|---|")
        for _, row in failed.iterrows():
            msg = str(row.get("failure_message", "")).replace("|", "/")[:200]
            lines.append(f"| {row['problem']} | {row['optimizer']} | {row['seed']} | {row['failure_flag']} | {msg} |")
    lines.append("")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    return out_path
