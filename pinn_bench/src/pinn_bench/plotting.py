"""Plotting. plot_loss_curves / plot_combined_loss_curves / plot_prediction
are roadmap-original; plot_summary is extended with a runtime-normalized
error panel, and plot_gradient_conflict is a new diagnostic plot.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(all_histories, figures_dir):
    for key, hist in all_histories.items():
        problem, optimizer, seed = key
        if len(hist) == 0:
            continue
        plt.figure()
        plt.plot(range(len(hist)), hist["total_loss"])
        plt.yscale("log")
        plt.xlabel("Logged step index")
        plt.ylabel("Total loss (log scale)")
        plt.title(f"Loss Curve: {problem} | {optimizer} | seed {seed}")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"loss_{problem}_{optimizer}_seed{seed}.png"), dpi=150)
        plt.close()


def plot_combined_loss_curves(all_histories, figures_dir):
    keys = list(all_histories.keys())
    problem_seed_pairs = sorted(set((k[0], k[2]) for k in keys))
    for problem, seed in problem_seed_pairs:
        plt.figure()
        for (p, opt, s), hist in all_histories.items():
            if p == problem and s == seed and len(hist) > 0:
                plt.plot(range(len(hist)), hist["total_loss"], label=opt)
        plt.yscale("log")
        plt.xlabel("Logged step index")
        plt.ylabel("Total loss (log scale)")
        plt.title(f"Combined Loss Curves: {problem} | seed {seed}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"combined_loss_{problem}_seed{seed}.png"), dpi=150)
        plt.close()


def plot_gradient_conflict(all_histories, figures_dir):
    keys = list(all_histories.keys())
    problem_seed_pairs = sorted(set((k[0], k[2]) for k in keys))
    for problem, seed in problem_seed_pairs:
        plt.figure()
        any_data = False
        for (p, opt, s), hist in all_histories.items():
            if p == problem and s == seed and len(hist) > 0 and "grad_conflict_cosine" in hist:
                sub = hist.dropna(subset=["grad_conflict_cosine"])
                if len(sub) > 0:
                    plt.plot(sub["step"], sub["grad_conflict_cosine"], marker="o", markersize=3, label=opt)
                    any_data = True
        if not any_data:
            plt.close()
            continue
        plt.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
        plt.xlabel("Step")
        plt.ylabel("cos(grad PDE, grad IC+BC)")
        plt.title(f"Gradient Conflict: {problem} | seed {seed}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"grad_conflict_{problem}_seed{seed}.png"), dpi=150)
        plt.close()


def plot_summary(all_results_df, figures_dir, csv_dir):
    summary = all_results_df.groupby(["problem", "optimizer"]).agg(
        mean_gap=("relative_solution_gap", "mean"),
        std_gap=("relative_solution_gap", "std"),
        mean_runtime=("runtime_seconds", "mean"),
        mean_residual=("pde_residual_mse", "mean"),
        mean_runtime_normalized_error=("runtime_normalized_error", "mean"),
        n_runs=("seed", "count"),
        n_failures=("failure_flag", lambda x: int(np.sum(x != "ok"))),
    ).reset_index()
    summary.to_csv(os.path.join(csv_dir, "summary_by_problem_optimizer.csv"), index=False)

    for problem in all_results_df["problem"].unique():
        dfp = all_results_df[all_results_df["problem"] == problem]

        plt.figure()
        dfp.boxplot(column="relative_solution_gap", by="optimizer")
        plt.title(f"Seed Variability: {problem}")
        plt.suptitle("")
        plt.ylabel("Relative solution gap")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"boxplot_gap_{problem}.png"), dpi=150)
        plt.close()

        plt.figure()
        dfp.groupby("optimizer")["runtime_seconds"].mean().plot(kind="bar")
        plt.ylabel("Runtime seconds")
        plt.title(f"Mean Runtime: {problem}")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"runtime_{problem}.png"), dpi=150)
        plt.close()

        plt.figure()
        dfp.groupby("optimizer")["pde_residual_mse"].mean().plot(kind="bar")
        plt.ylabel("PDE residual MSE")
        plt.yscale("log")
        plt.title(f"Mean PDE Residual: {problem}")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"residual_{problem}.png"), dpi=150)
        plt.close()

        plt.figure()
        dfp.groupby("optimizer")["runtime_normalized_error"].mean().plot(kind="bar")
        plt.ylabel("Runtime-normalized error (gap * sqrt(runtime))")
        plt.title(f"Runtime-Normalized Error: {problem}")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"runtime_normalized_{problem}.png"), dpi=150)
        plt.close()

    return summary


def plot_prediction(problem, y_pred, optimizer_name, seed, figures_dir):
    y_true = problem["y_test"].reshape(-1, 1)
    error = np.abs(y_true - y_pred)
    shape = problem["grid_shape"]
    U_pred = y_pred.reshape(shape)
    U_err = error.reshape(shape)
    name = problem["name"]

    plt.figure()
    plt.imshow(U_pred, aspect="auto", origin="lower")
    plt.colorbar(label="Predicted u")
    plt.title(f"Predicted: {name} | {optimizer_name} | seed {seed}")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"pred_{name}_{optimizer_name}_seed{seed}.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.imshow(U_err, aspect="auto", origin="lower")
    plt.colorbar(label="Absolute error")
    plt.title(f"Absolute Error: {name} | {optimizer_name} | seed {seed}")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"error_{name}_{optimizer_name}_seed{seed}.png"), dpi=150)
    plt.close()
