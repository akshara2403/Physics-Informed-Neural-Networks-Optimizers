"""Top-level orchestration: reads a config, loops over problems x optimizers
x seeds, catches per-run failures without crashing the whole sweep (roadmap-
original failure-handling behavior, kept), writes CSVs/plots/metadata, an
auto-generated report, and zips the results directory.
"""
import argparse
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from .config import load_config
from .plotting import (
    plot_combined_loss_curves,
    plot_gradient_conflict,
    plot_loss_curves,
    plot_prediction,
    plot_summary,
)
from .problems import PROBLEMS
from .report import generate_report
from .training import train_one_run


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Run the PINN optimizer benchmark.")
    p.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--problems", type=str, nargs="+", default=None)
    p.add_argument("--optimizers", type=str, nargs="+", default=None)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    return p.parse_args(argv)


def _results_subdirs(results_dir):
    figures_dir = os.path.join(results_dir, "figures")
    csv_dir = os.path.join(results_dir, "csv")
    data_dir = os.path.join(results_dir, "data")
    for d in (figures_dir, csv_dir, data_dir):
        os.makedirs(d, exist_ok=True)
    return figures_dir, csv_dir, data_dir


def _device(requested):
    if requested is not None:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    overrides = {
        "results_dir": args.results_dir,
        "seeds": args.seeds,
        "problems": args.problems,
        "optimizers": args.optimizers,
    }
    cfg = load_config(args.config, overrides)
    device = _device(args.device)
    dtype = torch.float32

    figures_dir, csv_dir, data_dir = _results_subdirs(cfg.results_dir)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "seeds": cfg.seeds,
        "problems": cfg.problems,
        "optimizers": cfg.optimizers,
        "steps_cfg": cfg.steps_cfg,
        "problem_kwargs": cfg.problem_kwargs(),
        "config_path": args.config,
    }
    with open(os.path.join(cfg.results_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Problems:", cfg.problems)
    print("Optimizers:", cfg.optimizers)
    print("Seeds:", cfg.seeds)
    print("Device:", device)

    all_results = []
    all_histories = {}
    sweep_start = time.time()

    for problem_name in cfg.problems:
        if problem_name not in PROBLEMS:
            raise ValueError(f"Unknown problem: {problem_name}")
        for seed in cfg.seeds:
            problem = PROBLEMS[problem_name].make_data(seed, data_dir=data_dir, **cfg.problem_kwargs())
            for optimizer_name in cfg.optimizers:
                tag = f"problem={problem_name} optimizer={optimizer_name} seed={seed}"
                print(f"\nRunning {tag}")
                t0 = time.time()
                try:
                    result, history, y_pred = train_one_run(
                        problem, problem_name, optimizer_name, seed,
                        cfg.steps_cfg, cfg.hparams, device, dtype,
                    )
                    all_results.append(result)
                    all_histories[(problem_name, optimizer_name, seed)] = history
                    history.to_csv(
                        os.path.join(csv_dir, f"history_{problem_name}_{optimizer_name}_seed{seed}.csv"),
                        index=False,
                    )
                    plot_prediction(problem, y_pred, optimizer_name, seed, figures_dir)
                    print(f"  done in {time.time() - t0:.1f}s: gap={result['relative_solution_gap']:.4g} "
                          f"flag={result['failure_flag']}")
                except Exception as e:  # noqa: BLE001 - intentionally broad: keep the sweep alive
                    print(f"  FAILED ({time.time() - t0:.1f}s):", repr(e))
                    all_results.append({
                        "problem": problem_name, "optimizer": optimizer_name, "seed": seed,
                        "relative_solution_gap": np.nan, "pde_residual_mse": np.nan,
                        "final_total_loss": np.nan, "final_pde_loss": np.nan,
                        "final_ic_loss": np.nan, "final_bc_loss": np.nan,
                        "runtime_seconds": np.nan, "runtime_normalized_error": np.nan,
                        "loss_oscillation_index": np.nan, "num_logged_points": 0,
                        "num_parameters": np.nan, "failure_flag": "failed",
                        "failure_message": str(e),
                    })

    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv(os.path.join(csv_dir, "all_runs.csv"), index=False)

    plot_loss_curves(all_histories, figures_dir)
    plot_combined_loss_curves(all_histories, figures_dir)
    plot_gradient_conflict(all_histories, figures_dir)
    summary_df = plot_summary(all_results_df, figures_dir, csv_dir)

    total_wall_time = time.time() - sweep_start
    metadata["total_wall_time_seconds"] = total_wall_time
    with open(os.path.join(cfg.results_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    report_path = generate_report(
        all_results_df, os.path.join(cfg.results_dir, "REPORT.md"),
        {**metadata, "total_wall_time_seconds": f"{total_wall_time:.1f}"},
    )

    print("\nFinal results:")
    print(all_results_df.to_string(index=False))
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print("\nReport written to:", report_path)

    zip_base = cfg.results_dir.rstrip("/")
    zip_path = shutil.make_archive(zip_base, "zip", root_dir=cfg.results_dir)
    print("Created ZIP:", zip_path)

    return all_results_df, summary_df, zip_path


if __name__ == "__main__":
    main()
