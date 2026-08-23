#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Train the forward-dynamics PINN and plot predicted vs. true q̈ on held-out data.

Usage:
    python scripts/train_forward_dynamics.py [--robot panda] [--samples 2000] [--iterations 1500]
"""
import argparse
import os
from pathlib import Path

import matplotlib

if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import numpy as np
import torch

from manipulapy_pinn import load_robot
from manipulapy_pinn.forward_dynamics import train
from manipulapy_pinn.metrics import (format_metrics, forward_dynamics_metrics,
                                     forward_dynamics_split_metrics)
from manipulapy_pinn.models import hidden_sizes
from manipulapy_pinn.plots import (forward_dynamics_figures, save_figures,
                                   split_comparison_figure)
from manipulapy_pinn.report import report_path, review_forward_dynamics, training_report, write_report

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "runs"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="panda")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--width", type=int, default=128, help="units per hidden layer")
    parser.add_argument("--depth", type=int, default=3,
                        help="number of hidden layers; >4 switches to residual blocks")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    robot = load_robot(args.robot)
    print(f"Robot: {robot.name} ({robot.n_joints} DOF)")

    hidden = hidden_sizes(args.width, args.depth)
    result = train(robot, n_samples=args.samples, iterations=args.iterations,
                   hidden=hidden, seed=args.seed)
    print(f"Network: {type(result.model.net).__name__} {args.depth}x{args.width} "
          f"({sum(p.numel() for p in result.model.parameters())} parameters)")

    checkpoint_path = OUTPUT_DIR / f"forward_dynamics_{robot.name}.pt"
    torch.save(result.model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

    metrics = forward_dynamics_metrics(result.model, robot, n_samples=500,
                                       rng=np.random.default_rng(args.seed + 2))
    print(format_metrics(metrics, "Forward-dynamics metrics (held-out)"))

    # Score the run's actual splits, not a fresh draw: this is what makes the
    # train/test/eval comparison mean anything.
    split_metrics = forward_dynamics_split_metrics(result.model, robot, result.splits)
    print("\nPer-split metrics")
    print("-" * 62)
    print(f"  {'split':<8}{'n':>6}{'rmse':>10}{'mae':>10}{'r2':>9}{'skill':>9}")
    for name in ("train", "test", "eval"):
        m = split_metrics[name]
        print(f"  {name:<8}{m['n_samples']:>6}{m['rmse_rad_s2']:>10.4f}"
              f"{m['mae_rad_s2']:>10.4f}{m['r2']:>9.4f}{m['skill_score']:>9.4f}")
    for k, v in split_metrics["gaps"].items():
        print(f"  {k:<28}{v:>8.3f}")

    figures = forward_dynamics_figures(result, robot, metrics,
                                       rng=np.random.default_rng(args.seed + 3))
    figures["split_comparison"] = split_comparison_figure(split_metrics, robot.n_joints)
    fig_paths = save_figures(figures, OUTPUT_DIR, "forward_dynamics", robot.name)
    for fp in fig_paths:
        print(f"Saved figure: {fp}")

    report = training_report(
        title=f"Forward-dynamics PINN — {robot.name}",
        robot_name=robot.name, n_joints=robot.n_joints,
        config={"task": "forward_dynamics", "samples": args.samples,
                "best_val_iteration": result.best_iteration().get("iter"),
                "split_train_test_eval": "70/20/10",
                "test_rmse_rad_s2": result.test_rmse,
                "eval_rmse_rad_s2": result.eval_rmse,
                **{f"gap_{k}": v for k, v in split_metrics["gaps"].items()},
                "iterations": args.iterations, "width": args.width,
                "depth": args.depth, "architecture": type(result.model.net).__name__,
                "seed": args.seed},
        metrics=metrics, history=result.loss_history,
        figure=[p.name for p in fig_paths], checkpoint=str(checkpoint_path.name),
        notes=review_forward_dynamics(metrics),
    )
    path = write_report(report_path(OUTPUT_DIR, "forward_dynamics", robot.name), report)
    print(f"Saved report: {path}")


if __name__ == "__main__":
    main()
