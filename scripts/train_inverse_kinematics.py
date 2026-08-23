#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Train the inverse-kinematics PINN and plot the held-out reach-error distribution.

Usage:
    python scripts/train_inverse_kinematics.py [--robot panda] [--iterations 600]
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
from manipulapy_pinn.inverse_kinematics import train
from manipulapy_pinn.metrics import format_metrics, inverse_kinematics_metrics
from manipulapy_pinn.models import hidden_sizes
from manipulapy_pinn.plots import inverse_kinematics_figures, save_figures
from manipulapy_pinn.report import report_path, review_inverse_kinematics, training_report, write_report

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "runs"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="panda")
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--width", type=int, default=128, help="units per hidden layer")
    parser.add_argument("--depth", type=int, default=5,
                        help="number of hidden layers; >4 switches to residual blocks")
    parser.add_argument("--position-weight", type=float, default=1.0,
                        help="weight on the position term [m²]")
    parser.add_argument("--orientation-weight", type=float, default=1.0,
                        help="weight on the orientation term [rad²]; 0 disables it")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    robot = load_robot(args.robot)
    print(f"Robot: {robot.name} ({robot.n_joints} DOF)")

    hidden = hidden_sizes(args.width, args.depth)
    result = train(robot, iterations=args.iterations, hidden=hidden,
                   position_weight=args.position_weight,
                   orientation_weight=args.orientation_weight, seed=args.seed)
    print(f"Network: {type(result.model.net).__name__} {args.depth}x{args.width} "
          f"({sum(p.numel() for p in result.model.parameters())} parameters)")

    checkpoint_path = OUTPUT_DIR / f"inverse_kinematics_{robot.name}.pt"
    torch.save(result.model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

    metrics = inverse_kinematics_metrics(result.model, robot, n_targets=300,
                                         rng=np.random.default_rng(args.seed + 2))
    print(format_metrics(metrics, "Inverse-kinematics metrics (held-out)"))

    figures = inverse_kinematics_figures(result, robot, metrics,
                                         rng=np.random.default_rng(args.seed + 3))
    fig_paths = save_figures(figures, OUTPUT_DIR, "inverse_kinematics", robot.name)
    for fp in fig_paths:
        print(f"Saved figure: {fp}")

    report = training_report(
        title=f"Inverse-kinematics PINN — {robot.name}",
        robot_name=robot.name, n_joints=robot.n_joints,
        config={"task": "inverse_kinematics", "iterations": args.iterations,
                "position_weight": args.position_weight,
                "orientation_weight": args.orientation_weight,
                "orientation_deg": result.val_orientation_deg,
                "width": args.width, "depth": args.depth,
                "architecture": type(result.model.net).__name__, "seed": args.seed},
        metrics=metrics, history=result.loss_history,
        figure=[p.name for p in fig_paths], checkpoint=str(checkpoint_path.name),
        notes=review_inverse_kinematics(metrics),
    )
    path = write_report(report_path(OUTPUT_DIR, "inverse_kinematics", robot.name), report)
    print(f"Saved report: {path}")


if __name__ == "__main__":
    main()
