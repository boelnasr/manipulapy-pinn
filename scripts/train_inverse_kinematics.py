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
import matplotlib.pyplot as plt
import numpy as np
import torch

from manipulapy_pinn import load_robot
from manipulapy_pinn.backend_utils import torch_context
from manipulapy_pinn.inverse_kinematics import train
from manipulapy_pinn.metrics import format_metrics, inverse_kinematics_metrics
from manipulapy_pinn.models import hidden_sizes
from manipulapy_pinn.physics import fk_position_residual
from manipulapy_pinn.report import report_path, review_inverse_kinematics, training_report, write_report

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "runs"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="panda")
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--width", type=int, default=128, help="units per hidden layer")
    parser.add_argument("--depth", type=int, default=5,
                        help="number of hidden layers; >4 switches to residual blocks")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    robot = load_robot(args.robot)
    print(f"Robot: {robot.name} ({robot.n_joints} DOF)")

    hidden = hidden_sizes(args.width, args.depth)
    result = train(robot, iterations=args.iterations, hidden=hidden, seed=args.seed)
    print(f"Network: {type(result.model.net).__name__} {args.depth}x{args.width} "
          f"({sum(p.numel() for p in result.model.parameters())} parameters)")

    checkpoint_path = OUTPUT_DIR / f"inverse_kinematics_{robot.name}.pt"
    torch.save(result.model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

    # Held-out reach-error distribution over a fresh batch of targets.
    rng = np.random.default_rng(args.seed + 1)
    q_val = robot.sample_configurations(300, rng)
    target_val = torch.tensor(robot.forward_kinematics(q_val)[:, :3, 3], dtype=torch.float64)
    with torch.no_grad():
        q_pred = result.model(target_val)
        with torch_context():
            residual = fk_position_residual(robot.serial, q_pred, target_val)
    reach_error_mm = residual.norm(dim=-1).numpy() * 1000

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(f"Inverse-kinematics PINN — {robot.name}", fontweight="bold")

    axes[0].plot([h["iter"] for h in result.loss_history], [h["loss"] for h in result.loss_history])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("FK-residual MSE (log scale)")
    axes[0].set_title("Training curve")
    axes[0].grid(alpha=0.3)

    axes[1].hist(reach_error_mm, bins=30, color="steelblue", alpha=0.85)
    axes[1].axvline(reach_error_mm.mean(), color="crimson", linestyle="--",
                     label=f"mean = {reach_error_mm.mean():.0f} mm")
    axes[1].set_xlabel("reach error [mm]")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"Held-out reach error (n={len(reach_error_mm)})")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig_path = OUTPUT_DIR / f"inverse_kinematics_{robot.name}.png"
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    print(f"Saved figure: {fig_path}")

    metrics = inverse_kinematics_metrics(result.model, robot, n_targets=300,
                                         rng=np.random.default_rng(args.seed + 2))
    print(format_metrics(metrics, "Inverse-kinematics metrics (held-out)"))

    report = training_report(
        title=f"Inverse-kinematics PINN — {robot.name}",
        robot_name=robot.name, n_joints=robot.n_joints,
        config={"task": "inverse_kinematics", "iterations": args.iterations,
                "width": args.width, "depth": args.depth,
                "architecture": type(result.model.net).__name__, "seed": args.seed},
        metrics=metrics, history=result.loss_history,
        figure=str(fig_path.name), checkpoint=str(checkpoint_path.name),
        notes=review_inverse_kinematics(metrics),
    )
    path = write_report(report_path(OUTPUT_DIR, "inverse_kinematics", robot.name), report)
    print(f"Saved report: {path}")


if __name__ == "__main__":
    main()
