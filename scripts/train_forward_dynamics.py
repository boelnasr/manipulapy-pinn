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
import matplotlib.pyplot as plt
import numpy as np
import torch

from manipulapy_pinn import load_robot
from manipulapy_pinn.forward_dynamics import train

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "runs"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="panda")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    robot = load_robot(args.robot)
    print(f"Robot: {robot.name} ({robot.n_joints} DOF)")

    result = train(robot, n_samples=args.samples, iterations=args.iterations, seed=args.seed)

    checkpoint_path = OUTPUT_DIR / f"forward_dynamics_{robot.name}.pt"
    torch.save(result.model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

    # Held-out scatter: predicted vs. true q̈, one point per (sample, joint).
    rng = np.random.default_rng(args.seed + 1)
    from manipulapy_pinn.data import generate_forward_dynamics_dataset

    val = generate_forward_dynamics_dataset(robot, 300, rng)
    with torch.no_grad():
        q = torch.tensor(val["q"], dtype=torch.float64)
        qdot = torch.tensor(val["qdot"], dtype=torch.float64)
        tau = torch.tensor(val["tau"], dtype=torch.float64)
        qddot_pred = result.model(q, qdot, tau).numpy()
    qddot_true = val["qddot"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(f"Forward-dynamics PINN — {robot.name}", fontweight="bold")

    axes[0].plot([h["iter"] for h in result.loss_history], [h["data"] for h in result.loss_history], label="data loss")
    axes[0].plot([h["iter"] for h in result.loss_history], [h["physics"] for h in result.loss_history], label="physics loss")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("loss (log scale)")
    axes[0].set_title("Training curves")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].scatter(qddot_true.flatten(), qddot_pred.flatten(), s=4, alpha=0.3)
    lo, hi = qddot_true.min(), qddot_true.max()
    axes[1].plot([lo, hi], [lo, hi], "r--", lw=1, label="perfect prediction")
    axes[1].set_xlabel("true q̈ [rad/s²]")
    axes[1].set_ylabel("predicted q̈ [rad/s²]")
    axes[1].set_title(f"Held-out accuracy (RMSE = {result.val_data_rmse:.3f} rad/s²)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig_path = OUTPUT_DIR / f"forward_dynamics_{robot.name}.png"
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
