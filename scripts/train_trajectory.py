#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Train a forward-dynamics surrogate, then solve a start→goal trajectory PINN
through it, and plot q(t), q̇(t), and τ(t).

Usage:
    python scripts/train_trajectory.py [--robot panda] [--dynamics-iterations 1500] [--trajectory-iterations 800]
"""
import argparse
import os
from pathlib import Path

import matplotlib

if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from manipulapy_pinn import load_robot
from manipulapy_pinn.forward_dynamics import train as train_dynamics
from manipulapy_pinn.metrics import format_metrics, trajectory_metrics
from manipulapy_pinn.report import report_path, review_trajectory, training_report, write_report
from manipulapy_pinn.trajectory import solve

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "runs"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="panda")
    parser.add_argument("--dynamics-iterations", type=int, default=1500)
    parser.add_argument("--dynamics-samples", type=int, default=2000)
    parser.add_argument("--trajectory-iterations", type=int, default=800)
    parser.add_argument("--collocation-points", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    robot = load_robot(args.robot)
    print(f"Robot: {robot.name} ({robot.n_joints} DOF)")

    print("\n== Step 1/2: forward-dynamics surrogate ==")
    dyn_result = train_dynamics(
        robot, n_samples=args.dynamics_samples, iterations=args.dynamics_iterations, seed=args.seed,
    )

    rng = np.random.default_rng(args.seed + 1)
    q_start = robot.sample_configurations(1, rng)[0]
    q_goal = robot.sample_configurations(1, rng)[0]
    print(f"\nq_start = {np.round(q_start, 3)}")
    print(f"q_goal  = {np.round(q_goal, 3)}")

    print("\n== Step 2/2: trajectory PINN ==")
    traj_result = solve(
        q_start, q_goal, dyn_result.model,
        n_collocation=args.collocation_points, iterations=args.trajectory_iterations,
    )

    sample = traj_result.sample(200)
    n = robot.n_joints

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Trajectory PINN — {robot.name} "
                 f"(surrogate held-out q̈ RMSE = {dyn_result.val_data_rmse:.3f} rad/s²)", fontweight="bold")

    for j in range(n):
        axes[0, 0].plot(sample["t"], sample["q"][:, j], label=f"q{j+1}")
    axes[0, 0].set_title("Joint positions q(t)")
    axes[0, 0].set_xlabel("t (normalized)")
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[0, 0].grid(alpha=0.3)

    for j in range(n):
        axes[0, 1].plot(sample["t"], sample["qdot"][:, j])
    axes[0, 1].axhline(0, color="gray", lw=0.8)
    axes[0, 1].set_title("Joint velocities q̇(t) — should be ~0 at both ends")
    axes[0, 1].set_xlabel("t (normalized)")
    axes[0, 1].grid(alpha=0.3)

    for j in range(n):
        axes[1, 0].plot(sample["t"], sample["tau"][:, j])
    axes[1, 0].set_title("Commanded torque τ(t)")
    axes[1, 0].set_xlabel("t (normalized)")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot([h["iter"] for h in traj_result.loss_history], [h["physics"] for h in traj_result.loss_history], label="physics (EOM) residual")
    axes[1, 1].plot([h["iter"] for h in traj_result.loss_history], [h["boundary"] for h in traj_result.loss_history], label="boundary velocity")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Trajectory-solve convergence")
    axes[1, 1].set_xlabel("iteration")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig_path = OUTPUT_DIR / f"trajectory_{robot.name}.png"
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    print(f"\nSaved figure: {fig_path}")

    # Scored against ManipulaPy's *true* inverse dynamics, not the surrogate the
    # solve optimized through — see metrics.trajectory_metrics for why that
    # distinction is the whole point of evaluating this task.
    metrics = trajectory_metrics(traj_result, robot, n_points=200,
                                 n_collocation_trained=args.collocation_points)
    metrics["surrogate_val_rmse_rad_s2"] = dyn_result.val_data_rmse
    print(format_metrics(metrics, "Trajectory metrics (vs. ManipulaPy inverse dynamics)"))

    report = training_report(
        title=f"Trajectory PINN — {robot.name}",
        robot_name=robot.name, n_joints=robot.n_joints,
        config={"task": "trajectory", "dynamics_samples": args.dynamics_samples,
                "dynamics_iterations": args.dynamics_iterations,
                "trajectory_iterations": args.trajectory_iterations,
                "collocation_points": args.collocation_points, "seed": args.seed,
                "q_start": np.round(q_start, 4), "q_goal": np.round(q_goal, 4)},
        metrics=metrics, history=traj_result.loss_history,
        figure=str(fig_path.name),
        notes=review_trajectory(metrics),
    )
    path = write_report(report_path(OUTPUT_DIR, "trajectory", robot.name), report)
    print(f"Saved report: {path}")


if __name__ == "__main__":
    main()
