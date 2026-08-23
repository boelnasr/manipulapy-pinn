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
import numpy as np

from manipulapy_pinn import load_robot
from manipulapy_pinn.forward_dynamics import train as train_dynamics
from manipulapy_pinn.metrics import format_metrics, trajectory_metrics
from manipulapy_pinn.models import hidden_sizes
from manipulapy_pinn.plots import save_figures, trajectory_figures
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
    parser.add_argument("--dynamics-width", type=int, default=128)
    parser.add_argument("--dynamics-depth", type=int, default=3)
    parser.add_argument("--trajectory-width", type=int, default=64)
    parser.add_argument("--trajectory-depth", type=int, default=2,
                        help="hidden layers in q(t) and tau(t); >4 switches to residual blocks")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    robot = load_robot(args.robot)
    print(f"Robot: {robot.name} ({robot.n_joints} DOF)")

    print("\n== Step 1/2: forward-dynamics surrogate ==")
    dyn_result = train_dynamics(
        robot, n_samples=args.dynamics_samples, iterations=args.dynamics_iterations,
        hidden=hidden_sizes(args.dynamics_width, args.dynamics_depth), seed=args.seed,
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
        hidden=hidden_sizes(args.trajectory_width, args.trajectory_depth),
        torque_hidden=hidden_sizes(args.trajectory_width, args.trajectory_depth),
    )

    # Scored against ManipulaPy's *true* inverse dynamics, not the surrogate the
    # solve optimized through — see metrics.trajectory_metrics for why that
    # distinction is the whole point of evaluating this task.
    metrics = trajectory_metrics(traj_result, robot, n_points=200,
                                 n_collocation_trained=args.collocation_points)
    metrics["surrogate_eval_rmse_rad_s2"] = dyn_result.eval_rmse
    print(format_metrics(metrics, "Trajectory metrics (vs. ManipulaPy inverse dynamics)"))

    figures = trajectory_figures(traj_result, robot, metrics, n_points=200)
    fig_paths = save_figures(figures, OUTPUT_DIR, "trajectory", robot.name)
    for fp in fig_paths:
        print(f"Saved figure: {fp}")

    report = training_report(
        title=f"Trajectory PINN — {robot.name}",
        robot_name=robot.name, n_joints=robot.n_joints,
        config={"task": "trajectory", "dynamics_samples": args.dynamics_samples,
                "dynamics_iterations": args.dynamics_iterations,
                "trajectory_iterations": args.trajectory_iterations,
                "collocation_points": args.collocation_points,
                "dynamics_width": args.dynamics_width, "dynamics_depth": args.dynamics_depth,
                "trajectory_width": args.trajectory_width,
                "trajectory_depth": args.trajectory_depth, "seed": args.seed,
                "q_start": np.round(q_start, 4), "q_goal": np.round(q_goal, 4)},
        metrics=metrics, history=traj_result.loss_history,
        figure=[p.name for p in fig_paths],
        notes=review_trajectory(metrics),
    )
    path = write_report(report_path(OUTPUT_DIR, "trajectory", robot.name), report)
    print(f"Saved report: {path}")


if __name__ == "__main__":
    main()
