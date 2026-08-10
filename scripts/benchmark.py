#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Benchmark: trained PINNs vs. ManipulaPy's own classical solvers.

This is not a "PINN wins" benchmark — it's here to be honest about the
trade being made. A PINN pays a one-time training cost and then answers
every subsequent query with a single forward pass; ManipulaPy's classical
solvers pay no training cost but redo their (iterative, or recursive
Newton-Euler) work on every single call. Which one is worth it depends on
how many queries you'll make against a fixed robot and how much accuracy
you need — this script prints real numbers from this machine so you can
decide, rather than asserting an answer.

Usage:
    python scripts/benchmark.py [--robot panda]
"""
import argparse
import time

import numpy as np
import torch

from manipulapy_pinn import load_robot
from manipulapy_pinn.backend_utils import torch_context
from manipulapy_pinn.forward_dynamics import train as train_dynamics
from manipulapy_pinn.inverse_kinematics import train as train_ik
from manipulapy_pinn.physics import fk_position_residual
from manipulapy_pinn.trajectory import solve as solve_trajectory


def _time_calls(fn, n=200):
    start = time.time()
    for _ in range(n):
        fn()
    return (time.time() - start) / n


def benchmark_forward_dynamics(robot, rng):
    print("\n" + "=" * 72)
    print("Forward dynamics: q̈ = f(q, q̇, τ)")
    print("=" * 72)

    result = train_dynamics(robot, n_samples=2000, iterations=1500, verbose=False)
    print(f"PINN held-out RMSE: {result.val_data_rmse:.4f} rad/s²")

    q = robot.sample_configurations(1, rng)[0]
    qdot = rng.uniform(-1, 1, robot.n_joints)
    tau = rng.uniform(-10, 10, robot.n_joints)
    gravity = np.array([0.0, 0.0, -9.81])

    manipulapy_time = _time_calls(
        lambda: robot.dynamics.forward_dynamics(q, qdot, tau, gravity, np.zeros(6))
    )

    with torch.no_grad():
        q_t = torch.tensor(q, dtype=torch.float64).unsqueeze(0)
        qdot_t = torch.tensor(qdot, dtype=torch.float64).unsqueeze(0)
        tau_t = torch.tensor(tau, dtype=torch.float64).unsqueeze(0)
        pinn_time = _time_calls(lambda: result.model(q_t, qdot_t, tau_t))

    print(f"ManipulaPy (NumPy, exact):     {manipulapy_time * 1e3:.3f} ms/call")
    print(f"PINN surrogate (single query): {pinn_time * 1e3:.3f} ms/call "
          f"({manipulapy_time / max(pinn_time, 1e-9):.1f}x)")
    print("Note: the PINN number does not include its one-time training cost "
          "(a few seconds to a couple of minutes, see forward_dynamics.train).")


def benchmark_inverse_kinematics(robot, rng):
    print("\n" + "=" * 72)
    print("Inverse kinematics: q = h(target pose)")
    print("=" * 72)

    result = train_ik(robot, iterations=600, verbose=False)
    print(f"PINN held-out reach error: {result.val_position_rmse * 100:.2f} cm")

    n_targets = 15  # ManipulaPy's own README documents ~90% DLS success rate from a
    # zero initial guess, with a long latency tail on the rest — small samples of this
    # solver are noisy by nature. Report median + success rate, not mean, for that reason.
    q_targets = robot.sample_configurations(n_targets, rng)
    targets = robot.forward_kinematics(q_targets)[:, :3, 3]

    # ManipulaPy's iterative DLS solver, timed and scored the same way.
    dls_times, dls_errors, dls_ok = [], [], []
    for target in targets:
        T_target = np.eye(4)
        T_target[:3, 3] = target
        start = time.time()
        q_sol, ok, _iters = robot.serial.iterative_inverse_kinematics(T_target, thetalist0=np.zeros(robot.n_joints))
        dls_times.append(time.time() - start)
        reached = robot.serial.forward_kinematics(q_sol)[:3, 3]
        dls_errors.append(np.linalg.norm(reached - target))
        dls_ok.append(bool(ok))
    dls_errors = np.array(dls_errors)
    success_rate = np.mean(dls_ok)

    target_t = torch.tensor(targets, dtype=torch.float64)
    with torch.no_grad():
        start = time.time()
        q_pred = result.model(target_t)
        pinn_batch_time = (time.time() - start) / len(targets)
        with torch_context():
            residual = fk_position_residual(robot.serial, q_pred, target_t)
        pinn_errors = residual.norm(dim=-1).numpy()

    print(f"\n{'':30s}{'median error':>14s}{'median time':>16s}")
    print(f"{'ManipulaPy DLS IK':30s}{np.median(dls_errors) * 100:12.2f} cm{np.median(dls_times) * 1e3:14.2f} ms"
          f"   ({success_rate * 100:.0f}% converged, of {n_targets})")
    print(f"{'PINN (batched query)':30s}{np.median(pinn_errors) * 100:12.2f} cm{pinn_batch_time * 1e3:14.4f} ms")
    print(f"Note: median (not mean) is the honest number for DLS — it has a real long tail "
          f"of hard targets that hit the iteration cap; ManipulaPy's own README documents "
          f"the same pattern for this solver. The PINN is a single fixed-cost forward pass, "
          f"and (unlike DLS) its whole batch of {n_targets} targets above ran in one "
          f"vectorized call.")


def benchmark_trajectory(robot, rng):
    print("\n" + "=" * 72)
    print("Trajectory generation: q(t) from q_start to q_goal")
    print("=" * 72)

    dyn_result = train_dynamics(robot, n_samples=1500, iterations=1200, verbose=False)
    q_start = robot.sample_configurations(1, rng)[0]
    q_goal = robot.sample_configurations(1, rng)[0]

    start = time.time()
    traj_result = solve_trajectory(
        q_start, q_goal, dyn_result.model, n_collocation=24, iterations=800, verbose=False,
    )
    pinn_solve_time = time.time() - start
    sample = traj_result.sample(50)
    boundary_err = np.linalg.norm(sample["q"][0] - q_start) + np.linalg.norm(sample["q"][-1] - q_goal)
    peak_speed = np.max(np.linalg.norm(sample["qdot"], axis=-1))

    from ManipulaPy.ManipulaPy_data import get_robot_urdf
    from ManipulaPy.path_planning import OptimizedTrajectoryPlanning

    # OptimizedTrajectoryPlanning's collision-avoidance step calls URDF.link_fk
    # directly, which (unlike SerialManipulator.forward_kinematics) requires
    # every actuated joint in the URDF, gripper included — pad with zeros for
    # any joints RobotModel excludes as "not the arm" (see robots.py).
    full_limits = robot.serial.joint_limits
    full_n = len(full_limits)
    q_start_full = np.zeros(full_n)
    q_goal_full = np.zeros(full_n)
    q_start_full[: robot.n_joints] = q_start
    q_goal_full[: robot.n_joints] = q_goal

    planner = OptimizedTrajectoryPlanning(
        robot.serial, get_robot_urdf(robot.name), robot.dynamics, joint_limits=full_limits,
        torque_limits=[(-50, 50)] * full_n,
    )
    start = time.time()
    quintic = planner.joint_trajectory(thetastart=q_start_full, thetaend=q_goal_full, Tf=2.0, N=50, method=5)
    quintic_time = time.time() - start

    print(f"\n{'':30s}{'boundary err':>14s}{'peak |q̇|':>12s}{'wall time':>14s}")
    print(f"{'PINN (physics-informed)':30s}{boundary_err:14.2e}{peak_speed:12.3f}{pinn_solve_time:14.2f}s")
    print(f"{'ManipulaPy quintic (kinematic)':30s}{'—':>14s}{'—':>12s}{quintic_time * 1e3:13.2f}ms")
    print("Note: these solve different problems. The quintic planner produces a smooth "
          "time-scaled joint path with no notion of torque or dynamics; the PINN solves "
          "for a trajectory *and* a torque profile jointly, consistent with the learned "
          "dynamics surrogate (see peak |q̇| and the boundary velocity term in "
          "trajectory.solve). The quintic planner is not trying to minimize torque and "
          "does not report one — its 'time' column is the fair comparison, not accuracy.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="panda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    robot = load_robot(args.robot)
    rng = np.random.default_rng(args.seed)
    print(f"Benchmarking on {robot.name} ({robot.n_joints} DOF)")

    benchmark_forward_dynamics(robot, rng)
    benchmark_inverse_kinematics(robot, rng)
    benchmark_trajectory(robot, rng)


if __name__ == "__main__":
    main()
