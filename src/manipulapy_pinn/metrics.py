# SPDX-License-Identifier: AGPL-3.0-or-later
"""Evaluation metrics for every trained model in this package.

Each ``*_metrics`` function returns a plain ``dict`` of floats (and small
arrays) so results are easy to log, diff between runs, or serialize.
``format_metrics`` renders one for a terminal.

Two principles run through this module:

- **ManipulaPy is the reference, not the network.** Every ground-truth
  quantity here — the true joint accelerations, the true torque a trajectory
  requires, the achieved end-effector pose — comes from a ManipulaPy call,
  never from a second model or a finite-difference approximation of one.
- **Report what the training loss cannot see.** A training loss is a single
  scalar on the data the optimizer chose; these metrics deliberately probe
  the places it says nothing about — per-joint breakdowns, worst cases,
  distribution tails, constraint violations, and (for the trajectory) points
  that were never collocation points.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from .backend_utils import numpy_context
from .data import GRAVITY, generate_forward_dynamics_dataset
from .robots import RobotModel


def _r2(true: np.ndarray, pred: np.ndarray) -> float:
    """Coefficient of determination, aggregated over every element."""
    ss_res = float(((true - pred) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def forward_dynamics_metrics(
    model, robot: RobotModel, n_samples: int = 500, rng: np.random.Generator = None
) -> Dict:
    """Evaluate a :class:`ForwardDynamicsPINN` on freshly sampled states.

    Beyond plain RMSE this reports:

    - **per-joint RMSE**, because a single aggregate number hides the usual
      failure mode — one poorly conditioned joint carrying most of the error.
    - **normalized RMSE and R²**, which are scale-free. Joint accelerations
      differ by an order of magnitude across a 7-DOF arm under the same
      torque range, so a raw rad/s² figure is dominated by whichever joints
      happen to move fastest.
    - **skill score** against the trivial "always predict the mean"
      predictor. A model can post a respectable-looking RMSE while beating
      that baseline by almost nothing; this states plainly whether the
      network learned the state dependence at all.
    - **torque-space RMSE**, the acceleration error mapped through the true
      mass matrix (‖M Δq̈‖, in N·m). This is the error in the units the robot
      is actually commanded in, and it weights each joint by its inertia
      rather than treating a rad/s² of the wrist as equal to one of the base.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    data = generate_forward_dynamics_dataset(robot, n_samples, rng)

    with torch.no_grad():
        pred = model(
            torch.tensor(data["q"], dtype=torch.float64),
            torch.tensor(data["qdot"], dtype=torch.float64),
            torch.tensor(data["tau"], dtype=torch.float64),
        ).numpy()
    true = data["qddot"]
    err = pred - true

    per_joint_rmse = np.sqrt((err ** 2).mean(axis=0))
    true_std = true.std(axis=0)
    baseline_rmse = float(np.sqrt(((true - true.mean(axis=0)) ** 2).mean()))
    rmse = float(np.sqrt((err ** 2).mean()))

    # Acceleration error expressed as the torque it corresponds to: M(q) Δq̈.
    with numpy_context():
        torque_err = np.stack(
            [robot.dynamics.mass_matrix(data["q"][i]) @ err[i] for i in range(n_samples)]
        )

    return {
        "n_samples": n_samples,
        "rmse_rad_s2": rmse,
        "mae_rad_s2": float(np.abs(err).mean()),
        "max_abs_error_rad_s2": float(np.abs(err).max()),
        "per_joint_rmse_rad_s2": per_joint_rmse,
        "nrmse_per_joint": per_joint_rmse / np.maximum(true_std, 1e-12),
        "r2": _r2(true, pred),
        "baseline_rmse_rad_s2": baseline_rmse,
        "skill_score": 1.0 - rmse / baseline_rmse if baseline_rmse > 0 else float("nan"),
        "torque_rmse_Nm": float(np.sqrt((torque_err ** 2).mean())),
        "torque_max_abs_Nm": float(np.abs(torque_err).max()),
    }


def inverse_kinematics_metrics(
    model, robot: RobotModel, n_targets: int = 300, rng: np.random.Generator = None
) -> Dict:
    """Evaluate an :class:`InverseKinematicsPINN` on freshly sampled targets.

    Reports the **error distribution** rather than only its mean — median,
    p95 and max — because reach error is heavy-tailed and a mean hides how
    bad the worst targets are.

    Two things the training loss cannot see are checked explicitly:

    - **Success rate at fixed tolerances.** This is how the IK literature
      reports learned solvers, and it is the number that matters if the
      output is used as a seed for a numerical solver rather than as a final
      answer.
    - **Joint-limit violations.** Nothing in the network architecture or the
      FK-residual loss constrains the output to the robot's joint limits, so
      a configuration that reaches the target perfectly may still be
      unreachable. Limits come from ManipulaPy's URDF parse.
    - **Orientation error.** The loss matches *position only*, so end-effector
      orientation is entirely unconstrained. Measuring it against the pose of
      the configuration that generated the target quantifies what that
      choice costs; it is not a failure of training, but it is not free
      either.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    q_ref = robot.sample_configurations(n_targets, rng)
    poses = robot.forward_kinematics(q_ref)  # (N, 4, 4), the reference poses
    targets = poses[:, :3, 3]

    with torch.no_grad():
        q_pred = model(torch.tensor(targets, dtype=torch.float64)).numpy()

    with numpy_context():
        achieved = np.stack([robot.serial.forward_kinematics(q) for q in q_pred])
    pos_err = np.linalg.norm(achieved[:, :3, 3] - targets, axis=-1)  # metres

    # Orientation gap vs. the reference pose, as a rotation angle.
    R_err = np.einsum("nij,nkj->nik", achieved[:, :3, :3], poses[:, :3, :3])
    cos = np.clip((np.trace(R_err, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    orient_err = np.degrees(np.arccos(cos))

    lo = np.array([l for l, _ in robot.joint_limits])
    hi = np.array([h for _, h in robot.joint_limits])
    violation = np.maximum(np.maximum(lo - q_pred, q_pred - hi), 0.0)

    mm = pos_err * 1000.0
    return {
        "n_targets": n_targets,
        "mean_mm": float(mm.mean()),
        "median_mm": float(np.median(mm)),
        "p95_mm": float(np.percentile(mm, 95)),
        "max_mm": float(mm.max()),
        "success_rate_1mm": float((mm <= 1.0).mean()),
        "success_rate_5mm": float((mm <= 5.0).mean()),
        "success_rate_10mm": float((mm <= 10.0).mean()),
        "success_rate_50mm": float((mm <= 50.0).mean()),
        "orientation_mean_deg": float(orient_err.mean()),
        "orientation_max_deg": float(orient_err.max()),
        "joint_limit_violation_rate": float((violation > 0).any(axis=-1).mean()),
        "joint_limit_max_violation_rad": float(violation.max()),
    }


def trajectory_metrics(
    result, robot: RobotModel, n_points: int = 200, n_collocation_trained: int = None
) -> Dict:
    """Evaluate a solved trajectory against ManipulaPy's **true** dynamics.

    This is the metric the trajectory solver otherwise lacks entirely. Its
    training loss measures a residual through a *learned* dynamics surrogate,
    evaluated at the same collocation points it optimizes — so a low value
    proves neither that the trajectory is dynamically feasible nor that it
    holds up between those points.

    Both gaps are closed here:

    - **True torque error.** ``ManipulatorDynamics.inverse_dynamics`` gives
      the exact torque the trajectory q(t), q̇(t), q̈(t) requires. Comparing it
      against the torque network's own output measures how far the solution
      is from being dynamically consistent under real physics rather than
      under the surrogate.
    - **Dense evaluation.** Metrics are computed on ``n_points`` samples,
      typically far more than the collocation points used in training. When
      ``n_collocation_trained`` is given, the same error is also reported on
      a grid matching the training points, so the two can be compared
      directly — a large gap between them is the signature of a trajectory
      fitted to its collocation points and wrong in between.

    Also reported: boundary-condition errors (position should be exact by
    construction, velocity is only softly penalized) and the peak speeds and
    torques that the loss does not bound at all.
    """
    s = result.sample(n_points)
    q, qdot, qddot, tau_net = s["q"], s["qdot"], s["qddot"], s["tau"]

    with numpy_context():
        tau_true = np.stack([
            robot.dynamics.inverse_dynamics(q[i], qdot[i], qddot[i], GRAVITY, np.zeros(6))
            for i in range(n_points)
        ])
    tau_err = tau_net - tau_true

    metrics = {
        "n_points": n_points,
        "torque_rmse_Nm": float(np.sqrt((tau_err ** 2).mean())),
        "torque_max_abs_error_Nm": float(np.abs(tau_err).max()),
        "torque_relative_error": float(
            np.linalg.norm(tau_err) / max(np.linalg.norm(tau_true), 1e-12)
        ),
        "peak_true_torque_Nm": float(np.abs(tau_true).max()),
        "peak_network_torque_Nm": float(np.abs(tau_net).max()),
        "peak_speed_rad_s": float(np.abs(qdot).max()),
        "peak_accel_rad_s2": float(np.abs(qddot).max()),
        "boundary_q_start_err_rad": float(
            np.linalg.norm(q[0] - result.trajectory_model.q_start.numpy())
        ),
        "boundary_q_goal_err_rad": float(
            np.linalg.norm(q[-1] - result.trajectory_model.q_goal.numpy())
        ),
        "boundary_qdot_start_rad_s": float(np.linalg.norm(qdot[0])),
        "boundary_qdot_end_rad_s": float(np.linalg.norm(qdot[-1])),
        "path_length_rad": float(np.abs(np.diff(q, axis=0)).sum()),
    }

    if n_collocation_trained:
        sc = result.sample(n_collocation_trained)
        with numpy_context():
            tau_true_c = np.stack([
                robot.dynamics.inverse_dynamics(
                    sc["q"][i], sc["qdot"][i], sc["qddot"][i], GRAVITY, np.zeros(6)
                )
                for i in range(n_collocation_trained)
            ])
        rmse_c = float(np.sqrt(((sc["tau"] - tau_true_c) ** 2).mean()))
        metrics["torque_rmse_on_collocation_Nm"] = rmse_c
        metrics["dense_vs_collocation_ratio"] = (
            metrics["torque_rmse_Nm"] / rmse_c if rmse_c > 0 else float("nan")
        )

    return metrics


def format_metrics(metrics: Dict, title: str = "Metrics") -> str:
    """Render a metrics dict as an aligned block for terminal output."""
    lines = [f"\n{title}", "-" * max(len(title), 46)]
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            lines.append(f"  {k:<34s} [{', '.join(f'{x:.3f}' for x in v)}]")
        elif isinstance(v, float):
            lines.append(f"  {k:<34s} {v:>12.5g}")
        else:
            lines.append(f"  {k:<34s} {v:>12}")
    return "\n".join(lines)
