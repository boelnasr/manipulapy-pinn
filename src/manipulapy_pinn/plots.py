# SPDX-License-Identifier: AGPL-3.0-or-later
"""Figure sets for each training task.

Each ``*_figures`` function returns an ordered ``{name: Figure}`` mapping; the
``train_*.py`` scripts save every entry as ``runs/<task>_<robot>_<name>.png``.

The figures here deliberately plot quantities the metrics already compute but
that a single summary number hides — per-joint error rather than an aggregate,
distributions rather than means, constraint violations rather than accuracy
alone, and (for the trajectory) the true torque from ManipulaPy's inverse
dynamics next to the one the network proposes.

Every axis carries units, and every axis with more than one series or a
reference line carries a legend.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

from .backend_utils import numpy_context
from .data import GRAVITY, generate_forward_dynamics_dataset
from .physics import pose_features

#: One colour per joint, stable across every figure in a run so joint 3 is the
#: same colour in the position, velocity and torque plots.
JOINT_CMAP = plt.get_cmap("viridis")
TRAIN_C, VAL_C, REF_C, WARN_C = "#1f77b4", "#c1121f", "#444444", "#e07a00"


def _joint_colors(n: int):
    return [JOINT_CMAP(i / max(n - 1, 1)) for i in range(n)]


def _finish(ax, xlabel, ylabel, title, legend=True, loc="best"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10.5)
    ax.grid(alpha=0.3)
    if legend:
        ax.legend(fontsize=8, loc=loc)
    return ax


# --------------------------------------------------------------------------
# Forward dynamics
# --------------------------------------------------------------------------
def forward_dynamics_figures(result, robot, metrics: Dict, rng=None, n_eval: int = 500) -> "OrderedDict[str, plt.Figure]":
    """Four figures: convergence, accuracy, per-joint error, error structure."""
    rng = rng if rng is not None else np.random.default_rng(0)
    data = generate_forward_dynamics_dataset(robot, n_eval, rng)
    with torch.no_grad():
        pred = result.model(
            torch.tensor(data["q"], dtype=torch.float64),
            torch.tensor(data["qdot"], dtype=torch.float64),
            torch.tensor(data["tau"], dtype=torch.float64),
        ).numpy()
    true = data["qddot"]
    err = pred - true
    figs = OrderedDict()

    # --- 1. convergence -----------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    it = [h["iter"] for h in result.loss_history]
    ax[0].plot(it, [h["data"] for h in result.loss_history], color=TRAIN_C, lw=1.2, label="train, data term")
    ax[0].plot(it, [h["physics"] for h in result.loss_history], color=TRAIN_C, lw=0.7, alpha=0.4,
               label="train, physics term")
    if result.val_history:
        ax[0].plot([h["iter"] for h in result.val_history], [h["val_mse"] for h in result.val_history],
                   color=VAL_C, lw=1.8, label="held-out")
        best = result.best_iteration()
        ax[0].axvline(best["iter"], color=VAL_C, ls=":", lw=1.2,
                      label=f"best held-out (iter {best['iter']})")
    ax[0].set_yscale("log")
    _finish(ax[0], "iteration", "MSE  [(rad/s²)²]", "Convergence — train vs. held-out")

    # The physics/data ratio is a Rayleigh quotient of MᵀM, so it wanders as the
    # error direction rotates even when the error magnitude falls smoothly. This
    # is what makes the physics curve look noisy; plotting it directly shows the
    # noise is the metric, not the optimizer.
    ratio = [h["physics"] / h["data"] for h in result.loss_history if h["data"] > 0]
    rit = [h["iter"] for h in result.loss_history if h["data"] > 0]
    ax[1].plot(rit, ratio, color=WARN_C, lw=0.8, label="physics / data")
    ax[1].axhline(1.0, color=REF_C, ls="--", lw=1, label="equal weight")
    ax[1].set_yscale("log")
    _finish(ax[1], "iteration", "ratio  [dimensionless]",
            "Why the physics curve looks noisy\n(ratio = ‖MΔ‖²/‖Δ‖², set by M's conditioning)")
    fig.tight_layout()
    figs["convergence"] = fig

    # --- 2. accuracy --------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax[0].scatter(true.flatten(), pred.flatten(), s=5, alpha=0.25, color=TRAIN_C,
                  label=f"held-out samples (n={true.size})")
    lo, hi = float(true.min()), float(true.max())
    ax[0].plot([lo, hi], [lo, hi], ls="--", lw=1.2, color=VAL_C, label="ideal (y = x)")
    _finish(ax[0], "true q̈  [rad/s²]", "predicted q̈  [rad/s²]",
            f"Held-out accuracy — RMSE {metrics['rmse_rad_s2']:.3f} rad/s², R² {metrics['r2']:.3f}")

    ax[1].hist(err.flatten(), bins=60, color=TRAIN_C, alpha=0.85, label="residuals")
    ax[1].axvline(0, color=REF_C, ls="--", lw=1, label="zero error")
    ax[1].axvline(err.mean(), color=VAL_C, ls="-", lw=1.4,
                  label=f"mean {err.mean():+.3f} rad/s²")
    _finish(ax[1], "prediction error  [rad/s²]", "count",
            f"Residual distribution — MAE {metrics['mae_rad_s2']:.3f}, max {metrics['max_abs_error_rad_s2']:.2f}")
    fig.tight_layout()
    figs["accuracy"] = fig

    # --- 3. per-joint -------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    n = robot.n_joints
    idx = np.arange(n)
    colors = _joint_colors(n)
    ax[0].bar(idx, metrics["per_joint_rmse_rad_s2"], color=colors, label="per-joint RMSE")
    ax[0].axhline(metrics["rmse_rad_s2"], color=VAL_C, ls="--", lw=1.2,
                  label=f"aggregate {metrics['rmse_rad_s2']:.2f} rad/s²")
    ax[0].set_xticks(idx, [f"q{i+1}" for i in idx])
    _finish(ax[0], "joint", "RMSE  [rad/s²]", "Absolute error by joint")

    # Normalized error is the fair comparison: joints differ by an order of
    # magnitude in how fast they accelerate under the same torque range, so raw
    # RMSE always makes the fastest joints look worst.
    ax[1].bar(idx, metrics["nrmse_per_joint"], color=colors, label="per-joint NRMSE")
    ax[1].axhline(1.0, color=VAL_C, ls="--", lw=1.2, label="no better than that joint's mean")
    ax[1].set_xticks(idx, [f"q{i+1}" for i in idx])
    ax[1].set_ylim(0, max(1.15, float(np.max(metrics["nrmse_per_joint"])) * 1.15))
    _finish(ax[1], "joint", "RMSE / std(q̈)  [dimensionless]", "Scale-free error by joint")
    fig.tight_layout()
    figs["per_joint"] = fig

    # --- 4. error structure -------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax[0].scatter(true.flatten(), err.flatten(), s=5, alpha=0.25, color=TRAIN_C, label="residuals")
    ax[0].axhline(0, color=REF_C, ls="--", lw=1, label="zero error")
    _finish(ax[0], "true q̈  [rad/s²]", "error  [rad/s²]",
            "Error vs. magnitude\n(a widening band means error scales with acceleration)")

    with numpy_context():
        cond = np.array([np.linalg.cond(robot.dynamics.mass_matrix(q)) for q in data["q"]])
    per_sample = np.sqrt((err ** 2).mean(axis=1))
    ax[1].scatter(cond, per_sample, s=6, alpha=0.35, color=TRAIN_C, label="held-out states")
    if len(cond) > 2:
        fit = np.poly1d(np.polyfit(cond, per_sample, 1))
        xs = np.linspace(cond.min(), cond.max(), 50)
        ax[1].plot(xs, fit(xs), color=VAL_C, lw=1.4, label="linear fit")
    _finish(ax[1], "cond(M(q))  [dimensionless]", "per-state RMSE  [rad/s²]",
            "Error vs. mass-matrix conditioning")
    fig.tight_layout()
    figs["error_structure"] = fig
    return figs


# --------------------------------------------------------------------------
# Inverse kinematics
# --------------------------------------------------------------------------
def inverse_kinematics_figures(result, robot, metrics: Dict, rng=None, n_eval: int = 300) -> "OrderedDict[str, plt.Figure]":
    """Three figures: convergence, error distribution, constraint satisfaction."""
    rng = rng if rng is not None else np.random.default_rng(0)
    q_ref = robot.sample_configurations(n_eval, rng)
    poses = robot.forward_kinematics(q_ref)
    targets = poses[:, :3, 3]
    with torch.no_grad():
        features = pose_features(torch.tensor(poses, dtype=torch.float64))
        q_pred = result.model(features).numpy()
    with numpy_context():
        achieved = np.stack([robot.serial.forward_kinematics(q) for q in q_pred])
    err_mm = np.linalg.norm(achieved[:, :3, 3] - targets, axis=-1) * 1000.0

    R_err = np.einsum("nij,nkj->nik", achieved[:, :3, :3], poses[:, :3, :3])
    cos = np.clip((np.trace(R_err, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    orient_deg = np.degrees(np.arccos(cos))

    lo = np.array([l for l, _ in robot.joint_limits])
    hi = np.array([h for _, h in robot.joint_limits])
    violation = np.maximum(np.maximum(lo - q_pred, q_pred - hi), 0.0)
    figs = OrderedDict()

    # --- 1. convergence -----------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    it = [h["iter"] for h in result.loss_history]
    loss = [h["loss"] for h in result.loss_history]
    if "position" in result.loss_history[0]:
        ax[0].plot(it, [h["position"] for h in result.loss_history], color=TRAIN_C, lw=0.9,
                   label="position term  [m²]")
        ax[0].plot(it, [h["orientation"] for h in result.loss_history], color=WARN_C, lw=0.9,
                   label="orientation term  [rad²]")
    ax[0].plot(it, loss, color=REF_C, lw=1.4, alpha=0.8, label="total (weighted)")
    if len(loss) > 40:
        w = max(5, len(loss) // 40)
        smooth = np.convolve(loss, np.ones(w) / w, mode="valid")
        ax[0].plot(it[w - 1:], smooth, color=VAL_C, lw=1.6, label=f"moving average ({w} steps)")
    ax[0].set_yscale("log")
    _finish(ax[0], "iteration", "loss term  [see legend]", "Convergence — position and orientation")

    # Each step draws fresh targets, so the per-step loss is itself a sample of
    # generalization: there is no fixed training set to memorize.
    rmse_mm = np.sqrt(np.array(loss)) * 1000.0
    ax[1].plot(it, rmse_mm, color=TRAIN_C, lw=1.0, label="per-step RMSE on fresh targets")
    ax[1].axhline(metrics["mean_mm"], color=VAL_C, ls="--", lw=1.4,
                  label=f"final held-out mean {metrics['mean_mm']:.1f} mm")
    ax[1].set_yscale("log")
    _finish(ax[1], "iteration", "reach error  [mm]", "Convergence in physical units")
    fig.tight_layout()
    figs["convergence"] = fig

    # --- 2. error distribution ---------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax[0].hist(err_mm, bins=40, color=TRAIN_C, alpha=0.85, label=f"held-out targets (n={n_eval})")
    ax[0].axvline(metrics["median_mm"], color=VAL_C, lw=1.6, label=f"median {metrics['median_mm']:.1f} mm")
    ax[0].axvline(metrics["p95_mm"], color=WARN_C, ls="--", lw=1.4, label=f"p95 {metrics['p95_mm']:.1f} mm")
    _finish(ax[0], "reach error  [mm]", "count", "Held-out reach error")

    # The CDF is the operational view: it answers "what fraction of targets
    # land inside a given tolerance", which is how learned IK is reported.
    xs = np.sort(err_mm)
    ax[1].plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=TRAIN_C, lw=1.8, label="empirical CDF")
    for tol, c in ((1.0, "#2a9d8f"), (5.0, "#457b9d"), (10.0, WARN_C), (50.0, VAL_C)):
        rate = float((err_mm <= tol).mean())
        ax[1].axvline(tol, color=c, ls=":", lw=1.2, label=f"≤{tol:g} mm: {rate*100:.1f}%")
    ax[1].set_xscale("log")
    ax[1].set_ylim(0, 1.02)
    _finish(ax[1], "tolerance  [mm]", "fraction of targets reached", "Success rate vs. tolerance", loc="upper left")
    fig.tight_layout()
    figs["error_distribution"] = fig

    # --- 3. constraints -----------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax[0].hist(orient_deg, bins=40, color=WARN_C, alpha=0.85, label=f"held-out targets (n={n_eval})")
    ax[0].axvline(orient_deg.mean(), color=VAL_C, lw=1.6, label=f"mean {orient_deg.mean():.0f}°")
    ax[0].axvline(90, color=REF_C, ls="--", lw=1.2, label="90° (uninformative)")
    _finish(ax[0], "orientation error  [deg]", "count",
            "Orientation error — now part of the loss")

    n = robot.n_joints
    idx = np.arange(n)
    rate = (violation > 0).mean(axis=0) * 100.0
    ax[1].bar(idx, rate, color=_joint_colors(n), label="configurations outside limits")
    ax[1].set_xticks(idx, [f"q{i+1}" for i in idx])
    ax[1].set_ylim(0, max(5.0, rate.max() * 1.2))
    _finish(ax[1], "joint", "violating configurations  [%]",
            f"Joint-limit violations — worst overshoot {violation.max():.2f} rad")
    fig.tight_layout()
    figs["constraints"] = fig
    return figs


# --------------------------------------------------------------------------
# Trajectory
# --------------------------------------------------------------------------
def trajectory_figures(result, robot, metrics: Dict, n_points: int = 200) -> "OrderedDict[str, plt.Figure]":
    """Three figures: kinematics, torque vs. true dynamics, convergence."""
    s = result.sample(n_points)
    t, q, qdot, qddot, tau_net = s["t"], s["q"], s["qdot"], s["qddot"], s["tau"]
    n = robot.n_joints
    colors = _joint_colors(n)

    with numpy_context():
        tau_true = np.stack([
            robot.dynamics.inverse_dynamics(q[i], qdot[i], qddot[i], GRAVITY, np.zeros(6))
            for i in range(n_points)
        ])
    figs = OrderedDict()

    # --- 1. kinematics ------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for j in range(n):
        ax[0].plot(t, q[:, j], color=colors[j], label=f"q{j+1}")
        ax[1].plot(t, qdot[:, j], color=colors[j], label=f"q̇{j+1}")
        ax[2].plot(t, qddot[:, j], color=colors[j], label=f"q̈{j+1}")
    ax[0].scatter([0, 1], [q[0, 0], q[-1, 0]], color=REF_C, zorder=5, s=18, label="endpoints (exact)")
    _finish(ax[0], "t  [normalized]", "position  [rad]", "Joint positions q(t)")
    ax[0].legend(fontsize=7, ncol=2)

    ax[1].axhline(0, color=REF_C, lw=0.9, ls="--", label="rest (target at both ends)")
    _finish(ax[1], "t  [normalized]", "velocity  [rad/s]",
            f"Joint velocities q̇(t) — peak {np.abs(qdot).max():.2f} rad/s")
    ax[1].legend(fontsize=7, ncol=2)

    _finish(ax[2], "t  [normalized]", "acceleration  [rad/s²]",
            f"Joint accelerations q̈(t) — peak {np.abs(qddot).max():.2f} rad/s²")
    ax[2].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    figs["kinematics"] = fig

    # --- 2. torque vs. ManipulaPy's true inverse dynamics -------------------
    # The training residual runs through a *learned* surrogate, so it can be
    # small while the trajectory is far from feasible. This is the figure that
    # says whether it actually is.
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for j in range(n):
        ax[0].plot(t, tau_net[:, j], color=colors[j], lw=1.4)
        ax[0].plot(t, tau_true[:, j], color=colors[j], lw=1.0, ls="--", alpha=0.75)
    ax[0].plot([], [], color=REF_C, lw=1.4, label="τ from the network")
    ax[0].plot([], [], color=REF_C, lw=1.0, ls="--", label="τ required (ManipulaPy inverse dynamics)")
    _finish(ax[0], "t  [normalized]", "torque  [N·m]",
            f"Commanded vs. required torque — {metrics.get('torque_relative_error', float('nan'))*100:.0f}% relative error")

    err = tau_net - tau_true
    for j in range(n):
        ax[1].plot(t, err[:, j], color=colors[j], label=f"joint {j+1}")
    ax[1].axhline(0, color=REF_C, ls="--", lw=1, label="dynamically consistent")
    _finish(ax[1], "t  [normalized]", "τ error  [N·m]",
            f"Torque error over time — RMSE {metrics.get('torque_rmse_Nm', float('nan')):.2f} N·m")
    ax[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    figs["torque"] = fig

    # --- 3. convergence -----------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    it = [h["iter"] for h in result.loss_history]
    for key, label, c in (("physics", "physics (EOM residual)", TRAIN_C),
                          ("effort", "control effort ‖τ‖²", WARN_C),
                          ("boundary", "boundary velocity", VAL_C)):
        ax[0].plot(it, [h[key] for h in result.loss_history], lw=1.3, color=c, label=label)
    ax[0].set_yscale("log")
    _finish(ax[0], "iteration", "loss term  [mixed units]", "Trajectory-solve convergence")

    # Nothing in the loss bounds the interior, so this is where an unusable
    # trajectory shows up: fast in the middle, at rest only at the endpoints.
    speed = np.abs(qdot).max(axis=1)
    ax[1].plot(t, speed, color=TRAIN_C, lw=1.6, label="peak joint speed")
    ax[1].axhline(2.6, color=VAL_C, ls="--", lw=1.4, label="Panda limit ≈ 2.6 rad/s")
    ax[1].fill_between(t, 2.6, speed, where=(speed > 2.6), color=VAL_C, alpha=0.18,
                       label="over limit")
    _finish(ax[1], "t  [normalized]", "speed  [rad/s]", "Velocity against the robot's limit")
    fig.tight_layout()
    figs["convergence"] = fig
    return figs


def save_figures(figures, output_dir, task: str, robot_name: str, dpi: int = 140):
    """Save a figure set as ``<task>_<robot>_<name>.png`` and close each figure.

    Closing matters: these scripts build a dozen figures per run, and matplotlib
    keeps every un-closed one alive, which warns and then leaks.
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fig in figures.items():
        path = output_dir / f"{task}_{robot_name}_{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
