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



#: Status colours, reserved for pass/fail state and never reused as series
#: colours. A metric's own hue never encodes whether it is good.
OK_C, WARN_C_S, BAD_C = "#1b7f5f", "#b8860b", "#a4243b"


def _scorecard(ax, rows, title):
    """Every checked metric against *its own* threshold, on one honest axis.

    The metrics have incompatible units — rad/s², mm, degrees, dimensionless
    ratios — so a shared value axis would be meaningless. Plotting
    ``value / threshold`` instead puts them on one axis that does mean
    something: distance from the line the automatic review draws. 1.0 is the
    line; which side is bad depends on the metric, so the bars are coloured by
    outcome rather than by direction.
    """
    from .report import threshold_status

    labels, ratios, colors, notes = [], [], [], []
    for key, value in rows:
        status = threshold_status(key, value)
        if status is None:
            continue
        passes, threshold, direction = status
        ratio = value / threshold if threshold else float("nan")
        labels.append(key)
        ratios.append(max(ratio, 1e-3))
        colors.append(OK_C if passes else BAD_C)
        notes.append(f"{value:.4g} vs {threshold:g} ({'≥' if direction == 'below' else '≤'})")

    if not labels:
        ax.text(0.5, 0.5, "no checked metrics for this task", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color=REF_C)
        ax.set_axis_off()
        return

    y = np.arange(len(labels))
    ax.barh(y, ratios, color=colors, height=0.62)
    ax.axvline(1.0, color=REF_C, lw=1.4, ls="--")
    for yi, (r, note) in enumerate(zip(ratios, notes)):
        ax.text(r * 1.06, yi, note, va="center", fontsize=7.5, color=REF_C)
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(min(ratios) * 0.5, max(ratios) * 4)
    ax.invert_yaxis()
    # Legend proxies: the bars carry state, so the legend must name the states.
    # These have to be real Patch artists — barh([], []) adds no patch, and the
    # legend then falls back to a default colour for every entry.
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=OK_C, label="passes"),
               Patch(facecolor=BAD_C, label="fails"),
               plt.Line2D([], [], color=REF_C, ls="--", lw=1.4, label="review threshold")]
    ax.set_xlabel("value ÷ its own threshold  [dimensionless]")
    ax.set_title(title, fontsize=10.5)
    ax.grid(alpha=0.3, axis="x")
    # Below the axes: any in-axes corner collides with a bar for some run.
    ax.legend(handles=handles, fontsize=8, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), frameon=False)


def _stat_tiles(ax, items, title):
    """Scalars with no threshold, as labelled tiles rather than a fake chart.

    These have no common scale and no reference line, so a bar chart of them
    would invent a comparison that does not exist. A tile grid reports the
    number and its unit, which is all there is to say.
    """
    ax.set_axis_off()
    ax.set_title(title, fontsize=10.5)
    if not items:
        return
    cols = 2
    rows = (len(items) + cols - 1) // cols
    for i, (label, value, unit) in enumerate(items):
        cx, cy = (i % cols) / cols, 1.0 - (i // cols + 1) / rows
        ax.add_patch(plt.Rectangle((cx + 0.01, cy + 0.02), 1 / cols - 0.02, 1 / rows - 0.04,
                                   transform=ax.transAxes, facecolor="#00000008",
                                   edgecolor="#00000022", lw=0.8))
        # Label sits just inside the tile top, value just inside the bottom, so
        # the gap between them grows with the tile rather than being fixed.
        value_size = min(13.0, 62.0 / rows)
        ax.text(cx + 0.035, cy + 1 / rows - 0.05, label, transform=ax.transAxes,
                fontsize=7.5, color=REF_C, va="top")
        ax.text(cx + 0.035, cy + 0.05, f"{value:.4g}", transform=ax.transAxes,
                fontsize=value_size, fontweight="bold", va="bottom")
        ax.text(cx + 1 / cols - 0.04, cy + 0.05, unit, transform=ax.transAxes,
                fontsize=8, color=REF_C, va="bottom", ha="right")



#: Categorical hues for the three data splits, in fixed order and never cycled.
#: Validated with the dataviz palette checker: worst adjacent pair ΔE 13.5
#: (deutan) and 29.9 (normal vision), chroma floor and contrast both passing.
#: Two plausible-looking alternatives failed CVD separation at ΔE 2.9 and 5.9 —
#: indistinguishable to a deuteranope — which is why this is measured, not
#: chosen by eye.
SPLIT_COLORS = {"train": "#0077BB", "test": "#CC3311", "eval": "#009988"}
SPLIT_ORDER = ("train", "test", "eval")


def split_comparison_figure(split_metrics: Dict, n_joints: int) -> plt.Figure:
    """Train vs. test vs. eval on the same metrics — the generalization picture.

    Four panels rather than one grouped chart, because the metrics do not share
    a unit: error in rad/s², the scale-free scores, per-joint error, and the
    ratios between splits each get their own axis.
    """
    present = [k for k in SPLIT_ORDER if k in split_metrics]
    colors = [SPLIT_COLORS[k] for k in present]
    x = np.arange(len(present))

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    # --- error in physical units ---
    rmse = [split_metrics[k]["rmse_rad_s2"] for k in present]
    mae = [split_metrics[k]["mae_rad_s2"] for k in present]
    ax[0, 0].bar(x - 0.19, rmse, width=0.36, color=colors, label="RMSE")
    ax[0, 0].bar(x + 0.19, mae, width=0.36, color=colors, alpha=0.5, label="MAE")
    for xi, v in zip(x, rmse):
        ax[0, 0].text(xi - 0.19, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax[0, 0].set_xticks(x, [f"{k}\n(n={split_metrics[k]['n_samples']})" for k in present])
    _finish(ax[0, 0], "split", "error  [rad/s²]", "Error by split — solid RMSE, faded MAE")

    # --- scale-free scores, both bounded above by 1 ---
    r2 = [split_metrics[k]["r2"] for k in present]
    skill = [split_metrics[k]["skill_score"] for k in present]
    ax[0, 1].bar(x - 0.19, r2, width=0.36, color=colors, label="R²")
    ax[0, 1].bar(x + 0.19, skill, width=0.36, color=colors, alpha=0.5, label="skill score")
    ax[0, 1].axhline(0.9, color=REF_C, ls="--", lw=1.1, label="R² threshold 0.9")
    ax[0, 1].axhline(0.5, color=REF_C, ls=":", lw=1.1, label="skill threshold 0.5")
    ax[0, 1].set_xticks(x, present)
    ax[0, 1].set_ylim(0, 1.05)
    _finish(ax[0, 1], "split", "score  [dimensionless]", "Scale-free scores by split")

    # --- per joint, so a split-wide gap can be traced to specific joints ---
    j = np.arange(n_joints)
    width = 0.8 / len(present)
    for i, k in enumerate(present):
        ax[1, 0].bar(j + (i - (len(present) - 1) / 2) * width,
                     split_metrics[k]["per_joint_rmse_rad_s2"], width=width,
                     color=SPLIT_COLORS[k], label=k)
    ax[1, 0].set_xticks(j, [f"q{i+1}" for i in j])
    _finish(ax[1, 0], "joint", "RMSE  [rad/s²]", "Per-joint error by split")

    # --- the ratios that actually answer "did it generalize" ---
    gaps = split_metrics.get("gaps", {})
    labels = [k for k in ("test_over_train", "eval_over_train", "eval_over_test") if k in gaps]
    values = [gaps[k] for k in labels]
    bar_colors = [OK_C if v <= 1.25 else BAD_C for v in values]
    ax[1, 1].bar(range(len(labels)), values, color=bar_colors, width=0.55)
    ax[1, 1].axhline(1.0, color=REF_C, ls="--", lw=1.4, label="no gap")
    for xi, v in enumerate(values):
        ax[1, 1].text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax[1, 1].set_xticks(range(len(labels)), [l.replace("_over_", " / ") for l in labels], fontsize=8)
    _finish(ax[1, 1], "", "ratio of RMSE  [dimensionless]",
            "Generalization gaps\n(eval/test above 1 means the test split flattered the model)")

    fig.tight_layout()
    return fig


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

    # --- 5. torque-space error ---------------------------------------------
    # The acceleration error mapped through the true mass matrix, which is the
    # unit the robot is actually commanded in and weights each joint by inertia.
    with numpy_context():
        torque_err = np.stack([robot.dynamics.mass_matrix(data["q"][i]) @ err[i]
                               for i in range(len(err))])
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax[0].hist(torque_err.flatten(), bins=60, color=TRAIN_C, alpha=0.85, label="M(q)·Δq̈")
    ax[0].axvline(0, color=REF_C, ls="--", lw=1, label="zero error")
    _finish(ax[0], "torque-space error  [N·m]", "count",
            f"Error in commanded units — RMSE {metrics['torque_rmse_Nm']:.2f} N·m, "
            f"max {metrics['torque_max_abs_Nm']:.1f}")

    idx = np.arange(robot.n_joints)
    ax[1].bar(idx, np.sqrt((torque_err ** 2).mean(axis=0)), color=_joint_colors(robot.n_joints),
              label="per-joint torque RMSE")
    ax[1].axhline(metrics["torque_rmse_Nm"], color=VAL_C, ls="--", lw=1.2,
                  label=f"aggregate {metrics['torque_rmse_Nm']:.2f} N·m")
    ax[1].set_xticks(idx, [f"q{i+1}" for i in idx])
    _finish(ax[1], "joint", "torque RMSE  [N·m]", "Torque-space error by joint")
    fig.tight_layout()
    figs["torque_space"] = fig

    # --- 6. scorecard -------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.3))
    nrmse = metrics.get("nrmse_per_joint")
    checked = [("skill_score", metrics.get("skill_score")), ("r2", metrics.get("r2")),
               ("nrmse_worst_joint", float(np.max(nrmse)) if nrmse is not None else None)]
    _scorecard(ax[0], checked, "Automatic review checks")
    _stat_tiles(ax[1], [
        ("rmse", metrics["rmse_rad_s2"], "rad/s²"),
        ("mae", metrics["mae_rad_s2"], "rad/s²"),
        ("max abs error", metrics["max_abs_error_rad_s2"], "rad/s²"),
        ("baseline rmse", metrics["baseline_rmse_rad_s2"], "rad/s²"),
        ("torque rmse", metrics["torque_rmse_Nm"], "N·m"),
        ("eval samples", metrics["n_samples"], "count"),
    ], "Unchecked scalars")
    fig.tight_layout()
    figs["summary"] = fig
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

    # --- 4. scorecard -------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.3))
    _scorecard(ax[0], [(k, metrics.get(k)) for k in
                       ("success_rate_10mm", "joint_limit_violation_rate", "orientation_mean_deg")],
               "Automatic review checks")
    _stat_tiles(ax[1], [
        ("mean reach error", metrics["mean_mm"], "mm"),
        ("median", metrics["median_mm"], "mm"),
        ("p95", metrics["p95_mm"], "mm"),
        ("max", metrics["max_mm"], "mm"),
        ("success ≤1 mm", metrics["success_rate_1mm"] * 100, "%"),
        ("success ≤5 mm", metrics["success_rate_5mm"] * 100, "%"),
        ("success ≤50 mm", metrics["success_rate_50mm"] * 100, "%"),
        ("orientation max", metrics["orientation_max_deg"], "deg"),
        ("worst limit overshoot", metrics["joint_limit_max_violation_rad"], "rad"),
        ("eval targets", metrics["n_targets"], "count"),
    ], "Unchecked scalars")
    fig.tight_layout()
    figs["summary"] = fig
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

    # --- 4. consistency: boundary conditions and collocation overfitting ----
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    bc_labels = ["q(0)", "q(T)", "q̇(0)", "q̇(T)"]
    bc_values = [metrics.get("boundary_q_start_err_rad", 0.0),
                 metrics.get("boundary_q_goal_err_rad", 0.0),
                 metrics.get("boundary_qdot_start_rad_s", 0.0),
                 metrics.get("boundary_qdot_end_rad_s", 0.0)]
    # Position endpoints are exact by construction and land near float epsilon;
    # a log axis is the only way to show them next to the soft velocity penalty.
    ax[0].bar(bc_labels, [max(v, 1e-17) for v in bc_values],
              color=[OK_C, OK_C, WARN_C, WARN_C], label="boundary error")
    ax[0].axhline(0.05, color=REF_C, ls="--", lw=1.2, label="velocity threshold 0.05")
    ax[0].set_yscale("log")
    _finish(ax[0], "boundary condition", "error  [rad or rad/s]",
            "Hard (position) vs. soft (velocity) constraints")

    dense = metrics.get("torque_rmse_Nm", float("nan"))
    colloc = metrics.get("torque_rmse_on_collocation_Nm", float("nan"))
    ax[1].bar(["training\ncollocation points", "dense\ngrid"], [colloc, dense],
              color=[TRAIN_C, VAL_C], label="torque RMSE vs. true dynamics")
    ratio = metrics.get("dense_vs_collocation_ratio", float("nan"))
    _finish(ax[1], "evaluated on", "torque RMSE  [N·m]",
            f"Collocation overfitting check — ratio {ratio:.2f}\n(≫1 means fitted to the points, wrong between them)")
    fig.tight_layout()
    figs["consistency"] = fig

    # --- 5. scorecard -------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.3))
    _scorecard(ax[0], [(k, metrics.get(k)) for k in
                       ("torque_relative_error", "dense_vs_collocation_ratio",
                        "boundary_qdot_start_rad_s", "boundary_qdot_end_rad_s",
                        "peak_speed_rad_s")],
               "Automatic review checks")
    _stat_tiles(ax[1], [
        ("torque rmse", metrics["torque_rmse_Nm"], "N·m"),
        ("torque max error", metrics["torque_max_abs_error_Nm"], "N·m"),
        ("peak true torque", metrics["peak_true_torque_Nm"], "N·m"),
        ("peak network torque", metrics["peak_network_torque_Nm"], "N·m"),
        ("peak acceleration", metrics["peak_accel_rad_s2"], "rad/s²"),
        ("path length", metrics["path_length_rad"], "rad"),
        ("surrogate eval rmse", metrics.get("surrogate_eval_rmse_rad_s2", float("nan")), "rad/s²"),
        ("sample points", metrics["n_points"], "count"),
    ], "Unchecked scalars")
    fig.tight_layout()
    figs["summary"] = fig
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
