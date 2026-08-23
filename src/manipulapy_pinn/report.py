# SPDX-License-Identifier: AGPL-3.0-or-later
"""Human-readable training reports.

Every ``train_*.py`` script writes one Markdown report per run into ``runs/``
alongside its checkpoint and figure. A report records what was run (full
configuration and environment), what happened (loss trajectory and wall
time), how the result scores (the full :mod:`metrics` block), and — the part
that makes it worth reading — **what looks wrong**.

That last section exists because the numbers in this package are easy to
misread. A forward-dynamics model can post a plausible RMSE while barely
beating a constant predictor; an inverse-kinematics model can hit its target
with a configuration the robot physically cannot adopt; a trajectory can
drive its training residual to nearly zero and still be far from
dynamically feasible. Each of those is invisible in a loss curve and obvious
in a threshold check, so the checks are run automatically and their verdicts
written into the report rather than left for the reader to derive.
"""
from __future__ import annotations

import importlib.metadata as _md
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


def _versions() -> Dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("ManipulaPy", "torch", "numpy", "manipulapy-pinn"):
        try:
            out[pkg] = _md.version(pkg)
        except Exception:  # pragma: no cover - only when running from a source tree
            out[pkg] = "unknown"
    return out


def _fmt(v) -> str:
    if isinstance(v, np.ndarray):
        return "[" + ", ".join(f"{x:.4g}" for x in v) + "]"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _loss_table(history: Sequence[dict], n_rows: int = 8) -> str:
    """A short table of evenly spaced rows from a loss history."""
    if not history:
        return "_(no loss history recorded)_\n"
    keys = [k for k in history[0] if k != "iter"]
    idx = np.unique(np.linspace(0, len(history) - 1, min(n_rows, len(history))).astype(int))
    lines = ["| iter | " + " | ".join(keys) + " |",
             "|---:|" + "---:|" * len(keys)]
    for i in idx:
        row = history[int(i)]
        lines.append(f"| {row['iter']} | " + " | ".join(f"{row[k]:.6g}" for k in keys) + " |")
    return "\n".join(lines) + "\n"


def review_forward_dynamics(m: Dict) -> List[str]:
    """Threshold checks a forward-dynamics loss curve cannot show."""
    notes = []
    if m.get("skill_score", 1.0) < 0.5:
        notes.append(
            f"**Weak skill score ({m['skill_score']:.2f}).** The model reduces error only "
            f"{m['skill_score'] * 100:.0f}% below a constant predictor that always outputs the "
            f"mean acceleration ({m['baseline_rmse_rad_s2']:.2f} rad/s²). It has learned little "
            f"of the state dependence — train longer, and especially on more samples."
        )
    nrmse = m.get("nrmse_per_joint")
    if nrmse is not None and len(nrmse) and float(np.max(nrmse)) > 0.7:
        j = int(np.argmax(nrmse))
        notes.append(
            f"**Joint {j} is poorly modelled** (normalized RMSE {float(nrmse[j]):.2f}; a value of "
            f"1.0 means no better than predicting that joint's mean). Check its mass-matrix "
            f"conditioning before trusting this model — see the URDF caveat in `robots.py`."
        )
    if m.get("r2", 1.0) < 0.9:
        notes.append(
            f"**R² = {m['r2']:.3f}** on held-out states. Below ~0.9 the surrogate is not accurate "
            f"enough to stand in for the true dynamics in `trajectory.solve`, whose result can be "
            f"no better than the model it optimizes through."
        )
    return notes


def review_inverse_kinematics(m: Dict) -> List[str]:
    """Threshold checks an FK-residual loss cannot show."""
    notes = []
    if m.get("joint_limit_violation_rate", 0.0) > 0.01:
        notes.append(
            f"**{m['joint_limit_violation_rate'] * 100:.1f}% of predicted configurations violate "
            f"the robot's joint limits** (worst overshoot {m['joint_limit_max_violation_rad']:.3f} "
            f"rad). Nothing in the network or the FK-residual loss constrains its output to the "
            f"reachable range, so a configuration that reaches the target may still be one the "
            f"robot cannot adopt. Any use of these outputs on hardware — or as a solver seed — "
            f"must clamp or penalize this."
        )
    if m.get("success_rate_10mm", 0.0) < 0.5:
        notes.append(
            f"**Only {m['success_rate_10mm'] * 100:.0f}% of targets are reached within 10 mm** "
            f"(median error {m['median_mm']:.1f} mm). This is the expected regime for a "
            f"single-forward-pass IK network and does not by itself make the model useless: it is "
            f"still viable as a warm start for a numerical solver, which converges from seeds far "
            f"worse than this. It is not usable as a final answer."
        )
    if m.get("orientation_mean_deg", 0.0) > 30:
        notes.append(
            f"**End-effector orientation is off by {m['orientation_mean_deg']:.0f}° on average.** "
            f"Expected, not a regression: the loss matches position only, so orientation is "
            f"entirely unconstrained. Stated here so the limitation is not mistaken for a result."
        )
    if m.get("p95_mm", 0) > 3 * max(m.get("median_mm", 1), 1e-9):
        notes.append(
            f"**Heavy tail:** p95 error ({m['p95_mm']:.1f} mm) is more than 3× the median "
            f"({m['median_mm']:.1f} mm). Report the median, not the mean, and expect a subset of "
            f"targets to be much harder than typical."
        )
    return notes


def review_trajectory(m: Dict) -> List[str]:
    """Threshold checks the surrogate-based training residual cannot show."""
    notes = []
    rel = m.get("torque_relative_error")
    if rel is not None and rel > 0.25:
        notes.append(
            f"**The trajectory is not dynamically consistent under real physics.** The torque "
            f"network's output differs from the torque ManipulaPy's `inverse_dynamics` says the "
            f"trajectory actually requires by {rel * 100:.0f}% of the true torque magnitude "
            f"(RMSE {m['torque_rmse_Nm']:.1f} N·m; peak true torque "
            f"{m['peak_true_torque_Nm']:.1f} N·m vs peak network torque "
            f"{m['peak_network_torque_Nm']:.1f} N·m). The training residual is measured through a "
            f"*learned* surrogate, so it can be small while this is large."
        )
    ratio = m.get("dense_vs_collocation_ratio")
    if ratio is not None and ratio > 1.5:
        notes.append(
            f"**Overfitted to the collocation points.** Torque error on a dense grid is "
            f"{ratio:.1f}× the error on the training collocation points. The residual is being "
            f"satisfied *at* the points and not between them — add more collocation points, or "
            f"resample them each iteration instead of reusing a fixed grid."
        )
    for key, label in (("boundary_qdot_start_rad_s", "start"), ("boundary_qdot_end_rad_s", "end")):
        if m.get(key, 0.0) > 0.05:
            notes.append(
                f"**Non-zero velocity at trajectory {label}** ({m[key]:.3f} rad/s). This is a soft "
                f"penalty, not a hard constraint, so it is never satisfied exactly — raise "
                f"`boundary_velocity_weight` if it matters."
            )
    if m.get("peak_speed_rad_s", 0.0) > 2.6:
        notes.append(
            f"**Peak joint speed {m['peak_speed_rad_s']:.2f} rad/s** exceeds a Franka Panda's "
            f"~2.6 rad/s limit. Nothing in the loss bounds the interior of the trajectory — this "
            f"is the documented missing velocity/torque constraint, visible in a real run."
        )
    return notes


def training_report(
    *,
    title: str,
    robot_name: str,
    n_joints: int,
    config: Dict,
    metrics: Dict,
    history: Sequence[dict] = (),
    elapsed_s: Optional[float] = None,
    figure: Optional[str] = None,
    checkpoint: Optional[str] = None,
    notes: Sequence[str] = (),
) -> str:
    """Render a full Markdown training report."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    v = _versions()

    out = [f"# {title}", "", f"**Robot:** {robot_name} ({n_joints} DOF)  ", f"**Run:** {stamp}  "]
    if elapsed_s is not None:
        out.append(f"**Training wall time:** {elapsed_s:.1f} s  ")
    out += ["", "## Configuration", "", "| parameter | value |", "|---|---|"]
    out += [f"| `{k}` | {_fmt(val)} |" for k, val in config.items()]

    out += ["", "## Environment", "", "| component | version |", "|---|---|"]
    out += [f"| {k} | {val} |" for k, val in v.items()]

    if history:
        out += ["", "## Loss trajectory", "", _loss_table(history)]
        first, last = history[0], history[-1]
        if "loss" in first:
            change = (
                f"{first['loss']:.6g} → {last['loss']:.6g} "
                f"({first['loss'] / last['loss']:.1f}× reduction)"
                if last["loss"] > 0 else f"{first['loss']:.6g} → {last['loss']:.6g}"
            )
            out.append(f"Total loss: {change}")

    out += ["", "## Metrics", "", "| metric | value |", "|---|---|"]
    out += [f"| `{k}` | {_fmt(val)} |" for k, val in metrics.items()]

    all_notes = list(notes)
    out += ["", "## Review", ""]
    if all_notes:
        out.append(
            f"{len(all_notes)} issue{'s' if len(all_notes) != 1 else ''} flagged by automatic "
            f"threshold checks. These are not necessarily errors — several are documented "
            f"limitations of the method — but each one is something a loss curve would hide.\n"
        )
        out += [f"{i}. {n}\n" for i, n in enumerate(all_notes, 1)]
    else:
        out.append("No automatic threshold checks were triggered for this run.")

    # `figure` may be a single name or a list of them, since each task now
    # writes a set rather than one combined plot.
    figure_names = [figure] if isinstance(figure, str) else list(figure or [])
    artifacts = [a for a in [checkpoint, *figure_names] if a]
    if artifacts:
        out += ["", "## Artifacts", ""] + [f"- `{a}`" for a in artifacts]

    out += ["", "---", "",
            "_Generated by `manipulapy_pinn.report`. Ground truth throughout is ManipulaPy._"]
    return "\n".join(out) + "\n"


def report_path(output_dir, task: str, robot_name: str) -> Path:
    """``runs/<task>_<robot>.md`` — the same naming as the checkpoint and figure.

    Deliberately a stable name rather than a timestamped one, so a report
    overwrites in place like the other per-run artifacts and ``runs/`` does
    not accumulate. Keep a report you care about by copying it out, or by
    committing it — ``runs/`` is gitignored.
    """
    return Path(output_dir) / f"{task}_{robot_name}.md"


def write_report(path: Path, text: str) -> Path:
    """Write a report to ``path``, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
