# SPDX-License-Identifier: AGPL-3.0-or-later
"""Metrics and report tests.

Deliberately tiny training configurations — these check that each metric is
computed, finite, and internally consistent, not that any model is good.
"""
import numpy as np
import pytest
import torch

from manipulapy_pinn.forward_dynamics import train as train_fd
from manipulapy_pinn.inverse_kinematics import train as train_ik
from manipulapy_pinn.metrics import (
    format_metrics,
    forward_dynamics_metrics,
    inverse_kinematics_metrics,
    trajectory_metrics,
)
from manipulapy_pinn.report import (
    report_path,
    review_forward_dynamics,
    review_inverse_kinematics,
    review_trajectory,
    training_report,
    write_report,
)
from manipulapy_pinn.trajectory import solve


@pytest.fixture(scope="module")
def fd_result(robot):
    return train_fd(robot, n_samples=120, iterations=15, verbose=False)


def test_forward_dynamics_metrics_are_finite_and_consistent(robot, fd_result, rng):
    m = forward_dynamics_metrics(fd_result.model, robot, n_samples=40, rng=rng)

    assert m["n_samples"] == 40
    assert len(m["per_joint_rmse_rad_s2"]) == robot.n_joints
    assert len(m["nrmse_per_joint"]) == robot.n_joints
    for k, v in m.items():
        assert np.all(np.isfinite(v)), f"{k} is not finite"

    # MAE <= RMSE <= max abs error, always.
    assert m["mae_rad_s2"] <= m["rmse_rad_s2"] + 1e-9
    assert m["rmse_rad_s2"] <= m["max_abs_error_rad_s2"] + 1e-9
    # The aggregate RMSE is the quadratic mean of the per-joint RMSEs.
    assert m["rmse_rad_s2"] == pytest.approx(
        float(np.sqrt((m["per_joint_rmse_rad_s2"] ** 2).mean())), rel=1e-9
    )
    # Skill score and R2 are tied to the same error, so they agree in sign.
    assert (m["skill_score"] > 0) == (m["r2"] > 0)


def test_forward_dynamics_metrics_are_perfect_for_an_oracle(robot, rng):
    """A model returning the true accelerations must score exactly zero error."""

    class Oracle(torch.nn.Module):
        """Reproduces ManipulaPy's own forward dynamics, so error must vanish."""

        def forward(self, q, qdot, tau):
            from manipulapy_pinn.backend_utils import numpy_context

            g = np.array([0.0, 0.0, -9.81])
            with numpy_context():
                out = np.stack([
                    robot.dynamics.forward_dynamics(
                        q[i].numpy(), qdot[i].numpy(), tau[i].numpy(), g, np.zeros(6)
                    )
                    for i in range(len(q))
                ])
            return torch.tensor(out, dtype=torch.float64)

    m = forward_dynamics_metrics(Oracle(), robot, n_samples=15, rng=rng)
    assert m["rmse_rad_s2"] == pytest.approx(0.0, abs=1e-9)
    assert m["torque_rmse_Nm"] == pytest.approx(0.0, abs=1e-9)
    assert m["r2"] == pytest.approx(1.0, abs=1e-9)
    assert m["skill_score"] == pytest.approx(1.0, abs=1e-9)


def test_inverse_kinematics_metrics(robot, rng):
    result = train_ik(robot, iterations=15, batch_size=8, verbose=False)
    m = inverse_kinematics_metrics(result.model, robot, n_targets=30, rng=rng)

    assert m["n_targets"] == 30
    for k, v in m.items():
        assert np.all(np.isfinite(v)), f"{k} is not finite"

    # Distribution ordering must hold by construction.
    assert m["median_mm"] <= m["p95_mm"] <= m["max_mm"] + 1e-9
    # Success rates are monotone in the tolerance, and are proper fractions.
    rates = [m["success_rate_1mm"], m["success_rate_5mm"],
             m["success_rate_10mm"], m["success_rate_50mm"]]
    assert rates == sorted(rates)
    assert all(0.0 <= r <= 1.0 for r in rates)
    assert 0.0 <= m["orientation_mean_deg"] <= 180.0
    assert 0.0 <= m["joint_limit_violation_rate"] <= 1.0


def test_trajectory_metrics_against_true_inverse_dynamics(robot, rng, fd_result):
    q_start = robot.sample_configurations(1, rng)[0]
    q_goal = robot.sample_configurations(1, rng)[0]
    traj = solve(q_start, q_goal, fd_result.model, n_collocation=6,
                 iterations=10, verbose=False)

    m = trajectory_metrics(traj, robot, n_points=20, n_collocation_trained=6)
    for k, v in m.items():
        assert np.all(np.isfinite(v)), f"{k} is not finite"

    # Position boundary conditions hold exactly, by the ansatz construction —
    # this is the property that makes them worth a hard constraint.
    assert m["boundary_q_start_err_rad"] == pytest.approx(0.0, abs=1e-12)
    assert m["boundary_q_goal_err_rad"] == pytest.approx(0.0, abs=1e-12)
    assert m["torque_rmse_Nm"] >= 0.0
    assert "dense_vs_collocation_ratio" in m


def test_reviews_flag_bad_metrics_and_pass_good_ones():
    assert review_forward_dynamics(
        {"skill_score": 0.99, "r2": 0.99, "nrmse_per_joint": np.array([0.1, 0.1])}
    ) == []
    assert review_forward_dynamics(
        {"skill_score": 0.1, "baseline_rmse_rad_s2": 10.0, "r2": 0.5,
         "nrmse_per_joint": np.array([0.9, 0.1])}
    )

    assert review_inverse_kinematics(
        {"joint_limit_violation_rate": 0.0, "success_rate_10mm": 0.99,
         "orientation_mean_deg": 1.0, "p95_mm": 1.0, "median_mm": 1.0}
    ) == []
    limits = review_inverse_kinematics(
        {"joint_limit_violation_rate": 0.9, "joint_limit_max_violation_rad": 2.0,
         "success_rate_10mm": 0.99, "orientation_mean_deg": 1.0,
         "p95_mm": 1.0, "median_mm": 1.0}
    )
    assert len(limits) == 1 and "joint limits" in limits[0]

    assert review_trajectory(
        {"torque_relative_error": 0.01, "dense_vs_collocation_ratio": 1.0,
         "boundary_qdot_start_rad_s": 0.0, "boundary_qdot_end_rad_s": 0.0,
         "peak_speed_rad_s": 1.0}
    ) == []
    assert review_trajectory(
        {"torque_relative_error": 0.9, "torque_rmse_Nm": 5.0, "peak_true_torque_Nm": 10.0,
         "peak_network_torque_Nm": 1.0, "dense_vs_collocation_ratio": 3.0,
         "boundary_qdot_start_rad_s": 0.0, "boundary_qdot_end_rad_s": 0.0,
         "peak_speed_rad_s": 1.0}
    )


def test_training_report_renders_and_writes(tmp_path, robot, fd_result, rng):
    m = forward_dynamics_metrics(fd_result.model, robot, n_samples=15, rng=rng)
    text = training_report(
        title="Test run", robot_name=robot.name, n_joints=robot.n_joints,
        config={"iterations": 15}, metrics=m, history=fd_result.loss_history,
        elapsed_s=1.0, figure="f.png", checkpoint="c.pt",
        notes=review_forward_dynamics(m),
    )
    for section in ("# Test run", "## Configuration", "## Environment",
                    "## Loss trajectory", "## Metrics", "## Review", "## Artifacts"):
        assert section in text
    assert robot.name in text

    path = write_report(report_path(tmp_path, "forward_dynamics", robot.name), text)
    assert path.exists() and path.name == f"forward_dynamics_{robot.name}.md"
    assert path.read_text(encoding="utf-8") == text


def test_format_metrics_handles_arrays_and_scalars():
    out = format_metrics({"a": 1.5, "b": np.array([1.0, 2.0]), "c": 7}, "T")
    assert "T" in out and "1.5" in out and "1.000, 2.000" in out and "7" in out


def test_split_metrics_score_the_real_splits(robot, fd_result):
    """Per-split metrics use the run's own data, not a fresh draw."""
    from manipulapy_pinn.metrics import forward_dynamics_split_metrics

    assert set(fd_result.splits) == {"train", "test", "eval"}
    sm = forward_dynamics_split_metrics(fd_result.model, robot, fd_result.splits)

    for name in ("train", "test", "eval"):
        assert sm[name]["n_samples"] == len(fd_result.splits[name]["q"]), \
            "metrics were computed on a different set than the split"
        assert np.isfinite(sm[name]["rmse_rad_s2"])

    gaps = sm["gaps"]
    assert gaps["test_over_train"] == pytest.approx(
        sm["test"]["rmse_rad_s2"] / sm["train"]["rmse_rad_s2"], rel=1e-9)
    assert gaps["eval_over_test"] == pytest.approx(
        sm["eval"]["rmse_rad_s2"] / sm["test"]["rmse_rad_s2"], rel=1e-9)


def test_metrics_accept_explicit_data(robot, fd_result, rng):
    """Passing `data` scores that set; omitting it samples a fresh one."""
    from manipulapy_pinn.metrics import forward_dynamics_metrics

    on_eval = forward_dynamics_metrics(fd_result.model, robot, data=fd_result.splits["eval"])
    again = forward_dynamics_metrics(fd_result.model, robot, data=fd_result.splits["eval"])
    assert on_eval["rmse_rad_s2"] == again["rmse_rad_s2"], "scoring a fixed set must be deterministic"

    fresh = forward_dynamics_metrics(fd_result.model, robot, n_samples=30, rng=rng)
    assert fresh["n_samples"] == 30
