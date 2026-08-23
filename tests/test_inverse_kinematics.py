# SPDX-License-Identifier: AGPL-3.0-or-later
import pytest
import torch

from manipulapy_pinn.inverse_kinematics import InverseKinematicsPINN, train


def test_ik_pinn_takes_a_full_pose(robot):
    """Input is position plus flattened rotation, not position alone."""
    model = InverseKinematicsPINN(robot.n_joints)
    assert model.POSE_DIM == 12

    pose = torch.zeros(5, model.POSE_DIM, dtype=torch.float64)
    out = model(pose)
    assert out.shape == (5, robot.n_joints)
    assert torch.isfinite(out).all()

    with pytest.raises(RuntimeError):
        model(torch.zeros(5, 3, dtype=torch.float64))


def test_pose_features_layout(robot, rng):
    """pose_features packs [position, R row-major] and round-trips."""
    from manipulapy_pinn.physics import pose_features

    q = robot.sample_configurations(4, rng)
    poses = torch.tensor(robot.forward_kinematics(q), dtype=torch.float64)
    feats = pose_features(poses)

    assert feats.shape == (4, 12)
    assert torch.allclose(feats[:, :3], poses[:, :3, 3])
    assert torch.allclose(feats[:, 3:].reshape(4, 3, 3), poses[:, :3, :3])


def test_orientation_residual_is_zero_only_at_the_target(robot, rng):
    """The orientation term must actually discriminate, not just be finite."""
    from manipulapy_pinn.backend_utils import torch_context
    from manipulapy_pinn.physics import fk_pose_residual

    q = robot.sample_configurations(3, rng)
    poses = torch.tensor(robot.forward_kinematics(q), dtype=torch.float64)
    qt = torch.tensor(q, dtype=torch.float64)

    with torch_context():
        _, exact = fk_pose_residual(robot.serial, qt, poses[:, :3, 3], poses[:, :3, :3])
        # Rotate every target by a fixed amount: the residual must grow.
        flipped = poses[:, :3, :3].flip(-1)
        _, wrong = fk_pose_residual(robot.serial, qt, poses[:, :3, 3], flipped)

    assert torch.allclose(exact, torch.zeros_like(exact), atol=1e-12)
    assert (wrong > 1e-6).all()


def test_ik_training_reduces_loss_and_stays_finite(robot):
    result = train(
        robot, iterations=60, batch_size=8, n_val=8,
        hidden=(16, 16), log_every=1000, verbose=False, seed=0,
    )
    losses = [h["loss"] for h in result.loss_history]
    assert len(losses) == 60
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)
    # Every step draws a fresh random batch, so individual steps are noisy —
    # compare early/late averages instead of first-vs-last to avoid flaking.
    early = sum(losses[:10]) / 10
    late = sum(losses[-10:]) / 10
    assert late < early
    assert result.val_position_rmse == result.val_position_rmse  # not NaN
    assert result.val_orientation_deg == result.val_orientation_deg
    assert 0.0 <= result.val_orientation_deg <= 180.0
    # Both terms are recorded so their balance can be inspected after the fact.
    assert {"position", "orientation"} <= set(result.loss_history[0])
