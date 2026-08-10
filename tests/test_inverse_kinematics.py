# SPDX-License-Identifier: AGPL-3.0-or-later
import torch

from manipulapy_pinn.inverse_kinematics import InverseKinematicsPINN, train


def test_ik_pinn_output_shape(robot):
    model = InverseKinematicsPINN(robot.n_joints)
    target = torch.zeros(5, 3, dtype=torch.float64)
    out = model(target)
    assert out.shape == (5, robot.n_joints)
    assert torch.isfinite(out).all()


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
