# SPDX-License-Identifier: AGPL-3.0-or-later
import torch

from manipulapy_pinn.forward_dynamics import ForwardDynamicsPINN, train


def test_forward_dynamics_pinn_output_shape(robot):
    model = ForwardDynamicsPINN(robot.n_joints)
    n = robot.n_joints
    q = torch.zeros(5, n, dtype=torch.float64)
    qdot = torch.zeros(5, n, dtype=torch.float64)
    tau = torch.zeros(5, n, dtype=torch.float64)
    out = model(q, qdot, tau)
    assert out.shape == (5, n)
    assert torch.isfinite(out).all()


def test_forward_dynamics_training_reduces_loss_and_stays_finite(robot):
    result = train(
        robot, n_samples=40, iterations=30,
        hidden=(16, 16), log_every=1000, verbose=False,
    )
    losses = [h["loss"] for h in result.loss_history]
    assert len(losses) == 30
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)
    # Loss need not be monotone step-to-step, but should trend down.
    assert losses[-1] < losses[0]
    assert result.val_data_rmse == result.val_data_rmse  # not NaN
