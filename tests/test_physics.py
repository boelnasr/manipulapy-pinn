# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np
import torch

from manipulapy_pinn.backend_utils import torch_context
from manipulapy_pinn.physics import eom_residual, fk_position_residual, precompute_dynamics_operators


def test_precompute_dynamics_operators_shapes(robot, rng):
    q = robot.sample_configurations(4, rng)
    qdot = rng.uniform(-1, 1, size=(4, robot.n_joints))
    M, C, g = precompute_dynamics_operators(robot, q, qdot)
    n = robot.n_joints
    assert M.shape == (4, n, n)
    assert C.shape == (4, n)
    assert g.shape == (4, n)
    assert torch.isfinite(M).all() and torch.isfinite(C).all() and torch.isfinite(g).all()
    # Mass matrices are symmetric.
    for i in range(4):
        assert torch.allclose(M[i], M[i].T, atol=1e-8)


def test_eom_residual_zero_at_true_forward_dynamics(robot, rng):
    """If qddot came from ManipulaPy's own forward_dynamics(q, qdot, tau, ...),
    the EOM residual against the same tau must vanish — the operators and the
    ground-truth acceleration are self-consistent by construction."""
    q = robot.sample_configurations(6, rng)
    qdot = rng.uniform(-1, 1, size=(6, robot.n_joints))
    tau = rng.uniform(-10, 10, size=(6, robot.n_joints))
    gravity = np.array([0.0, 0.0, -9.81])
    qddot_true = np.stack(
        [robot.dynamics.forward_dynamics(q[i], qdot[i], tau[i], gravity, np.zeros(6)) for i in range(6)]
    )

    M, C, g = precompute_dynamics_operators(robot, q, qdot, gravity=gravity)
    residual = eom_residual(M, C, g, torch.tensor(qddot_true, dtype=torch.float64), torch.tensor(tau, dtype=torch.float64))
    assert residual.abs().max().item() < 1e-6


def test_fk_position_residual_zero_for_matching_configuration(robot, rng):
    q = robot.sample_configurations(3, rng)
    target = torch.tensor(robot.forward_kinematics(q)[:, :3, 3], dtype=torch.float64)
    q_pred = torch.tensor(q, dtype=torch.float64)
    with torch_context():
        residual = fk_position_residual(robot.serial, q_pred, target)
    assert residual.abs().max().item() < 1e-8
