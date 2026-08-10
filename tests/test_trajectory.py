# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np
import torch

from manipulapy_pinn.forward_dynamics import train as train_dynamics
from manipulapy_pinn.trajectory import TrajectoryAnsatz, solve, time_derivatives


def test_time_derivatives_match_finite_differences():
    torch.manual_seed(0)
    model = TrajectoryAnsatz(np.array([0.0, 0.0]), np.array([1.0, -1.0]), hidden=(8, 8))
    t = torch.linspace(0, 1, 8, dtype=torch.float64).view(-1, 1).requires_grad_(True)
    q, qdot, qddot = time_derivatives(model, t)

    eps = 1e-6
    t_flat = t.detach().flatten()
    qdot_fd = torch.zeros_like(qdot)
    with torch.no_grad():
        for i in range(len(t_flat)):
            tp = (t_flat[i] + eps).view(1, 1)
            tm = (t_flat[i] - eps).view(1, 1)
            qdot_fd[i] = (model(tp)[0] - model(tm)[0]) / (2 * eps)
    assert torch.allclose(qdot, qdot_fd, atol=1e-6)


def test_ansatz_satisfies_boundary_positions_exactly():
    q_start = np.array([0.1, -0.2, 0.3])
    q_goal = np.array([0.5, 0.6, -0.1])
    model = TrajectoryAnsatz(q_start, q_goal, hidden=(8, 8))
    t = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    q = model(t)
    np.testing.assert_allclose(q[0].detach().numpy(), q_start, atol=1e-12)
    np.testing.assert_allclose(q[1].detach().numpy(), q_goal, atol=1e-12)


def test_solve_end_to_end_stays_finite_and_hits_boundaries(robot):
    dyn_result = train_dynamics(
        robot, n_samples=30, iterations=20, hidden=(16, 16), log_every=1000, verbose=False,
    )
    rng = np.random.default_rng(1)
    q_start = robot.sample_configurations(1, rng)[0]
    q_goal = robot.sample_configurations(1, rng)[0]

    result = solve(
        q_start, q_goal, dyn_result.model,
        n_collocation=6, iterations=15, hidden=(8, 8), torque_hidden=(8, 8),
        log_every=1000, verbose=False,
    )
    losses = [h["loss"] for h in result.loss_history]
    assert len(losses) == 15
    assert all(np.isfinite(l) for l in losses)

    sample = result.sample(10)
    np.testing.assert_allclose(sample["q"][0], q_start, atol=1e-10)
    np.testing.assert_allclose(sample["q"][-1], q_goal, atol=1e-10)
    assert np.all(np.isfinite(sample["qdot"]))
    assert np.all(np.isfinite(sample["qddot"]))
    assert np.all(np.isfinite(sample["tau"]))
