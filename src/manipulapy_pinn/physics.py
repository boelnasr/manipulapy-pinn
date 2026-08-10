# SPDX-License-Identifier: AGPL-3.0-or-later
"""Differentiable physics residuals built on ManipulaPy's dynamics/kinematics.

Two different strategies are used here, and the difference matters:

- **Forward dynamics residual** (``eom_residual``): M(q), C(q, q̇) and g(q)
  depend only on the *inputs* (q, q̇), which are fixed data at training time
  — they never change as the network trains. So they're computed **once**,
  offline, through ManipulaPy's fast NumPy dynamics (``precompute_dynamics_operators``),
  and reused as constants every step. Only q̈_pred (the network's output)
  needs a gradient.

- **Kinematic residuals** (``fk_position_residual``): the network's output
  (q_pred) feeds directly into ``forward_kinematics``, and that dependency
  changes every step — so it has to run live, under the torch backend, with
  autograd tracking through the FK call itself. ManipulaPy's FK is cheap
  enough (~3ms/call on CPU) for this to be practical inside a training loop.

Why not run the manipulator dynamics (mass matrix, Coriolis) live under
torch the same way? Measured on this package's reference hardware,
``velocity_quadratic_forces`` (Coriolis/centrifugal) alone costs on the
order of 250ms/call under the torch backend — ManipulaPy's dynamics module
recurses per-link with many small tensor ops, and PyTorch's per-op eager
dispatch overhead dominates for a chain this size. NumPy runs the same
algorithm in a few milliseconds. That gap is why ``forward_dynamics.py``'s
physics residual precomputes M/C/g instead of differentiating through
``dynamics.mass_matrix`` directly, and why ``trajectory.py`` optimizes
through a *trained* :class:`~manipulapy_pinn.forward_dynamics.ForwardDynamicsPINN`
surrogate rather than ManipulaPy's own dynamics calls.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from .backend_utils import numpy_context


def precompute_dynamics_operators(
    robot, q: np.ndarray, qdot: np.ndarray, gravity: np.ndarray = np.array([0.0, 0.0, -9.81])
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """M(q), C(q, q̇), g(q) for a batch, via ManipulaPy's NumPy-backend dynamics.

    Returns three float64 tensors: M with shape (N, n, n), C and g with
    shape (N, n). These are plain constants — no graph, nothing to
    backpropagate through — meant to be reused across every training step.
    Explicitly scoped to the NumPy backend so this stays fast regardless of
    what backend another part of a calling script left active.
    """
    n_samples = q.shape[0]
    Ms, Cs, Gs = [], [], []
    with numpy_context():
        for i in range(n_samples):
            Ms.append(robot.dynamics.mass_matrix(q[i]))
            Cs.append(robot.dynamics.velocity_quadratic_forces(q[i], qdot[i]))
            Gs.append(robot.dynamics.gravity_forces(q[i], gravity))
    return (
        torch.tensor(np.stack(Ms), dtype=torch.float64),
        torch.tensor(np.stack(Cs), dtype=torch.float64),
        torch.tensor(np.stack(Gs), dtype=torch.float64),
    )


def eom_residual(
    M: torch.Tensor, C: torch.Tensor, g: torch.Tensor, qddot_pred: torch.Tensor, tau: torch.Tensor
) -> torch.Tensor:
    """Batched equation-of-motion residual: τ − (M(q) q̈ + C(q, q̇) + g(q)).

    Zero everywhere the network's predicted acceleration is dynamically
    consistent with the applied torque under the true (precomputed) M/C/g
    at that state.
    """
    predicted_tau = torch.einsum("nij,nj->ni", M, qddot_pred) + C + g
    return tau - predicted_tau


def fk_position_residual(serial, q_pred: torch.Tensor, target_position: torch.Tensor) -> torch.Tensor:
    """Batched end-effector position residual, differentiated live through FK.

    ``q_pred`` must be a torch tensor produced under ``use_backend("torch")``
    with ``requires_grad`` tracing back to network parameters — this is what
    makes the inverse-kinematics PINN "physics-informed" rather than a plain
    regression: the loss is defined by the forward-kinematics constraint
    itself, not by a table of (pose, q) pairs.
    """
    positions = torch.stack([serial.forward_kinematics(q)[:3, 3] for q in q_pred])
    return target_position - positions
