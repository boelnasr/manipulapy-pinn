# SPDX-License-Identifier: AGPL-3.0-or-later
"""Differentiable physics residuals built on ManipulaPy's dynamics/kinematics.

Two different strategies are used here, and the difference matters:

- **Forward dynamics residual** (``eom_residual``): M(q), C(q, q̇) and g(q)
  depend only on the *inputs* (q, q̇), which are fixed data at training time
  — they never change as the network trains. So they're computed **once**,
  offline, through ManipulaPy's fast NumPy dynamics (``precompute_dynamics_operators``),
  and reused as constants every step. Only q̈_pred (the network's output)
  needs a gradient.

- **Kinematic residuals** (``fk_pose_residual``): the network's output
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


def fk_pose_residual(serial, q_pred: torch.Tensor, target_position: torch.Tensor,
                     target_rotation: torch.Tensor = None):
    """Batched end-effector pose residual, differentiated live through FK.

    Returns ``(position_residual, orientation_residual)``:

    - **position** — ``(N, 3)`` in metres, so its squared norm is m².
    - **orientation** — ``(N,)``, half the squared Frobenius distance between
      the achieved and target rotation matrices. That quantity is
      ``2(1 − cos θ)`` for a relative rotation angle θ, so it behaves like θ²
      for small errors and puts the two terms in comparable units: a weight of
      1 on each makes a 1 mm position error cost about the same as a 1 mrad
      orientation error.

    The Frobenius (chordal) form is used rather than the geodesic angle
    ``arccos((tr R − 1)/2)`` on purpose. The geodesic distance is the more
    natural metric, but its gradient is unbounded at θ = 0 — exactly where a
    converging optimizer spends its time — so training on it destabilizes as
    the model gets good. The chordal form is smooth everywhere and monotone in
    θ over [0, π], so it ranks solutions identically.

    ``q_pred`` must be a torch tensor produced under ``use_backend("torch")``
    with ``requires_grad`` tracing back to network parameters — this is what
    makes the inverse-kinematics PINN "physics-informed" rather than a plain
    regression: the loss is defined by the forward-kinematics constraint
    itself, not by a table of (pose, q) pairs.

    ``target_rotation`` may be omitted, in which case only the position
    residual is meaningful and the orientation residual is returned as zeros.
    """
    poses = torch.stack([serial.forward_kinematics(q) for q in q_pred])
    position_residual = target_position - poses[..., :3, 3]
    if target_rotation is None:
        return position_residual, torch.zeros(len(q_pred), dtype=position_residual.dtype)
    rot_diff = poses[..., :3, :3] - target_rotation
    orientation_residual = 0.5 * rot_diff.pow(2).sum(dim=(-2, -1))
    return position_residual, orientation_residual


def fk_position_residual(serial, q_pred: torch.Tensor, target_position: torch.Tensor) -> torch.Tensor:
    """Position-only pose residual — see :func:`fk_pose_residual`."""
    return fk_pose_residual(serial, q_pred, target_position)[0]


def pose_features(poses: torch.Tensor) -> torch.Tensor:
    """Flatten 4x4 poses into the 12-vector the IK network takes as input.

    Layout is ``[position (3), rotation matrix row-major (9)]``. The rotation
    is passed as the full matrix rather than Euler angles or a quaternion
    because both of those are discontinuous as functions of the pose — a
    network has to learn around a wraparound that isn't in the geometry.
    """
    return torch.cat([poses[..., :3, 3], poses[..., :3, :3].reshape(*poses.shape[:-2], 9)], dim=-1)
