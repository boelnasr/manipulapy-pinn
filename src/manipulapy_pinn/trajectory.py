# SPDX-License-Identifier: AGPL-3.0-or-later
"""Trajectory optimization as a PINN-for-ODEs problem.

The classic Raissi-style framing: represent the *whole trajectory* q(t) as
a neural network of time, get velocity and acceleration for free by
differentiating the network with respect to its input (``torch.autograd``,
twice), and train the network so that those quantities satisfy the
manipulator's equation of motion at a set of collocation points — no time
discretization, no finite differences between waypoints.

**Why this uses a trained surrogate instead of ManipulaPy's own dynamics.**
The natural physics residual is the equation of motion evaluated at each
collocation point: does the torque this trajectory implies match what the
*true* dynamics says q̈(t) should produce? That requires evaluating the
dynamics operator fresh at every collocation point, every training step,
with autograd tracking back into the trajectory network's parameters.
ManipulaPy's own Coriolis computation (``velocity_quadratic_forces``) costs
on the order of 250ms/call under the torch backend (measured on this
package's reference hardware) — for even a modest 16 collocation points and
a few hundred iterations, that's tens of minutes. A
:class:`~manipulapy_pinn.forward_dynamics.ForwardDynamicsPINN`, once
trained, replaces that call with a few matrix multiplies through a small
MLP: milliseconds instead of hundreds of them, differentiable by
construction. The trade is honest: the trajectory optimizer is only as
accurate as the surrogate's fit to ManipulaPy's true dynamics (see the
surrogate's own validation RMSE), not exact — the standard trade any
learned-dynamics-model-based controller makes.

**Boundary conditions.** Position boundary conditions (q(0) = q_start,
q(T) = q_goal) are satisfied *exactly*, by construction, via the ansatz

    q_θ(t) = q_start + t·(q_goal − q_start) + t·(1−t)·NN_θ(t)

rather than as a soft loss term — t·(1−t) vanishes at both endpoints
regardless of what the network outputs, so there is nothing for the
optimizer to get slightly wrong there. Velocity boundary conditions
(start/end at rest) are not free the same way and are enforced as a soft
penalty instead.

**Known limitation: no joint velocity or torque limits are enforced.** The
loss softly minimizes control effort (mean τ²) and pins the endpoints at
rest, but nothing stops the interior of the trajectory from swinging through
velocities or torques a real robot couldn't achieve — on the reference
robot/problem this package ships with, unconstrained solves have produced
peak joint speeds around 10 rad/s, well past a real Panda's ~2.6 rad/s
limit. Adding hard limits would mean the same kind of penalty term
``differentiable_reach_showcase.py`` in the main ManipulaPy repo uses for
obstacle clearance: ``clip(|q̇| − q̇_max, 0, None)**2`` evaluated at the
collocation points and added to the loss. Left as a natural next step rather
than included here, so this stays a clean minimal example of the
collocation pattern.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .forward_dynamics import ForwardDynamicsPINN
from .models import MLP


class TrajectoryAnsatz(nn.Module):
    """q_θ(t) with position boundary conditions satisfied exactly, for all θ."""

    def __init__(self, q_start: np.ndarray, q_goal: np.ndarray, hidden=(64, 64)):
        super().__init__()
        n = q_start.shape[0]
        self.register_buffer("q_start", torch.tensor(q_start, dtype=torch.float64))
        self.register_buffer("q_goal", torch.tensor(q_goal, dtype=torch.float64))
        self.correction = MLP(1, n, hidden=hidden)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (N, 1) in [0, 1]
        linear = self.q_start[None, :] + t * (self.q_goal - self.q_start)[None, :]
        return linear + t * (1 - t) * self.correction(t)


def time_derivatives(
    traj_fn, t: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """q(t), q̇(t), q̈(t) via double backprop through ``traj_fn`` w.r.t. ``t``.

    ``t`` must have ``requires_grad=True``. Each output column is
    differentiated independently via a ``.sum()`` reduction, which is valid
    because ``traj_fn``'s i-th output row depends only on ``t[i]`` (true for
    any per-sample MLP) — there is no cross-sample coupling for the
    reduction to accidentally mix in.
    """
    q = traj_fn(t)
    n = q.shape[-1]

    qdot_cols = []
    for j in range(n):
        (grad_j,) = torch.autograd.grad(q[:, j].sum(), t, create_graph=True, retain_graph=True)
        qdot_cols.append(grad_j.squeeze(-1))
    qdot = torch.stack(qdot_cols, dim=-1)

    qddot_cols = []
    for j in range(n):
        (grad_j,) = torch.autograd.grad(qdot[:, j].sum(), t, create_graph=True, retain_graph=True)
        qddot_cols.append(grad_j.squeeze(-1))
    qddot = torch.stack(qddot_cols, dim=-1)

    return q, qdot, qddot


@dataclass
class TrajectoryResult:
    trajectory_model: TrajectoryAnsatz
    torque_model: nn.Module
    loss_history: List[dict] = field(default_factory=list)
    dynamics_model: Optional[ForwardDynamicsPINN] = None

    def sample(self, n: int = 100) -> dict:
        """Evaluate q, q̇, q̈, τ on an ``n``-point time grid, detached to NumPy."""
        t = torch.linspace(0.0, 1.0, n, dtype=torch.float64).view(-1, 1).requires_grad_(True)
        q, qdot, qddot = time_derivatives(self.trajectory_model, t)
        tau = self.torque_model(t.detach())
        return {
            "t": t.detach().numpy().flatten(),
            "q": q.detach().numpy(),
            "qdot": qdot.detach().numpy(),
            "qddot": qddot.detach().numpy(),
            "tau": tau.detach().numpy(),
        }


def solve(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    dynamics_model: ForwardDynamicsPINN,
    n_collocation: int = 24,
    iterations: int = 800,
    lr: float = 2e-3,
    physics_weight: float = 1.0,
    effort_weight: float = 1e-3,
    boundary_velocity_weight: float = 10.0,
    hidden=(64, 64),
    torque_hidden=(64, 64),
    seed: int = 0,
    log_every: int = 100,
    verbose: bool = True,
) -> TrajectoryResult:
    """Solve a start→goal trajectory as a PINN, physics-informed by ``dynamics_model``.

    ``dynamics_model`` should be a :class:`ForwardDynamicsPINN` already
    trained for the same robot (see ``forward_dynamics.train``) — it stands
    in for ManipulaPy's own dynamics in the physics residual (see the module
    docstring for why). Its parameters are frozen here; only the trajectory
    and torque networks are optimized.
    """
    torch.manual_seed(seed)
    n = q_start.shape[0]
    dynamics_model.eval()
    for p in dynamics_model.parameters():
        p.requires_grad_(False)

    traj_model = TrajectoryAnsatz(q_start, q_goal, hidden=hidden)
    torque_model = MLP(1, n, hidden=torque_hidden)
    optimizer = torch.optim.Adam(
        list(traj_model.parameters()) + list(torque_model.parameters()), lr=lr
    )
    history = []

    if verbose:
        print(f"   Solving a {n}-DOF trajectory PINN "
              f"({n_collocation} collocation points, {iterations} steps)...")
    start_time = time.time()
    for it in range(iterations):
        t = torch.linspace(0.0, 1.0, n_collocation, dtype=torch.float64).view(-1, 1)
        t.requires_grad_(True)

        optimizer.zero_grad()
        q, qdot, qddot = time_derivatives(traj_model, t)
        tau = torque_model(t)

        qddot_dynamics = dynamics_model(q, qdot, tau)
        physics_residual = qddot_dynamics - qddot
        physics_loss = physics_residual.pow(2).mean()

        effort_loss = tau.pow(2).mean()
        boundary_loss = qdot[0].pow(2).sum() + qdot[-1].pow(2).sum()

        loss = (
            physics_weight * physics_loss
            + effort_weight * effort_loss
            + boundary_velocity_weight * boundary_loss
        )
        loss.backward()
        optimizer.step()

        history.append(
            {
                "iter": it,
                "loss": loss.item(),
                "physics": physics_loss.item(),
                "effort": effort_loss.item(),
                "boundary": boundary_loss.item(),
            }
        )
        if verbose and (it % log_every == 0 or it == iterations - 1):
            print(f"     step {it:4d}   loss={loss.item():.5f}  physics={physics_loss.item():.5f}  "
                  f"effort={effort_loss.item():.5f}  boundary_v={boundary_loss.item():.5f}")
    elapsed = time.time() - start_time
    if verbose:
        print(f"   ✅ Solved in {elapsed:.1f}s")

    return TrajectoryResult(
        trajectory_model=traj_model, torque_model=torque_model, loss_history=history,
        dynamics_model=dynamics_model,
    )
