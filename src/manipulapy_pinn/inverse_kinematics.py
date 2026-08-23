# SPDX-License-Identifier: AGPL-3.0-or-later
"""Physics-informed inverse kinematics: q = h_θ(target position).

Trained on the forward-kinematics *constraint* itself — FK(h_θ(x)) ≈ x —
differentiated live through ManipulaPy's ``forward_kinematics`` under the
torch backend (see ``physics.fk_position_residual``), not on a stored table
of (pose, q) pairs. That distinction matters at a redundant robot: there is
no single correct q for a given pose, so supervising against one arbitrarily
chosen reference q would actively fight other equally valid solutions. The
FK-residual loss only cares whether the predicted configuration reaches the
target — any solution the network settles into is scored the same way
ManipulaPy's own iterative solvers are.

Target poses are generated online every step (sample a random reachable
configuration, take its true FK) rather than from a fixed dataset — poses
are free to generate and this keeps the network from ever seeing the same
target twice, which is a meaningfully different training regime from
data-table supervision.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch

from .backend_utils import torch_context
from .models import build_mlp
from .physics import fk_position_residual
from .robots import RobotModel


class InverseKinematicsPINN(torch.nn.Module):
    """q = h_θ(target end-effector position)."""

    def __init__(self, n_joints: int, hidden=(128,) * 5):
        super().__init__()
        self.n_joints = n_joints
        self.net = build_mlp(3, n_joints, hidden=hidden)

    def forward(self, target_position: torch.Tensor) -> torch.Tensor:
        return self.net(target_position)


@dataclass
class InverseKinematicsResult:
    model: InverseKinematicsPINN
    loss_history: List[dict] = field(default_factory=list)
    #: Mean Euclidean end-effector position error on held-out targets, in
    #: **metres** (SI, matching ManipulaPy). Human-readable output is printed
    #: in mm; convert with ``val_position_rmse * 1000``.
    val_position_rmse: float = float("nan")


def train(
    robot: RobotModel,
    iterations: int = 600,
    batch_size: int = 32,
    lr: float = 1e-3,
    n_val: int = 64,
    # Five layers, which build_mlp turns into residual blocks. Confirmed at this
    # task's own default budget of 600 iterations, two seeds per depth, held-out
    # reach error:
    #
    #   depth 3   84.7 / 100.6 mm   -> 92.7
    #   depth 5   49.0 /  64.9 mm   -> 57.0   <- default
    #   (at 1500 iterations: 57.4 vs 37.8 mm, same ordering)
    #
    # The worst depth-5 run beats the best depth-3 run, so the separation
    # exceeds the run-to-run spread.
    #
    # Note this is the opposite conclusion from forward dynamics, where depth 5
    # is a wash at the default budget. The reason is that this task has no fixed
    # dataset: every step draws fresh targets, so there is nothing to memorize
    # and extra capacity goes into fitting the map rather than the sample.
    # Forward dynamics trains on a fixed set and overfits it, which is why the
    # deeper network there needs a bigger dataset before it pays.
    hidden=(128,) * 5,
    seed: int = 0,
    log_every: int = 100,
    verbose: bool = True,
) -> InverseKinematicsResult:
    # Seed torch as well as NumPy — see the note in forward_dynamics.train; here it
    # matters even more, since targets are resampled every step and initialization
    # is the only other source of run-to-run variation.
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = InverseKinematicsPINN(robot.n_joints, hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    if verbose:
        print(f"   Training inverse-kinematics PINN on {robot.name} "
              f"({batch_size} fresh reachable targets/step, {iterations} steps)...")
    start = time.time()
    for it in range(iterations):
        # Target generation runs on the fast NumPy backend (the default) —
        # only the network's own FK call, on q_pred, needs torch autograd.
        q_sample = robot.sample_configurations(batch_size, rng)
        target = torch.tensor(robot.forward_kinematics(q_sample)[:, :3, 3], dtype=torch.float64)

        optimizer.zero_grad()
        q_pred = model(target)
        with torch_context():
            residual = fk_position_residual(robot.serial, q_pred, target)
        loss = residual.pow(2).mean()
        loss.backward()
        optimizer.step()

        history.append({"iter": it, "loss": loss.item()})
        if verbose and (it % log_every == 0 or it == iterations - 1):
            print(f"     step {it:4d}   position MSE = {loss.item():.6f} m²")
    elapsed = time.time() - start

    with torch.no_grad():
        q_val = robot.sample_configurations(n_val, rng)
        target_val = torch.tensor(robot.forward_kinematics(q_val)[:, :3, 3], dtype=torch.float64)
        q_pred_val = model(target_val)
        with torch_context():
            val_residual = fk_position_residual(robot.serial, q_pred_val, target_val)
        val_rmse = val_residual.pow(2).sum(dim=-1).sqrt().mean().item()

    if verbose:
        print(f"   ✅ Trained in {elapsed:.1f}s — held-out reach error = {val_rmse * 1000:.1f} mm "
              f"(mean over {n_val} unseen targets)")

    return InverseKinematicsResult(model=model, loss_history=history, val_position_rmse=val_rmse)
