# SPDX-License-Identifier: AGPL-3.0-or-later
"""Physics-informed forward dynamics: q̈ = f_θ(q, q̇, τ).

Trained against two losses:

- **Data loss**: MSE against ManipulaPy's true forward dynamics on sampled
  (q, q̇, τ) triples.
- **Physics loss**: the equation-of-motion residual
  τ − (M(q) q̈_pred + C(q, q̇) + g(q)) — see ``physics.eom_residual``. M, C, g
  are precomputed once via ManipulaPy's NumPy backend (see the module
  docstring in ``physics.py`` for why), so this term costs nothing extra
  ManipulaPy-side at training time; it's the network's own consistency with
  the *true* dynamics operator at each sampled state, not just its distance
  to a stored q̈ label.

The trained network is a fast, differentiable dynamics surrogate: a single
forward pass replaces the (measured) ~300ms ManipulaPy Coriolis computation
under torch with a few matrix multiplies. ``trajectory.py`` uses exactly
that property to make trajectory optimization tractable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch

from .data import generate_forward_dynamics_dataset, train_val_split
from .models import build_mlp
from .physics import eom_residual, precompute_dynamics_operators
from .robots import RobotModel


class ForwardDynamicsPINN(torch.nn.Module):
    """q̈ = f_θ(q, q̇, τ), an MLP over the concatenated state and input torque.

    Normalizes inputs and outputs internally using statistics fixed at
    construction time (typically the training set's), so a saved model is
    self-contained and callers never need to remember to standardize their
    own inputs. Without this, q (order 1 rad), q̇ (order 1 rad/s) and τ
    (order 10 N·m) sit at wildly different scales feeding the same Tanh
    layers — the larger torque inputs dominate and the network underfits
    the smaller-magnitude joint-position dependence.
    """

    def __init__(
        self,
        n_joints: int,
        hidden=(128, 128, 128),
        input_mean: torch.Tensor = None,
        input_std: torch.Tensor = None,
        output_mean: torch.Tensor = None,
        output_std: torch.Tensor = None,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.net = build_mlp(3 * n_joints, n_joints, hidden=hidden)
        zeros3, ones3 = torch.zeros(3 * n_joints), torch.ones(3 * n_joints)
        zeros1, ones1 = torch.zeros(n_joints), torch.ones(n_joints)
        self.register_buffer("input_mean", input_mean if input_mean is not None else zeros3)
        self.register_buffer("input_std", input_std if input_std is not None else ones3)
        self.register_buffer("output_mean", output_mean if output_mean is not None else zeros1)
        self.register_buffer("output_std", output_std if output_std is not None else ones1)

    def forward(self, q: torch.Tensor, qdot: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        x = torch.cat([q, qdot, tau], dim=-1)
        x = (x - self.input_mean) / self.input_std
        y = self.net(x)
        return y * self.output_std + self.output_mean


@dataclass
class ForwardDynamicsResult:
    model: ForwardDynamicsPINN
    loss_history: List[dict] = field(default_factory=list)
    #: Held-out error sampled periodically during training: one dict per check
    #: with ``iter``, ``val_mse`` and ``val_rmse``. The training loss alone
    #: cannot tell you when to stop — it keeps falling while the model
    #: memorizes — so the useful signal is where this curve turns back up.
    val_history: List[dict] = field(default_factory=list)
    val_data_rmse: float = float("nan")

    def best_iteration(self) -> dict:
        """The recorded checkpoint with the lowest held-out error.

        If this is far from the last entry, the run trained past its own
        optimum and the extra iterations did harm rather than nothing.
        """
        return min(self.val_history, key=lambda h: h["val_rmse"]) if self.val_history else {}


def train(
    robot: RobotModel,
    n_samples: int = 2000,
    iterations: int = 1500,
    lr: float = 1e-3,
    data_weight: float = 1.0,
    physics_weight: float = 1.0,
    val_fraction: float = 0.15,
    # Three layers. Depth 5 measured better at 3000 samples / 3000 iterations
    # (2.557 vs 2.711), but at the DEFAULT budget below it is a wash while
    # costing roughly twice the training time -- 2000 samples, 1500 iterations,
    # two seeds each:
    #
    #             1500 iters      4000 iters
    #   depth 3   3.147 / 3.206   3.252 / 3.277
    #   depth 5   3.145 / 3.163   3.281 / 3.316
    #
    # Note the columns as well as the rows: at 2000 samples, MORE iterations
    # make both depths worse. Depth and budget have to be raised together, so
    # the shipped default stays shallow and `--depth 5` is there for when the
    # sample count justifies it. See the pairing table in the README.
    hidden=(128, 128, 128),
    seed: int = 0,
    log_every: int = 200,
    val_every: int = 100,
    verbose: bool = True,
) -> ForwardDynamicsResult:
    # Seed torch as well as NumPy: NumPy alone controls the sampled states, but
    # network initialization comes from torch's global generator, so without this
    # two runs with the same `seed` differ by more than most effects worth measuring.
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    data = generate_forward_dynamics_dataset(robot, n_samples, rng)
    train_data, val_data = train_val_split(data, val_fraction, rng)

    M, C, g = precompute_dynamics_operators(robot, train_data["q"], train_data["qdot"])
    q_t = torch.tensor(train_data["q"], dtype=torch.float64)
    qdot_t = torch.tensor(train_data["qdot"], dtype=torch.float64)
    tau_t = torch.tensor(train_data["tau"], dtype=torch.float64)
    qddot_t = torch.tensor(train_data["qddot"], dtype=torch.float64)

    x_all = torch.cat([q_t, qdot_t, tau_t], dim=-1)
    input_mean, input_std = x_all.mean(dim=0), x_all.std(dim=0).clamp_min(1e-6)
    output_mean, output_std = qddot_t.mean(dim=0), qddot_t.std(dim=0).clamp_min(1e-6)

    model = ForwardDynamicsPINN(
        robot.n_joints, hidden=hidden,
        input_mean=input_mean, input_std=input_std,
        output_mean=output_mean, output_std=output_std,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history, val_history = [], []

    q_v = torch.tensor(val_data["q"], dtype=torch.float64)
    qdot_v = torch.tensor(val_data["qdot"], dtype=torch.float64)
    tau_v = torch.tensor(val_data["tau"], dtype=torch.float64)
    qddot_v = torch.tensor(val_data["qddot"], dtype=torch.float64)

    if verbose:
        print(f"   Training forward-dynamics PINN on {robot.name} "
              f"({len(train_data['q'])} train / {len(val_data['q'])} val samples)...")
    start = time.time()
    for it in range(iterations):
        optimizer.zero_grad()
        qddot_pred = model(q_t, qdot_t, tau_t)
        data_loss = (qddot_pred - qddot_t).pow(2).mean()
        residual = eom_residual(M, C, g, qddot_pred, tau_t)
        physics_loss = residual.pow(2).mean()
        loss = data_weight * data_loss + physics_weight * physics_loss
        loss.backward()
        optimizer.step()
        history.append({"iter": it, "loss": loss.item(), "data": data_loss.item(), "physics": physics_loss.item()})

        # Sampled on the held-out split, in the same units as the data loss, so
        # the two curves can be read on one axis: they track together while the
        # model generalizes and separate once it starts memorizing.
        if val_every and (it % val_every == 0 or it == iterations - 1):
            with torch.no_grad():
                val_mse = (model(q_v, qdot_v, tau_v) - qddot_v).pow(2).mean().item()
            val_history.append({"iter": it, "val_mse": val_mse, "val_rmse": val_mse ** 0.5})

        if verbose and (it % log_every == 0 or it == iterations - 1):
            print(f"     step {it:5d}   loss={loss.item():.5f}  data={data_loss.item():.5f}  "
                  f"physics={physics_loss.item():.5f}")
    elapsed = time.time() - start

    with torch.no_grad():
        val_rmse = (model(q_v, qdot_v, tau_v) - qddot_v).pow(2).mean().sqrt().item()

    if verbose:
        print(f"   ✅ Trained in {elapsed:.1f}s — held-out q̈ RMSE = {val_rmse:.4f} rad/s²")
        if val_history:
            best = min(val_history, key=lambda h: h["val_rmse"])
            if best["iter"] < 0.8 * (iterations - 1):
                print(f"   ⚠️  Best held-out error was {best['val_rmse']:.4f} at iteration "
                      f"{best['iter']} — the last {iterations - 1 - best['iter']} iterations "
                      f"made it worse. Train on more samples rather than for longer.")

    return ForwardDynamicsResult(model=model, loss_history=history,
                                 val_history=val_history, val_data_rmse=val_rmse)
