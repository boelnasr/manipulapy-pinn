# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ground-truth dataset generation, entirely through ManipulaPy's NumPy backend.

Every PINN in this package trains against data (and, for forward dynamics,
physics operators) generated here — ManipulaPy is the source of truth, not
a synthetic approximation of one.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .backend_utils import numpy_context
from .robots import RobotModel

GRAVITY = np.array([0.0, 0.0, -9.81])


def generate_forward_dynamics_dataset(
    robot: RobotModel,
    n_samples: int,
    rng: np.random.Generator,
    qdot_scale: float = 1.5,
    tau_scale: float = 20.0,
    gravity: np.ndarray = GRAVITY,
) -> Dict[str, np.ndarray]:
    """Sample (q, q̇, τ) and label with the true q̈ = ManipulaPy.forward_dynamics(...)."""
    q = robot.sample_configurations(n_samples, rng)
    qdot = rng.uniform(-qdot_scale, qdot_scale, size=(n_samples, robot.n_joints))
    tau = rng.uniform(-tau_scale, tau_scale, size=(n_samples, robot.n_joints))
    with numpy_context():
        qddot = np.stack(
            [
                robot.dynamics.forward_dynamics(q[i], qdot[i], tau[i], gravity, np.zeros(6))
                for i in range(n_samples)
            ]
        )
    return {"q": q, "qdot": qdot, "tau": tau, "qddot": qddot}


def generate_ik_dataset(robot: RobotModel, n_samples: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Sample joint configs and label with their true end-effector pose.

    Every target pose is reachable by construction (it's the FK image of a
    sampled configuration) — the PINN is never asked to hit an unreachable
    target. ``q_reference`` is kept only as an evaluation aid (one valid
    solution among possibly many, at a redundant robot); it is never part of
    the training loss.
    """
    q = robot.sample_configurations(n_samples, rng)
    poses = robot.forward_kinematics(q)  # (N, 4, 4)
    positions = poses[:, :3, 3]
    return {"q_reference": q, "pose": poses, "position": positions}


def train_val_split(data: Dict[str, np.ndarray], val_fraction: float, rng: np.random.Generator):
    """Shuffle and split a dataset dict into (train, val) dicts, same keys."""
    n = len(next(iter(data.values())))
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train = {k: v[train_idx] for k, v in data.items()}
    val = {k: v[val_idx] for k, v in data.items()}
    return train, val
