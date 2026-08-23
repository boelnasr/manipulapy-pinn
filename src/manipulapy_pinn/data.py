# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ground-truth dataset generation, entirely through ManipulaPy's NumPy backend.

Every PINN in this package trains against data (and, for forward dynamics,
physics operators) generated here — ManipulaPy is the source of truth, not
a synthetic approximation of one.
"""
from __future__ import annotations

from typing import Dict, Sequence

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


#: Default train / test / eval proportions. See :func:`split_dataset` for what
#: each one is allowed to be used for.
DEFAULT_SPLIT = (0.70, 0.20, 0.10)


def split_dataset(
    data: Dict[str, np.ndarray],
    fractions: Sequence[float] = DEFAULT_SPLIT,
    rng: np.random.Generator = None,
):
    """Shuffle and split a dataset dict three ways, same keys in each part.

    The three splits exist because they are allowed to be used for different
    things, and conflating them quietly flatters the reported result:

    - **train** — the only split gradients are computed on.
    - **test** — evaluated repeatedly *during* training to trace the held-out
      curve and pick the best iteration. Because it steers a decision, a score
      on it is no longer fully independent: the run has been fitted to it, if
      only through when to stop.
    - **eval** — touched exactly once, after training ends, and never used to
      choose anything. This is the honest number to report.

    Note the naming follows the caller's request rather than the more common
    train/validation/test convention; ``test`` here plays the role usually
    called *validation*, and ``eval`` the role usually called *test*.

    ``fractions`` must be positive and sum to 1, and every split must receive
    at least one sample — a silently empty split would make its metrics NaN
    rather than fail.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    fractions = tuple(float(f) for f in fractions)
    if len(fractions) != 3:
        raise ValueError(f"expected three fractions (train, test, eval), got {len(fractions)}")
    if any(f <= 0 for f in fractions):
        raise ValueError(f"every fraction must be positive, got {fractions}")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {fractions} summing to {sum(fractions)}")

    n = len(next(iter(data.values())))
    n_train = int(round(n * fractions[0]))
    n_test = int(round(n * fractions[1]))
    n_eval = n - n_train - n_test          # remainder, so the parts always sum to n
    if min(n_train, n_test, n_eval) < 1:
        raise ValueError(
            f"{n} samples cannot be split {fractions} — every split needs at least one "
            f"sample, and this gives {n_train}/{n_test}/{n_eval}. Use more samples."
        )

    idx = rng.permutation(n)
    bounds = (idx[:n_train], idx[n_train:n_train + n_test], idx[n_train + n_test:])
    return tuple({k: v[part] for k, v in data.items()} for part in bounds)


def train_val_split(data: Dict[str, np.ndarray], val_fraction: float, rng: np.random.Generator):
    """Shuffle and split a dataset dict into (train, val) dicts, same keys.

    Kept for callers that only need a two-way split; :func:`split_dataset` is
    what the training code uses.
    """
    n = len(next(iter(data.values())))
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train = {k: v[train_idx] for k, v in data.items()}
    val = {k: v[val_idx] for k, v in data.items()}
    return train, val
