# SPDX-License-Identifier: AGPL-3.0-or-later
"""Robot loading built on ManipulaPy's bundled URDF catalog.

Ground truth for every PINN in this package comes from ManipulaPy's own
kinematics and dynamics — this module is the one place that decides which
robot and which joints are "the arm" for that purpose.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from ManipulaPy.ManipulaPy_data import get_robot_urdf, list_robots
from ManipulaPy.urdf_processor import URDFToSerialManipulator

# Number of actuated *arm* joints to use per bundled robot, excluding
# grippers and any other non-arm actuated joints in the URDF. Add an entry
# here before calling load_robot() with a new robot name.
_ARM_DOF = {
    "panda": 7,
    "ur3": 6,
    "ur3e": 6,
    "ur5": 6,
    "ur5e": 6,
    "ur10": 6,
    "ur10e": 6,
    "ur16e": 6,
    "xarm6": 6,
    "iiwa7": 7,
    "iiwa14": 7,
}


@dataclass
class RobotModel:
    """A loaded robot plus the arm-joint bookkeeping every PINN task needs."""

    name: str
    serial: "ManipulaPy.kinematics.SerialManipulator"  # noqa: F821
    dynamics: "ManipulaPy.dynamics.ManipulatorDynamics"  # noqa: F821
    joint_limits: List[Tuple[float, float]]

    @property
    def n_joints(self) -> int:
        return len(self.joint_limits)

    def sample_configurations(
        self, n: int, rng: np.random.Generator, margin: float = 0.05
    ) -> np.ndarray:
        """Uniformly sample ``n`` joint configurations, shrunk off the hard limits.

        ``margin`` (a fraction of each joint's range) keeps samples away from
        the exact limit boundary, where the dynamics can be poorly
        conditioned and where a physical robot would rarely be commanded.
        """
        lo = np.array([l for l, _ in self.joint_limits])
        hi = np.array([h for _, h in self.joint_limits])
        span = hi - lo
        lo, hi = lo + margin * span, hi - margin * span
        return rng.uniform(lo, hi, size=(n, self.n_joints))

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """4x4 end-effector pose. Accepts a single config or a batch (N, n_joints)."""
        if q.ndim == 1:
            return self.serial.forward_kinematics(q)
        return np.stack([self.serial.forward_kinematics(qi) for qi in q])


def available_robots() -> List[str]:
    """Robot names this package knows how to restrict to an arm-only DOF count."""
    return sorted(set(list_robots()) & set(_ARM_DOF))


def load_robot(name: str = "panda") -> RobotModel:
    """Load a bundled robot by name and its accurate dynamics.

    Uses ``URDFToSerialManipulator`` rather than ``URDF.to_manipulator_dynamics()``
    directly — the latter skips per-link home matrices and silently falls back
    to a legacy mass-matrix approximation that ManipulaPy itself warns is
    "incorrect for non-trivial robots".
    """
    if name not in _ARM_DOF:
        raise ValueError(
            f"Unknown robot {name!r}. Known robots: {sorted(_ARM_DOF)} "
            f"(add an entry to manipulapy_pinn.robots._ARM_DOF to support another)."
        )
    proc = URDFToSerialManipulator(get_robot_urdf(name))
    n = _ARM_DOF[name]
    joint_limits = list(proc.serial_manipulator.joint_limits)[:n]
    return RobotModel(
        name=name,
        serial=proc.serial_manipulator,
        dynamics=proc.dynamics,
        joint_limits=joint_limits,
    )
