# SPDX-License-Identifier: AGPL-3.0-or-later
"""manipulapy-pinn: physics-informed neural networks built on ManipulaPy.

Three tasks, each in its own module:

- :mod:`manipulapy_pinn.forward_dynamics` — q̈ = f_θ(q, q̇, τ)
- :mod:`manipulapy_pinn.inverse_kinematics` — q = h_θ(target pose)
- :mod:`manipulapy_pinn.trajectory` — q_θ(t), solved as a boundary-value ODE

All three train against ManipulaPy as ground truth: sampled data and/or a
differentiable physics residual computed through ManipulaPy's own
kinematics and dynamics (see each module's docstring for exactly how).
"""

from .robots import RobotModel, load_robot, available_robots

__all__ = ["RobotModel", "load_robot", "available_robots"]

__version__ = "0.1.0"
