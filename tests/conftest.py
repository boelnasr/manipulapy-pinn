# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np
import pytest

from manipulapy_pinn import load_robot


@pytest.fixture(scope="session")
def robot():
    """A well-conditioned robot, loaded once for the whole test session.

    Panda specifically: some bundled URDFs (xarm6 in particular) carry a
    near-zero mass-matrix entry on their last joint — apparently a
    placeholder/degenerate inertia tag, constant across every configuration
    tested — which blows sampled accelerations up to ~1e5 rad/s² under
    ordinary torques. That's a property of that URDF's inertial data, not a
    bug in this package or in ManipulaPy's dynamics, but it makes such a
    robot a bad default for tests: huge-but-finite values pass naive
    shape/finiteness checks while making convergence checks meaningless.
    Panda's mass matrix is well-conditioned across the sampled range.
    """
    return load_robot("panda")


@pytest.fixture()
def rng():
    return np.random.default_rng(0)
