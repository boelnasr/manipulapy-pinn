# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np
import pytest

from manipulapy_pinn.robots import available_robots, load_robot


def test_available_robots_nonempty():
    names = available_robots()
    assert "panda" in names
    assert "xarm6" in names


def test_unknown_robot_raises():
    with pytest.raises(ValueError):
        load_robot("not_a_real_robot")


def test_load_robot_joint_limits_match_n_joints(robot):
    assert robot.n_joints == len(robot.joint_limits) > 0


def test_sample_configurations_within_shrunk_limits(robot, rng):
    q = robot.sample_configurations(200, rng, margin=0.05)
    lo = np.array([l for l, _ in robot.joint_limits])
    hi = np.array([h for _, h in robot.joint_limits])
    assert q.shape == (200, robot.n_joints)
    assert np.all(q >= lo) and np.all(q <= hi)


def test_forward_kinematics_single_and_batch(robot, rng):
    q = robot.sample_configurations(5, rng)
    T_batch = robot.forward_kinematics(q)
    assert T_batch.shape == (5, 4, 4)
    T_single = robot.forward_kinematics(q[0])
    assert T_single.shape == (4, 4)
    np.testing.assert_allclose(T_batch[0], T_single)
