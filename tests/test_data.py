# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np

from manipulapy_pinn.data import (
    generate_forward_dynamics_dataset,
    generate_ik_dataset,
    train_val_split,
)


def test_forward_dynamics_dataset_shapes_and_finite(robot, rng):
    data = generate_forward_dynamics_dataset(robot, 20, rng)
    n = robot.n_joints
    for key in ("q", "qdot", "tau", "qddot"):
        assert data[key].shape == (20, n)
        assert np.all(np.isfinite(data[key]))


def test_ik_dataset_poses_are_true_fk(robot, rng):
    data = generate_ik_dataset(robot, 10, rng)
    assert data["q_reference"].shape == (10, robot.n_joints)
    assert data["pose"].shape == (10, 4, 4)
    assert data["position"].shape == (10, 3)
    # position must be exactly the translation column of the pose
    np.testing.assert_allclose(data["position"], data["pose"][:, :3, 3])
    # and the pose must be the true FK of q_reference, not a stand-in
    recomputed = robot.forward_kinematics(data["q_reference"])
    np.testing.assert_allclose(data["pose"], recomputed)


def test_train_val_split_sizes_and_no_overlap(rng):
    data = {"x": np.arange(100).reshape(100, 1)}
    train, val = train_val_split(data, val_fraction=0.2, rng=rng)
    assert len(train["x"]) + len(val["x"]) == 100
    assert len(val["x"]) == 20
    train_set = set(train["x"].flatten().tolist())
    val_set = set(val["x"].flatten().tolist())
    assert train_set.isdisjoint(val_set)
