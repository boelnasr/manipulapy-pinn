# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np
import pytest

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


def test_split_dataset_is_70_20_10_and_disjoint(rng):
    """Three-way split: exact proportions, no overlap, nothing dropped."""
    from manipulapy_pinn.data import split_dataset

    data = {"q": np.arange(1000), "qdot": np.arange(1000) * 2}
    train, test, evaluation = split_dataset(data, (0.70, 0.20, 0.10), rng)

    assert (len(train["q"]), len(test["q"]), len(evaluation["q"])) == (700, 200, 100)
    joined = np.concatenate([train["q"], test["q"], evaluation["q"]])
    assert len(set(joined.tolist())) == 1000, "splits overlap or drop samples"
    # every key is split consistently, so rows stay aligned across arrays
    for part in (train, test, evaluation):
        assert np.array_equal(part["qdot"], part["q"] * 2)


def test_split_dataset_rejects_bad_fractions_and_tiny_datasets(rng):
    """Silent failure here would produce NaN metrics rather than an error."""
    from manipulapy_pinn.data import split_dataset

    data = {"q": np.arange(100)}
    for bad in [(0.7, 0.2), (0.5, 0.5, 0.5), (0.7, 0.3, 0.0), (0.8, 0.3, -0.1)]:
        with pytest.raises(ValueError):
            split_dataset(data, bad, rng)
    with pytest.raises(ValueError, match="at least one"):
        split_dataset({"q": np.arange(5)}, (0.70, 0.20, 0.10), rng)


def test_split_sizes_always_sum_to_n(rng):
    """The remainder goes to eval, so rounding never loses or duplicates a row."""
    from manipulapy_pinn.data import split_dataset

    for n in (37, 101, 999, 1234):
        parts = split_dataset({"q": np.arange(n)}, (0.70, 0.20, 0.10), rng)
        assert sum(len(p["q"]) for p in parts) == n
