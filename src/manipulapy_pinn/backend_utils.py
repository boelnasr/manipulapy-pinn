# SPDX-License-Identifier: AGPL-3.0-or-later
"""Thin wrappers around ``ManipulaPy.backend.use_backend`` for this package's needs.

Multiple PINN tasks call ManipulaPy from the same process — the trajectory
task in particular calls both a live torch FK/dynamics path and (via its
forward-dynamics surrogate) plain torch tensor ops. Scoping every
ManipulaPy call explicitly, instead of relying on whatever backend a
previous call happened to leave active, keeps that safe to compose.
"""
from __future__ import annotations

from ManipulaPy.backend import use_backend


def torch_context():
    """Scoped context: ManipulaPy calls inside differentiate under PyTorch."""
    return use_backend("torch")


def numpy_context():
    """Scoped context: ManipulaPy calls inside run on the fast NumPy backend."""
    return use_backend("numpy")
