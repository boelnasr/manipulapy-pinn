# SPDX-License-Identifier: AGPL-3.0-or-later
"""Small MLP architectures shared by every PINN task in this package."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    """A plain feed-forward network, double precision throughout.

    float64 matches ManipulaPy's own default under the torch backend, and
    every physics residual in this package mixes network outputs with
    ManipulaPy-derived tensors (mass matrices, FK poses) — a dtype mismatch
    there raises, so the network needs to match rather than the other way
    around.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int] = (128, 128, 128),
        activation: type = nn.Tanh,
    ) -> None:
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), activation()]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
