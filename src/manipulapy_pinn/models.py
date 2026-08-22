# SPDX-License-Identifier: AGPL-3.0-or-later
"""Network architectures shared by every PINN task in this package.

Two feed-forward shapes, both double precision:

- :class:`MLP` — a plain stack of ``Linear`` + activation. Fine to about four
  hidden layers.
- :class:`ResidualMLP` — the same, but with an identity skip across every
  pair of equal-width layers. This is what makes a genuinely deep network
  trainable here.

Depth is not free with ``Tanh``. Its derivative is at most 1 and shrinks
away from the origin, so the product of per-layer Jacobians decays with
depth and gradients reaching the early layers of a deep plain stack are
small — the network trains slowly and can plateau above where a shallower
one lands. PINNs feel this more sharply than ordinary regressors because
the physics residual differentiates the network *again*, so second
derivatives depend on that same product. Skip connections give the gradient
a path that does not pass through every activation, which is why depth
beyond a few layers should use :class:`ResidualMLP` rather than
:class:`MLP`.

``build_mlp`` picks between them, and ``hidden_sizes`` turns a
``(width, depth)`` pair into the tuple both constructors take.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn


def hidden_sizes(width: int, depth: int) -> Tuple[int, ...]:
    """``(width,) * depth`` — the tuple form the model constructors take.

    Convenience for command-line flags, where a width and a layer count are
    easier to pass than a variable-length tuple.
    """
    if width < 1 or depth < 1:
        raise ValueError(f"width and depth must both be >= 1, got {width} and {depth}")
    return (width,) * depth


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


class _ResidualBlock(nn.Module):
    """``x + W₂ σ(W₁ σ(x))`` — two layers with an identity skip across them.

    Pre-activation ordering (activation first, then linear) so the skip path
    is a clean identity with no activation on it: the gradient reaches the
    block input undamped regardless of how many blocks sit above it.
    """

    def __init__(self, width: int, activation: type) -> None:
        super().__init__()
        self.act1, self.act2 = activation(), activation()
        self.lin1 = nn.Linear(width, width)
        self.lin2 = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin1(self.act1(x))
        return x + self.lin2(self.act2(h))


class ResidualMLP(nn.Module):
    """A deep network of residual blocks, double precision throughout.

    ``hidden`` is interpreted the same way as for :class:`MLP` — one entry
    per hidden layer — but every layer must be the same width, since an
    identity skip requires matching dimensions. An input projection maps
    ``in_dim`` to that width and a final linear maps it to ``out_dim``.

    An odd number of hidden layers leaves one layer over after pairing them
    into blocks; that layer is appended as a plain (non-residual) layer
    rather than rounding the depth silently up or down.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int] = (128,) * 6,
        activation: type = nn.Tanh,
    ) -> None:
        super().__init__()
        hidden = tuple(hidden)
        if len(set(hidden)) != 1:
            raise ValueError(
                f"ResidualMLP needs every hidden layer at the same width (an identity "
                f"skip cannot change dimension); got {hidden}. Use MLP for varying widths."
            )
        width, depth = hidden[0], len(hidden)

        layers = [nn.Linear(in_dim, width)]
        layers += [_ResidualBlock(width, activation) for _ in range(depth // 2)]
        if depth % 2:
            layers += [activation(), nn.Linear(width, width)]
        layers += [activation(), nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden: Sequence[int] = (128, 128, 128),
    activation: type = nn.Tanh,
    residual: bool = None,
) -> nn.Module:
    """Build the appropriate network for ``hidden``.

    With ``residual=None`` (the default) the choice is made from the depth:
    plain :class:`MLP` up to four hidden layers, :class:`ResidualMLP` beyond
    that — see this module's docstring for why depth needs the skips. Pass
    ``residual=True``/``False`` to decide explicitly. Layers of differing
    widths always use :class:`MLP`, since a skip cannot change dimension.
    """
    hidden = tuple(hidden)
    if residual is None:
        residual = len(hidden) > 4 and len(set(hidden)) == 1
    if residual:
        return ResidualMLP(in_dim, out_dim, hidden=hidden, activation=activation)
    return MLP(in_dim, out_dim, hidden=hidden, activation=activation)
