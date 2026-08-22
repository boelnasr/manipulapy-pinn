# SPDX-License-Identifier: AGPL-3.0-or-later
"""Architecture tests: shapes, dtype, depth selection, and gradient flow."""
import pytest
import torch
import torch.nn as nn

from manipulapy_pinn.models import MLP, ResidualMLP, build_mlp, hidden_sizes


def test_hidden_sizes():
    assert hidden_sizes(64, 3) == (64, 64, 64)
    assert hidden_sizes(8, 1) == (8,)
    for bad in ((0, 3), (64, 0), (-1, 2)):
        with pytest.raises(ValueError):
            hidden_sizes(*bad)


@pytest.mark.parametrize("cls", [MLP, ResidualMLP])
@pytest.mark.parametrize("depth", [1, 2, 5, 8])
def test_shapes_and_dtype(cls, depth):
    """Every architecture maps (N, in) -> (N, out) in float64."""
    net = cls(3, 7, hidden=(16,) * depth)
    out = net(torch.randn(5, 3, dtype=torch.float64))
    assert out.shape == (5, 7)
    assert out.dtype == torch.float64


def test_build_mlp_selects_by_depth():
    """Shallow -> plain MLP; deep uniform -> residual; explicit flag wins."""
    assert isinstance(build_mlp(3, 7, hidden=(64,) * 4), MLP)
    assert isinstance(build_mlp(3, 7, hidden=(64,) * 5), ResidualMLP)
    # Varying widths cannot use identity skips, so they stay plain at any depth.
    assert isinstance(build_mlp(3, 7, hidden=(64, 32, 64, 32, 64, 32)), MLP)
    assert isinstance(build_mlp(3, 7, hidden=(64,) * 8, residual=False), MLP)
    assert isinstance(build_mlp(3, 7, hidden=(64,) * 2, residual=True), ResidualMLP)


def test_residual_mlp_rejects_varying_widths():
    with pytest.raises(ValueError, match="same width"):
        ResidualMLP(3, 7, hidden=(64, 32))


def test_residual_block_is_an_identity_skip():
    """Zeroing both linears in every block leaves the block output unchanged.

    This is what the skip buys: signal (and gradient) reaches the far side of
    a block even when the block's own weights contribute nothing.
    """
    net = ResidualMLP(4, 4, hidden=(4,) * 4)
    blocks = [m for m in net.net if type(m).__name__ == "_ResidualBlock"]
    assert len(blocks) == 2
    with torch.no_grad():
        for b in blocks:
            for lin in (b.lin1, b.lin2):
                lin.weight.zero_()
                lin.bias.zero_()
        x = torch.randn(3, 4, dtype=torch.float64)
        after_input = net.net[0](x)
        for b in blocks:
            assert torch.equal(b(after_input), after_input)


def test_deep_residual_keeps_gradients_alive_in_early_layers():
    """A deep plain Tanh stack starves its first layer; the residual one does not.

    This is the whole reason ResidualMLP exists, so it is worth asserting
    rather than trusting: at depth 12 the plain network's first-layer
    gradient is orders of magnitude smaller than the residual network's.
    """
    torch.manual_seed(0)
    x = torch.randn(16, 3, dtype=torch.float64)

    def first_layer_grad_norm(net):
        net.zero_grad()
        net(x).pow(2).mean().backward()
        first = next(m for m in net.net if isinstance(m, nn.Linear))
        return first.weight.grad.abs().max().item()

    plain = first_layer_grad_norm(MLP(3, 7, hidden=(32,) * 12))
    residual = first_layer_grad_norm(ResidualMLP(3, 7, hidden=(32,) * 12))
    assert residual > plain
    assert residual > 1e-8, "residual path should keep a usable gradient"


def test_networks_are_twice_differentiable_in_their_input():
    """trajectory.py differentiates q(t) twice w.r.t. t, so both must support it."""
    for net in (MLP(1, 2, hidden=(8, 8)), ResidualMLP(1, 2, hidden=(8,) * 6)):
        t = torch.linspace(0, 1, 5, dtype=torch.float64).view(-1, 1).requires_grad_(True)
        q = net(t)
        (qdot,) = torch.autograd.grad(q[:, 0].sum(), t, create_graph=True)
        (qddot,) = torch.autograd.grad(qdot.sum(), t, create_graph=True)
        assert qdot.shape == t.shape and qddot.shape == t.shape
        assert torch.isfinite(qddot).all()
