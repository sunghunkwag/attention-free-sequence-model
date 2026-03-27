"""
Unit tests for fractal_gnn.py

Covers:
  - Output shapes for all modules
  - Masked vs unmasked pooling correctness
  - Padding edge cases (L exactly divisible, L=1, L < chunk_size)
  - The mask_pad bug fix (mask=None with pad_len > 0)
  - Gradient flow through the gated residual path
  - Multi-scale layer fusion dimensions
  - DeeperGraphConv residual stacking
  - layer_norm and dropout options
"""

import pytest
import torch
from architectures.fractal_gnn_original import (
    DeeperGraphConv,
    FractalGNNBlock,
    MultiScaleFractalLayer,
    SimpleGraphConv,
)

# ── helpers ──────────────────────────────────────────────────────────────────

B, D = 2, 32          # batch size and hidden dim used across most tests
torch.manual_seed(42)


def _randn(B, L, D):
    return torch.randn(B, L, D)


def _mask(B, L, valid_len):
    """Create a mask where the first *valid_len* positions per batch are 1."""
    m = torch.zeros(B, L)
    m[:, :valid_len] = 1.0
    return m


# ── SimpleGraphConv ──────────────────────────────────────────────────────────

class TestSimpleGraphConv:

    def test_output_shape(self):
        conv = SimpleGraphConv(D)
        h = _randn(B, 10, D)
        out = conv(h)
        assert out.shape == (B, 10, D)

    def test_single_node(self):
        """A graph with a single node should still work (no neighbours)."""
        conv = SimpleGraphConv(D)
        h = _randn(B, 1, D)
        out = conv(h)
        assert out.shape == (B, 1, D)

    def test_two_nodes(self):
        conv = SimpleGraphConv(D)
        h = _randn(B, 2, D)
        out = conv(h)
        assert out.shape == (B, 2, D)


# ── DeeperGraphConv ─────────────────────────────────────────────────────────

class TestDeeperGraphConv:

    def test_output_shape(self):
        conv = DeeperGraphConv(D, depth=3)
        h = _randn(B, 8, D)
        out = conv(h)
        assert out.shape == (B, 8, D)

    def test_residual_non_zero(self):
        """The residual connections mean output differs from a fresh init's
        linear projections alone."""
        conv = DeeperGraphConv(D, depth=2)
        h = torch.ones(B, 4, D)
        out = conv(h)
        # Output should not be identical to input (the layers add deltas)
        assert not torch.allclose(out, h)


# ── FractalGNNBlock — shape tests ───────────────────────────────────────────

class TestFractalGNNBlockShapes:

    @pytest.mark.parametrize("L,c", [
        (128, 128),    # exact multiple
        (130, 128),    # requires padding
        (1, 16),       # very short sequence
        (15, 16),      # one less than chunk
        (256, 64),     # multiple chunks exact
        (100, 33),     # awkward sizes
    ])
    def test_output_shape(self, L, c):
        blk = FractalGNNBlock(D, chunk_size=c)
        x = _randn(B, L, D)
        y = blk(x)
        assert y.shape == (B, L, D)

    def test_with_mask(self):
        L, c = 130, 64
        blk = FractalGNNBlock(D, chunk_size=c)
        x = _randn(B, L, D)
        mask = _mask(B, L, valid_len=100)
        y = blk(x, mask=mask)
        assert y.shape == (B, L, D)

    def test_gnn_depth(self):
        blk = FractalGNNBlock(D, chunk_size=16, gnn_depth=4)
        x = _randn(B, 48, D)
        y = blk(x)
        assert y.shape == (B, 48, D)


# ── FractalGNNBlock — mask_pad bug fix ──────────────────────────────────────

class TestMaskPadBugFix:
    """The original code raised UnboundLocalError when mask=None and
    pad_len > 0.  These tests verify the fix."""

    def test_no_mask_with_padding(self):
        """mask=None, L not a multiple of chunk_size → pad_len > 0."""
        blk = FractalGNNBlock(D, chunk_size=16)
        x = _randn(B, 20, D)            # 20 % 16 = 4 → pad_len = 12
        # Should NOT raise
        y = blk(x, mask=None)
        assert y.shape == (B, 20, D)

    def test_no_mask_no_padding(self):
        """mask=None, L exactly divisible → pad_len = 0."""
        blk = FractalGNNBlock(D, chunk_size=16)
        x = _randn(B, 32, D)
        y = blk(x, mask=None)
        assert y.shape == (B, 32, D)

    def test_mask_with_padding(self):
        """mask provided, pad_len > 0."""
        blk = FractalGNNBlock(D, chunk_size=16)
        L = 20
        x = _randn(B, L, D)
        mask = _mask(B, L, valid_len=18)
        y = blk(x, mask=mask)
        assert y.shape == (B, L, D)

    def test_mask_no_padding(self):
        """mask provided, pad_len = 0."""
        blk = FractalGNNBlock(D, chunk_size=16)
        L = 32
        x = _randn(B, L, D)
        mask = _mask(B, L, valid_len=30)
        y = blk(x, mask=mask)
        assert y.shape == (B, L, D)


# ── FractalGNNBlock — masked pooling correctness ────────────────────────────

class TestMaskedPooling:

    def test_masked_mean_ignores_padding(self):
        """When trailing tokens are masked out, their values should not
        affect the chunk pool.  We verify by setting masked positions to
        a huge value and checking the output is unchanged."""
        blk = FractalGNNBlock(D, chunk_size=4)
        blk.eval()

        L = 4
        x_clean = torch.randn(1, L, D)
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])  # last token masked

        # Run with clean zeros in the masked slot
        x_a = x_clean.clone()
        x_a[0, 3, :] = 0.0
        y_a = blk(x_a, mask=mask)

        # Run with huge values in the masked slot
        x_b = x_clean.clone()
        x_b[0, 3, :] = 1e6
        y_b = blk(x_b, mask=mask)

        # The first 3 token outputs should be identical
        assert torch.allclose(y_a[:, :3], y_b[:, :3], atol=1e-5), (
            "Masked positions leaked into chunk pool"
        )


# ── FractalGNNBlock — gradient flow ─────────────────────────────────────────

class TestGradientFlow:

    def test_gradient_flows_through_gate(self):
        blk = FractalGNNBlock(D, chunk_size=8)
        x = _randn(1, 16, D).requires_grad_(True)
        y = blk(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_gradient_with_mask(self):
        blk = FractalGNNBlock(D, chunk_size=8)
        x = _randn(1, 16, D).requires_grad_(True)
        mask = _mask(1, 16, 12)
        y = blk(x, mask=mask)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None


# ── FractalGNNBlock — options ────────────────────────────────────────────────

class TestBlockOptions:

    def test_layer_norm(self):
        blk = FractalGNNBlock(D, chunk_size=8, layer_norm=True)
        x = _randn(B, 24, D)
        y = blk(x)
        assert y.shape == (B, 24, D)

    def test_dropout(self):
        blk = FractalGNNBlock(D, chunk_size=8, dropout=0.1)
        blk.train()
        x = _randn(B, 24, D)
        y = blk(x)
        assert y.shape == (B, 24, D)


# ── MultiScaleFractalLayer ──────────────────────────────────────────────────

class TestMultiScaleFractalLayer:

    def test_output_shape(self):
        layer = MultiScaleFractalLayer(D, chunk_sizes=(4, 8, 16))
        x = _randn(B, 32, D)
        y = layer(x)
        assert y.shape == (B, 32, D)

    def test_with_mask(self):
        layer = MultiScaleFractalLayer(D, chunk_sizes=(4, 16))
        L = 20
        x = _randn(B, L, D)
        mask = _mask(B, L, valid_len=18)
        y = layer(x, mask=mask)
        assert y.shape == (B, L, D)

    def test_single_scale_equivalent(self):
        """With one chunk size the layer should still produce (B, L, D)."""
        layer = MultiScaleFractalLayer(D, chunk_sizes=(8,))
        x = _randn(B, 16, D)
        y = layer(x)
        assert y.shape == (B, 16, D)

    def test_gradient(self):
        layer = MultiScaleFractalLayer(D, chunk_sizes=(4, 8))
        x = _randn(1, 16, D).requires_grad_(True)
        y = layer(x)
        y.sum().backward()
        assert x.grad is not None

    def test_options_propagated(self):
        layer = MultiScaleFractalLayer(
            D, chunk_sizes=(4, 8), gnn_depth=2, dropout=0.1, layer_norm=True,
        )
        x = _randn(B, 16, D)
        y = layer(x)
        assert y.shape == (B, 16, D)


# ── Edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_sequence_length_one(self):
        blk = FractalGNNBlock(D, chunk_size=4)
        x = _randn(B, 1, D)
        y = blk(x)
        assert y.shape == (B, 1, D)

    def test_chunk_size_one(self):
        """chunk_size=1 means every token is its own chunk."""
        blk = FractalGNNBlock(D, chunk_size=1)
        x = _randn(B, 10, D)
        y = blk(x)
        assert y.shape == (B, 10, D)

    def test_chunk_size_larger_than_seq(self):
        blk = FractalGNNBlock(D, chunk_size=64)
        x = _randn(B, 5, D)
        y = blk(x)
        assert y.shape == (B, 5, D)

    def test_batch_size_one(self):
        blk = FractalGNNBlock(D, chunk_size=4)
        x = _randn(1, 12, D)
        y = blk(x)
        assert y.shape == (1, 12, D)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
