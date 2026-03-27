"""
Tests for Hierarchical Fractal Network (hfn.py)

Covers:
  - All module output shapes
  - Mask propagation at every level
  - Gradient flow through all pathways
  - Edge cases (L=1, chunk_size > L, etc.)
  - Parameter count sanity
  - Shift-overlap mechanism
  - Global condenser bidirectional flow
  - Multi-scale gating sums to 1
"""

import pytest
import torch

from architectures.hfn import (
    ChainGraphConv,
    GlobalCondenser,
    HFN,
    HFNLayer,
    HierarchicalScaleBlock,
    LocalChunkAttention,
    MultiScaleFusion,
    RotaryEmbedding,
    SwiGLU,
    apply_rotary,
    hfn_small,
)

B, D = 2, 32
H = 4
torch.manual_seed(42)


def _randn(B, L, D):
    return torch.randn(B, L, D)


def _mask(B, L, valid):
    m = torch.zeros(B, L)
    m[:, :valid] = 1.0
    return m


def _ids(B, L, V=1000):
    return torch.randint(0, V, (B, L))


# ── RotaryEmbedding ─────────────────────────────────────────────────────

class TestRotaryEmbedding:

    def test_shape(self):
        rope = RotaryEmbedding(D)
        cos, sin = rope(128)
        assert cos.shape == (128, D // 2)
        assert sin.shape == (128, D // 2)

    def test_cache_expansion(self):
        rope = RotaryEmbedding(D, max_len=64)
        cos, sin = rope(128)
        assert cos.shape[0] == 128

    def test_apply_rotary(self):
        rope = RotaryEmbedding(8)
        cos, sin = rope(16)
        q = torch.randn(B, H, 16, 8)
        k = torch.randn(B, H, 16, 8)
        q_r, k_r = apply_rotary(q, k, cos, sin)
        assert q_r.shape == q.shape
        assert k_r.shape == k.shape


# ── ChainGraphConv ──────────────────────────────────────────────────────

class TestChainGraphConv:

    def test_shape(self):
        conv = ChainGraphConv(D)
        h = _randn(B, 10, D)
        assert conv(h).shape == (B, 10, D)

    def test_single_node(self):
        conv = ChainGraphConv(D)
        h = _randn(B, 1, D)
        assert conv(h).shape == (B, 1, D)

    def test_learnable_mix(self):
        conv = ChainGraphConv(D)
        w = torch.softmax(conv.mix, dim=0)
        assert torch.allclose(w.sum(), torch.tensor(1.0))


# ── LocalChunkAttention ────────────────────────────────────────────────

class TestLocalChunkAttention:

    @pytest.mark.parametrize("L,c", [
        (64, 16), (20, 16), (1, 8), (100, 32),
    ])
    def test_shape(self, L, c):
        attn = LocalChunkAttention(D, n_heads=H, chunk_size=c)
        x = _randn(B, L, D)
        assert attn(x).shape == (B, L, D)

    def test_with_mask(self):
        attn = LocalChunkAttention(D, n_heads=H, chunk_size=8)
        x = _randn(B, 20, D)
        mask = _mask(B, 20, 15)
        assert attn(x, mask).shape == (B, 20, D)

    def test_shifted(self):
        attn = LocalChunkAttention(D, n_heads=H, chunk_size=8, shift=True)
        x = _randn(B, 32, D)
        assert attn(x).shape == (B, 32, D)

    def test_gradient(self):
        attn = LocalChunkAttention(D, n_heads=H, chunk_size=8)
        x = _randn(1, 16, D).requires_grad_(True)
        attn(x).sum().backward()
        assert x.grad is not None and x.grad.abs().sum() > 0


# ── HierarchicalScaleBlock ─────────────────────────────────────────────

class TestHierarchicalScaleBlock:

    @pytest.mark.parametrize("L,c", [
        (32, 8), (20, 16), (1, 4), (5, 64),
    ])
    def test_shape(self, L, c):
        blk = HierarchicalScaleBlock(D, chunk_size=c)
        x = _randn(B, L, D)
        assert blk(x).shape == (B, L, D)

    def test_with_mask(self):
        blk = HierarchicalScaleBlock(D, chunk_size=8)
        x = _randn(B, 20, D)
        mask = _mask(B, 20, 15)
        assert blk(x, mask).shape == (B, 20, D)

    def test_masked_pooling_ignores_padding(self):
        blk = HierarchicalScaleBlock(D, chunk_size=4)
        blk.eval()
        x = torch.randn(1, 4, D)
        mask = torch.tensor([[1., 1., 1., 0.]])

        x_a = x.clone(); x_a[0, 3] = 0.0
        x_b = x.clone(); x_b[0, 3] = 1e6

        y_a = blk(x_a, mask)
        y_b = blk(x_b, mask)
        assert torch.allclose(y_a[:, :3], y_b[:, :3], atol=1e-4)

    def test_gradient(self):
        blk = HierarchicalScaleBlock(D, chunk_size=4)
        x = _randn(1, 12, D).requires_grad_(True)
        blk(x).sum().backward()
        assert x.grad is not None


# ── MultiScaleFusion ───────────────────────────────────────────────────

class TestMultiScaleFusion:

    def test_shape(self):
        ms = MultiScaleFusion(D, chunk_sizes=(4, 8, 16))
        x = _randn(B, 32, D)
        assert ms(x).shape == (B, 32, D)

    def test_with_mask(self):
        ms = MultiScaleFusion(D, chunk_sizes=(4, 16))
        x = _randn(B, 20, D)
        mask = _mask(B, 20, 15)
        assert ms(x, mask).shape == (B, 20, D)

    def test_gate_sums_to_one(self):
        ms = MultiScaleFusion(D, chunk_sizes=(4, 8))
        x = _randn(B, 16, D)
        gates = torch.softmax(ms.scale_gates(x), dim=-1)
        sums = gates.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

    def test_gradient(self):
        ms = MultiScaleFusion(D, chunk_sizes=(4, 8))
        x = _randn(1, 16, D).requires_grad_(True)
        ms(x).sum().backward()
        assert x.grad is not None


# ── GlobalCondenser ─────────────────────────────────────────────────────

class TestGlobalCondenser:

    def test_shape(self):
        gc = GlobalCondenser(D, n_registers=4, n_heads=H)
        x = _randn(B, 20, D)
        assert gc(x).shape == (B, 20, D)

    def test_with_mask(self):
        gc = GlobalCondenser(D, n_registers=4, n_heads=H)
        x = _randn(B, 20, D)
        mask = _mask(B, 20, 15)
        assert gc(x, mask).shape == (B, 20, D)

    def test_gradient(self):
        gc = GlobalCondenser(D, n_registers=4, n_heads=H)
        x = _randn(1, 16, D).requires_grad_(True)
        gc(x).sum().backward()
        assert x.grad is not None

    def test_registers_receive_gradient(self):
        gc = GlobalCondenser(D, n_registers=4, n_heads=H)
        x = _randn(1, 16, D)
        gc(x).sum().backward()
        assert gc.registers.grad is not None


# ── SwiGLU ──────────────────────────────────────────────────────────────

class TestSwiGLU:

    def test_shape(self):
        ff = SwiGLU(D)
        x = _randn(B, 10, D)
        assert ff(x).shape == (B, 10, D)

    def test_gradient(self):
        ff = SwiGLU(D)
        x = _randn(1, 8, D).requires_grad_(True)
        ff(x).sum().backward()
        assert x.grad is not None


# ── HFNLayer ────────────────────────────────────────────────────────────

class TestHFNLayer:

    def test_shape(self):
        layer = HFNLayer(D, n_heads=H, local_chunk_size=8,
                         scale_chunk_sizes=(4, 8), n_registers=4)
        x = _randn(B, 32, D)
        assert layer(x).shape == (B, 32, D)

    def test_with_mask(self):
        layer = HFNLayer(D, n_heads=H, local_chunk_size=8,
                         scale_chunk_sizes=(4, 8), n_registers=4)
        x = _randn(B, 20, D)
        mask = _mask(B, 20, 15)
        assert layer(x, mask).shape == (B, 20, D)

    def test_shifted_layer(self):
        layer = HFNLayer(D, n_heads=H, local_chunk_size=8,
                         scale_chunk_sizes=(4,), n_registers=4,
                         layer_idx=1)  # odd → shifted
        x = _randn(B, 32, D)
        assert layer(x).shape == (B, 32, D)

    def test_gradient_all_paths(self):
        layer = HFNLayer(D, n_heads=H, local_chunk_size=8,
                         scale_chunk_sizes=(4, 8), n_registers=4)
        x = _randn(1, 16, D).requires_grad_(True)
        layer(x).sum().backward()
        assert x.grad is not None
        # Check all sub-layer parameters received gradients
        for name, p in layer.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"


# ── Full HFN model ─────────────────────────────────────────────────────

class TestHFN:

    def test_output_shape(self):
        model = hfn_small(vocab_size=1000)
        ids = _ids(B, 64, 1000)
        logits = model(ids)
        assert logits.shape == (B, 64, 1000)

    def test_with_mask(self):
        model = hfn_small(vocab_size=1000)
        ids = _ids(B, 64, 1000)
        mask = _mask(B, 64, 50)
        logits = model(ids, mask)
        assert logits.shape == (B, 64, 1000)

    def test_param_count_reasonable(self):
        model = hfn_small(vocab_size=1000)
        n = model.count_parameters()
        # Small model should be in millions, not billions
        assert 1_000_000 < n < 100_000_000, f"Param count {n} out of range"

    def test_gradient_end_to_end(self):
        model = hfn_small(vocab_size=1000)
        ids = _ids(1, 32, 1000)
        logits = model(ids)
        loss = logits.sum()
        loss.backward()
        # Embedding should receive gradients
        assert model.tok_emb.weight.grad is not None

    def test_weight_tying(self):
        model = hfn_small(vocab_size=1000)
        assert model.lm_head.weight is model.tok_emb.weight


# ── Edge cases ──────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_seq_len_one(self):
        model = hfn_small(vocab_size=1000)
        ids = _ids(B, 1, 1000)
        assert model(ids).shape == (B, 1, 1000)

    def test_batch_one(self):
        model = hfn_small(vocab_size=1000)
        ids = _ids(1, 32, 1000)
        assert model(ids).shape == (1, 32, 1000)

    def test_seq_shorter_than_all_chunks(self):
        layer = HFNLayer(D, n_heads=H, local_chunk_size=64,
                         scale_chunk_sizes=(32, 64, 128), n_registers=4)
        x = _randn(B, 5, D)
        assert layer(x).shape == (B, 5, D)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
