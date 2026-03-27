"""
Tests for Fractal Equilibrium Network (fen.py)

Covers:
  - All module output shapes
  - Mask propagation
  - Gradient flow through V-cycle iterations
  - Adaptive halting behaviour
  - Edge cases
  - Parameter count sanity
  - Weight sharing vs unshared
  - V-cycle structural correctness
"""

import pytest
import torch

from architectures.fen import (
    AdaptiveVCycle,
    ChunkAttention,
    FEN,
    FENLayer,
    LocalSmoother,
    Prolongator,
    Restrictor,
    RotaryEmbedding,
    SwiGLU,
    VCycle,
    fen_small,
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
        cos, sin = rope(64)
        assert cos.shape == (64, D // 2)

    def test_expansion(self):
        rope = RotaryEmbedding(D, max_len=32)
        cos, sin = rope(64)
        assert cos.shape[0] == 64


# ── LocalSmoother ───────────────────────────────────────────────────────

class TestLocalSmoother:

    @pytest.mark.parametrize("L,c", [
        (32, 8), (20, 16), (1, 4), (5, 32),
    ])
    def test_shape(self, L, c):
        sm = LocalSmoother(D, n_heads=H, chunk_size=c)
        x = _randn(B, L, D)
        assert sm(x).shape == (B, L, D)

    def test_with_mask(self):
        sm = LocalSmoother(D, n_heads=H, chunk_size=8)
        x = _randn(B, 20, D)
        mask = _mask(B, 20, 15)
        assert sm(x, mask).shape == (B, 20, D)

    def test_residual(self):
        """Output should differ from input (pre-norm residual adds delta)."""
        sm = LocalSmoother(D, n_heads=H, chunk_size=8)
        x = _randn(B, 16, D)
        out = sm(x)
        assert not torch.allclose(out, x)

    def test_gradient(self):
        sm = LocalSmoother(D, n_heads=H, chunk_size=8)
        x = _randn(1, 16, D).requires_grad_(True)
        sm(x).sum().backward()
        assert x.grad is not None and x.grad.abs().sum() > 0


# ── ChunkAttention ──────────────────────────────────────────────────────

class TestChunkAttention:

    @pytest.mark.parametrize("N", [1, 4, 16])
    def test_shape(self, N):
        ca = ChunkAttention(D, n_heads=H)
        h = _randn(B, N, D)
        assert ca(h).shape == (B, N, D)

    def test_gradient(self):
        ca = ChunkAttention(D, n_heads=H)
        h = _randn(1, 8, D).requires_grad_(True)
        ca(h).sum().backward()
        assert h.grad is not None


# ── Restrictor ──────────────────────────────────────────────────────────

class TestRestrictor:

    def test_shape(self):
        r = Restrictor(D, chunk_size=4)
        x = _randn(B, 20, D)
        h, N, pad = r(x)
        assert h.shape == (B, N, D)
        assert N == 5  # ceil(20/4) = 5

    def test_with_mask(self):
        r = Restrictor(D, chunk_size=4)
        x = _randn(B, 20, D)
        mask = _mask(B, 20, 15)
        h, N, pad = r(x, mask)
        assert h.shape == (B, N, D)

    def test_single_chunk(self):
        r = Restrictor(D, chunk_size=64)
        x = _randn(B, 5, D)
        h, N, pad = r(x)
        assert N == 1 and h.shape == (B, 1, D)

    def test_masked_ignores_padding(self):
        r = Restrictor(D, chunk_size=4)
        r.eval()
        x = torch.randn(1, 4, D)
        mask = torch.tensor([[1., 1., 1., 0.]])

        x_a = x.clone(); x_a[0, 3] = 0.0
        x_b = x.clone(); x_b[0, 3] = 1e6

        h_a, _, _ = r(x_a, mask)
        h_b, _, _ = r(x_b, mask)
        assert torch.allclose(h_a, h_b, atol=1e-4)


# ── Prolongator ─────────────────────────────────────────────────────────

class TestProlongator:

    def test_shape(self):
        p = Prolongator(D, chunk_size=4)
        x_fine = _randn(B, 20, D)
        h_coarse = _randn(B, 5, D)
        out = p(x_fine, h_coarse, 20)
        assert out.shape == (B, 20, D)

    def test_gradient(self):
        p = Prolongator(D, chunk_size=4)
        x_fine = _randn(1, 12, D).requires_grad_(True)
        h_coarse = _randn(1, 3, D).requires_grad_(True)
        p(x_fine, h_coarse, 12).sum().backward()
        assert x_fine.grad is not None
        assert h_coarse.grad is not None


# ── VCycle ──────────────────────────────────────────────────────────────

class TestVCycle:

    def test_shape(self):
        vc = VCycle(D, n_heads=H, fine_chunk_size=8,
                    coarse_chunk_sizes=(16,))
        x = _randn(B, 32, D)
        assert vc(x).shape == (B, 32, D)

    def test_shape_two_coarse(self):
        vc = VCycle(D, n_heads=H, fine_chunk_size=4,
                    coarse_chunk_sizes=(8, 32))
        x = _randn(B, 64, D)
        assert vc(x).shape == (B, 64, D)

    def test_with_mask(self):
        vc = VCycle(D, n_heads=H, fine_chunk_size=8,
                    coarse_chunk_sizes=(16,))
        x = _randn(B, 32, D)
        mask = _mask(B, 32, 25)
        assert vc(x, mask).shape == (B, 32, D)

    def test_gradient(self):
        vc = VCycle(D, n_heads=H, fine_chunk_size=8,
                    coarse_chunk_sizes=(16,))
        x = _randn(1, 32, D).requires_grad_(True)
        vc(x).sum().backward()
        assert x.grad is not None

    def test_gradient_deep(self):
        """Gradient flows through all coarse levels."""
        vc = VCycle(D, n_heads=H, fine_chunk_size=4,
                    coarse_chunk_sizes=(8, 32))
        x = _randn(1, 64, D).requires_grad_(True)
        vc(x).sum().backward()
        assert x.grad is not None
        for name, p in vc.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_short_sequence(self):
        vc = VCycle(D, n_heads=H, fine_chunk_size=8,
                    coarse_chunk_sizes=(16,))
        x = _randn(B, 3, D)
        assert vc(x).shape == (B, 3, D)


# ── AdaptiveVCycle ──────────────────────────────────────────────────────

class TestAdaptiveVCycle:

    def test_shape(self):
        avc = AdaptiveVCycle(D, n_heads=H, fine_chunk_size=8,
                             coarse_chunk_sizes=(16,),
                             max_iterations=3)
        x = _randn(B, 32, D)
        assert avc(x).shape == (B, 32, D)

    def test_shape_shared(self):
        avc = AdaptiveVCycle(D, n_heads=H, fine_chunk_size=8,
                             coarse_chunk_sizes=(16,),
                             max_iterations=3, share_weights=True)
        x = _randn(B, 32, D)
        assert avc(x).shape == (B, 32, D)

    def test_shape_unshared(self):
        avc = AdaptiveVCycle(D, n_heads=H, fine_chunk_size=8,
                             coarse_chunk_sizes=(16,),
                             max_iterations=2, share_weights=False)
        x = _randn(B, 32, D)
        assert avc(x).shape == (B, 32, D)

    def test_with_mask(self):
        avc = AdaptiveVCycle(D, n_heads=H, fine_chunk_size=8,
                             coarse_chunk_sizes=(16,),
                             max_iterations=2)
        x = _randn(B, 20, D)
        mask = _mask(B, 20, 15)
        assert avc(x, mask).shape == (B, 20, D)

    def test_gradient(self):
        avc = AdaptiveVCycle(D, n_heads=H, fine_chunk_size=8,
                             coarse_chunk_sizes=(16,),
                             max_iterations=2)
        x = _randn(1, 16, D).requires_grad_(True)
        avc(x).sum().backward()
        assert x.grad is not None

    def test_halting_proj_receives_grad(self):
        avc = AdaptiveVCycle(D, n_heads=H, fine_chunk_size=8,
                             coarse_chunk_sizes=(16,),
                             max_iterations=2)
        x = _randn(1, 16, D)
        avc(x).sum().backward()
        assert avc.halt_proj.weight.grad is not None

    def test_single_iteration(self):
        avc = AdaptiveVCycle(D, n_heads=H, fine_chunk_size=8,
                             coarse_chunk_sizes=(16,),
                             max_iterations=1)
        x = _randn(B, 16, D)
        assert avc(x).shape == (B, 16, D)


# ── SwiGLU ──────────────────────────────────────────────────────────────

class TestSwiGLU:

    def test_shape(self):
        ff = SwiGLU(D)
        assert ff(_randn(B, 10, D)).shape == (B, 10, D)

    def test_gradient(self):
        ff = SwiGLU(D)
        x = _randn(1, 8, D).requires_grad_(True)
        ff(x).sum().backward()
        assert x.grad is not None


# ── FENLayer ────────────────────────────────────────────────────────────

class TestFENLayer:

    def test_shape(self):
        layer = FENLayer(D, n_heads=H, fine_chunk_size=8,
                         coarse_chunk_sizes=(16,), max_iterations=2)
        x = _randn(B, 32, D)
        assert layer(x).shape == (B, 32, D)

    def test_with_mask(self):
        layer = FENLayer(D, n_heads=H, fine_chunk_size=8,
                         coarse_chunk_sizes=(16,), max_iterations=2)
        x = _randn(B, 20, D)
        mask = _mask(B, 20, 15)
        assert layer(x, mask).shape == (B, 20, D)

    def test_gradient_all_params(self):
        layer = FENLayer(D, n_heads=H, fine_chunk_size=8,
                         coarse_chunk_sizes=(16,), max_iterations=2)
        x = _randn(1, 16, D).requires_grad_(True)
        layer(x).sum().backward()
        assert x.grad is not None
        for name, p in layer.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"


# ── Full FEN model ──────────────────────────────────────────────────────

class TestFEN:

    def test_output_shape(self):
        model = fen_small(vocab_size=1000)
        ids = _ids(B, 64, 1000)
        logits = model(ids)
        assert logits.shape == (B, 64, 1000)

    def test_with_mask(self):
        model = fen_small(vocab_size=1000)
        ids = _ids(B, 64, 1000)
        mask = _mask(B, 64, 50)
        assert model(ids, mask).shape == (B, 64, 1000)

    def test_param_count(self):
        model = fen_small(vocab_size=1000)
        n = model.count_parameters()
        assert 1_000_000 < n < 100_000_000, f"Param count {n}"

    def test_gradient_e2e(self):
        model = fen_small(vocab_size=1000)
        ids = _ids(1, 32, 1000)
        model(ids).sum().backward()
        assert model.tok_emb.weight.grad is not None

    def test_weight_tying(self):
        model = fen_small(vocab_size=1000)
        assert model.lm_head.weight is model.tok_emb.weight


# ── Edge cases ──────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_seq_len_one(self):
        model = fen_small(vocab_size=1000)
        ids = _ids(B, 1, 1000)
        assert model(ids).shape == (B, 1, 1000)

    def test_batch_one(self):
        model = fen_small(vocab_size=1000)
        ids = _ids(1, 32, 1000)
        assert model(ids).shape == (1, 32, 1000)

    def test_seq_shorter_than_chunks(self):
        layer = FENLayer(D, n_heads=H, fine_chunk_size=32,
                         coarse_chunk_sizes=(64,), max_iterations=2)
        x = _randn(B, 5, D)
        assert layer(x).shape == (B, 5, D)

    def test_vcycle_no_coarse(self):
        """V-cycle with empty coarse_chunk_sizes should still work."""
        vc = VCycle(D, n_heads=H, fine_chunk_size=8,
                    coarse_chunk_sizes=())
        x = _randn(B, 16, D)
        assert vc(x).shape == (B, 16, D)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
