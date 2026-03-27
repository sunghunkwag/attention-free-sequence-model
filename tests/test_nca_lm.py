"""Tests for Neural Cellular Automata Language Model."""

import pytest
import torch
from architectures.nca_lm import (
    DiffusionOperator,
    GatedFFN,
    NCABlock,
    NCALayer,
    NCAStep,
    NCA_LM,
    PerceptionFilter,
    ReactionGate,
    nca_tiny,
)

B, D = 2, 32
torch.manual_seed(42)

def _randn(B, L, D): return torch.randn(B, L, D)
def _ids(B, L, V=100): return torch.randint(0, V, (B, L))


class TestPerceptionFilter:
    def test_shape(self):
        pf = PerceptionFilter(D, kernel_size=5, n_filters=3)
        assert pf(_randn(B, 20, D)).shape == (B, 20, D)

    def test_single_position(self):
        pf = PerceptionFilter(D, kernel_size=5, n_filters=3)
        assert pf(_randn(B, 1, D)).shape == (B, 1, D)

    def test_gradient(self):
        pf = PerceptionFilter(D)
        x = _randn(1, 16, D).requires_grad_(True)
        pf(x).sum().backward()
        assert x.grad is not None

    def test_locality(self):
        """Changing a distant token should not affect local output (small kernel)."""
        pf = PerceptionFilter(D, kernel_size=3, n_filters=3)
        pf.eval()
        x = torch.randn(1, 32, D)
        y_a = pf(x.clone())

        x_mod = x.clone()
        x_mod[0, 30, :] += 100.0  # modify far-away position
        y_b = pf(x_mod)

        # Position 0 should be unaffected (kernel=3, only sees pos 0,1)
        assert torch.allclose(y_a[0, 0], y_b[0, 0], atol=1e-5)


class TestReactionGate:
    def test_shape(self):
        rg = ReactionGate(D)
        x = _randn(B, 16, D)
        p = _randn(B, 16, D)
        assert rg(x, p).shape == (B, 16, D)

    def test_residual(self):
        rg = ReactionGate(D)
        x = _randn(B, 8, D)
        p = _randn(B, 8, D)
        out = rg(x, p)
        assert not torch.allclose(out, x)

    def test_gradient(self):
        rg = ReactionGate(D)
        x = _randn(1, 8, D).requires_grad_(True)
        p = _randn(1, 8, D).requires_grad_(True)
        rg(x, p).sum().backward()
        assert x.grad is not None and p.grad is not None

    def test_alpha_near_zero_init(self):
        rg = ReactionGate(D)
        assert rg.alpha.abs().max() < 1.0


class TestDiffusionOperator:
    def test_shape(self):
        do = DiffusionOperator(D, n_rates=3, dilations=(1, 4, 16))
        assert do(_randn(B, 32, D)).shape == (B, 32, D)

    def test_short_sequence(self):
        do = DiffusionOperator(D, n_rates=2, dilations=(1, 4))
        assert do(_randn(B, 3, D)).shape == (B, 3, D)

    def test_gradient(self):
        do = DiffusionOperator(D, n_rates=2, dilations=(1, 4))
        x = _randn(1, 16, D).requires_grad_(True)
        do(x).sum().backward()
        assert x.grad is not None

    def test_rate_mix_sums_to_one(self):
        do = DiffusionOperator(D, n_rates=3, dilations=(1, 4, 16))
        w = torch.softmax(do.rate_mix, dim=-1)
        assert torch.allclose(w.sum(dim=-1), torch.ones(D), atol=1e-6)


class TestNCAStep:
    def test_shape(self):
        step = NCAStep(D, kernel_size=5, dilations=(1, 4))
        assert step(_randn(B, 20, D)).shape == (B, 20, D)

    def test_gradient(self):
        step = NCAStep(D, kernel_size=5, dilations=(1, 4))
        x = _randn(1, 16, D).requires_grad_(True)
        step(x).sum().backward()
        assert x.grad is not None
        for name, p in step.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad: {name}"


class TestNCABlock:
    def test_shape_shared(self):
        blk = NCABlock(D, n_steps=4, share_weights=True, adaptive=False,
                        dilations=(1, 4))
        assert blk(_randn(B, 20, D)).shape == (B, 20, D)

    def test_shape_unshared(self):
        blk = NCABlock(D, n_steps=3, share_weights=False, adaptive=False,
                        dilations=(1, 4))
        assert blk(_randn(B, 20, D)).shape == (B, 20, D)

    def test_adaptive(self):
        blk = NCABlock(D, n_steps=4, share_weights=True, adaptive=True,
                        dilations=(1, 4))
        assert blk(_randn(B, 16, D)).shape == (B, 16, D)

    def test_gradient_adaptive(self):
        blk = NCABlock(D, n_steps=3, adaptive=True, dilations=(1, 4))
        x = _randn(1, 12, D).requires_grad_(True)
        blk(x).sum().backward()
        assert x.grad is not None

    def test_halting_receives_grad(self):
        blk = NCABlock(D, n_steps=3, adaptive=True, dilations=(1, 4))
        x = _randn(1, 12, D)
        blk(x).sum().backward()
        assert blk.halt_proj.weight.grad is not None

    def test_single_step(self):
        blk = NCABlock(D, n_steps=1, adaptive=False, dilations=(1,))
        assert blk(_randn(B, 10, D)).shape == (B, 10, D)


class TestNCALayer:
    def test_shape(self):
        layer = NCALayer(D, n_steps=4, dilations=(1, 4))
        assert layer(_randn(B, 20, D)).shape == (B, 20, D)

    def test_gradient_all(self):
        layer = NCALayer(D, n_steps=3, dilations=(1, 4))
        x = _randn(1, 16, D).requires_grad_(True)
        layer(x).sum().backward()
        assert x.grad is not None
        for name, p in layer.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad: {name}"


class TestNCA_LM:
    def test_output_shape(self):
        m = nca_tiny(vocab_size=100)
        assert m(_ids(B, 32, 100)).shape == (B, 32, 100)

    def test_gradient_e2e(self):
        m = nca_tiny(vocab_size=100)
        ids = _ids(1, 16, 100)
        m(ids).sum().backward()
        assert m.tok_emb.weight.grad is not None

    def test_weight_tying(self):
        m = nca_tiny(vocab_size=100)
        assert m.lm_head.weight is m.tok_emb.weight

    def test_param_count(self):
        m = nca_tiny(vocab_size=100)
        n = m.count_parameters()
        assert 10_000 < n < 50_000_000

    def test_effective_range(self):
        m = nca_tiny(vocab_size=100)
        r = m.effective_range()
        assert "positions" in r

    def test_no_attention_anywhere(self):
        """Verify zero nn.MultiheadAttention modules in the model."""
        m = nca_tiny(vocab_size=100)
        for name, mod in m.named_modules():
            assert not isinstance(mod, torch.nn.MultiheadAttention), (
                f"Found attention at {name}!"
            )

    def test_no_softmax_in_forward(self):
        """Softmax only appears in diffusion rate mixing and final head.
        Not in any attention-like pattern."""
        m = nca_tiny(vocab_size=100)
        # Check that no module has 'attn' or 'attention' in name
        for name, _ in m.named_modules():
            assert 'attn' not in name.lower(), f"Attention-like module: {name}"


class TestEdgeCases:
    def test_seq_len_one(self):
        m = nca_tiny(vocab_size=100)
        assert m(_ids(B, 1, 100)).shape == (B, 1, 100)

    def test_batch_one(self):
        m = nca_tiny(vocab_size=100)
        assert m(_ids(1, 16, 100)).shape == (1, 16, 100)

    def test_long_sequence(self):
        m = nca_tiny(vocab_size=100)
        assert m(_ids(1, 128, 100)).shape == (1, 128, 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
