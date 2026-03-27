"""Tests for Adaptive Field Network v3 (afn_v3.py)."""
import pytest
import torch
import torch.nn.functional as F
from architectures.afn_v3 import (
    AFN3, AFN3Layer, CoarseNCA, GatedFFN, GatedShiftMixer,
    MultiRateDiffusion, NCAStep, PerceptionFilter, ReactionGate,
    SqueezeExcite,
)

B, D = 2, 32
torch.manual_seed(42)

def _randn(B, L, D): return torch.randn(B, L, D)
def _ids(B, L, V=100): return torch.randint(0, V, (B, L))


class TestPerceptionFilter:
    def test_shape(self):
        assert PerceptionFilter(D)(_randn(B, 20, D)).shape == (B, 20, D)

    def test_gradient(self):
        pf = PerceptionFilter(D)
        x = _randn(1, 16, D).requires_grad_(True)
        pf(x).sum().backward()
        assert x.grad is not None


class TestReactionGate:
    def test_shape(self):
        rg = ReactionGate(D)
        assert rg(_randn(B, 10, D), _randn(B, 10, D)).shape == (B, 10, D)

    def test_alpha_small_init(self):
        rg = ReactionGate(D)
        assert rg.alpha.abs().max().item() < 1.0


class TestMultiRateDiffusion:
    def test_shape(self):
        d = MultiRateDiffusion(D, dilations=(1, 4))
        assert d(_randn(B, 20, D)).shape == (B, 20, D)

    def test_content_adaptive(self):
        d = MultiRateDiffusion(D, dilations=(1, 4, 16))
        x = _randn(B, 16, D)
        w = torch.softmax(d.sel(x), dim=-1)
        assert torch.allclose(w.sum(dim=-1), torch.ones(B, 16), atol=1e-6)


class TestNCAStep:
    def test_shape(self):
        s = NCAStep(D, k=5, dilations=(1, 4))
        assert s(_randn(B, 20, D)).shape == (B, 20, D)

    def test_gradient_all(self):
        s = NCAStep(D, k=5, dilations=(1, 4))
        x = _randn(1, 12, D).requires_grad_(True)
        s(x).sum().backward()
        assert x.grad is not None
        for name, p in s.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad: {name}"


class TestGatedShiftMixer:
    def test_shape(self):
        gsm = GatedShiftMixer(D, shifts=(-4, -1, 1, 4))
        assert gsm(_randn(B, 20, D)).shape == (B, 20, D)

    def test_no_attention(self):
        gsm = GatedShiftMixer(D)
        for n, m in gsm.named_modules():
            assert not isinstance(m, torch.nn.MultiheadAttention)
            assert "attn" not in n.lower()

    def test_gradient(self):
        gsm = GatedShiftMixer(D, shifts=(-4, -1, 1, 4))
        x = _randn(1, 16, D).requires_grad_(True)
        gsm(x).sum().backward()
        assert x.grad is not None

    def test_content_dependent(self):
        gsm = GatedShiftMixer(D, shifts=(-4, -1, 1, 4))
        gsm.eval()
        x1 = torch.randn(1, 16, D)
        x2 = torch.randn(1, 16, D) * 5
        y1 = gsm(x1)
        y2 = gsm(x2)
        assert not torch.allclose(y1, y2)

    def test_shift_reaches_far(self):
        """Information from position 0 should affect position 32."""
        gsm = GatedShiftMixer(D, shifts=(-32, -16, -4, -1, 1, 4, 16, 32))
        x = _randn(1, 64, D).requires_grad_(True)
        out = gsm(x)
        out[0, 32].sum().backward()
        assert x.grad[0, 0].abs().sum() > 0, "Shift=32 should connect pos 0 to pos 32"

    def test_single_position(self):
        gsm = GatedShiftMixer(D, shifts=(-4, -1, 1, 4))
        assert gsm(_randn(B, 1, D)).shape == (B, 1, D)


class TestSqueezeExcite:
    def test_shape(self):
        se = SqueezeExcite(D)
        assert se(_randn(B, 20, D)).shape == (B, 20, D)

    def test_gradient(self):
        se = SqueezeExcite(D)
        x = _randn(1, 16, D).requires_grad_(True)
        se(x).sum().backward()
        assert x.grad is not None


class TestCoarseNCA:
    def test_shape(self):
        c = CoarseNCA(D, stride=4, n_steps=2, dilations=(1, 4))
        assert c(_randn(B, 32, D)).shape == (B, 32, D)

    def test_short_seq(self):
        c = CoarseNCA(D, stride=4, n_steps=1, dilations=(1,))
        assert c(_randn(B, 8, D)).shape == (B, 8, D)

    def test_gradient(self):
        c = CoarseNCA(D, stride=4, n_steps=1, dilations=(1,))
        x = _randn(1, 16, D).requires_grad_(True)
        c(x).sum().backward()
        assert x.grad is not None


class TestAFN3Layer:
    def test_shape(self):
        layer = AFN3Layer(D, nca_steps=2, nca_k=5, dilations=(1, 4),
                          shifts=(-4, -1, 1, 4), coarse_stride=4, coarse_steps=1)
        assert layer(_randn(B, 32, D)).shape == (B, 32, D)

    def test_gradient_all_params(self):
        layer = AFN3Layer(D, nca_steps=2, nca_k=5, dilations=(1, 4),
                          shifts=(-4, -1, 1, 4), coarse_stride=4, coarse_steps=1)
        x = _randn(1, 16, D).requires_grad_(True)
        layer(x).sum().backward()
        assert x.grad is not None
        for name, p in layer.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad: {name}"


class TestAFN3:
    def test_output_shape(self):
        m = AFN3(100, d_model=32, n_layers=2, nca_steps=2, nca_k=5,
                 dilations=(1, 4), shifts=(-4, -1, 1, 4),
                 coarse_stride=4, coarse_steps=1, max_len=64)
        assert m(_ids(B, 32, 100)).shape == (B, 32, 100)

    def test_gradient_e2e(self):
        m = AFN3(100, d_model=32, n_layers=2, nca_steps=2, nca_k=5,
                 dilations=(1, 4), shifts=(-4, -1, 1, 4),
                 coarse_stride=4, coarse_steps=1, max_len=64)
        m(_ids(1, 16, 100)).sum().backward()
        assert m.te.weight.grad is not None

    def test_weight_tying(self):
        m = AFN3(100, d_model=32, n_layers=1, max_len=64)
        assert m.head.weight is m.te.weight

    def test_no_attention_anywhere(self):
        m = AFN3(100, d_model=32, n_layers=2, max_len=64)
        for n, mod in m.named_modules():
            assert not isinstance(mod, torch.nn.MultiheadAttention), \
                f"Found attention at {n}"

    def test_seq_len_one(self):
        m = AFN3(100, d_model=32, n_layers=1, nca_steps=1,
                 dilations=(1,), shifts=(-1, 1),
                 coarse_stride=2, coarse_steps=1, max_len=64)
        assert m(_ids(B, 1, 100)).shape == (B, 1, 100)

    def test_batch_one(self):
        m = AFN3(100, d_model=32, n_layers=1, max_len=64)
        assert m(_ids(1, 16, 100)).shape == (1, 16, 100)

    def test_param_count(self):
        m = AFN3(100, d_model=32, n_layers=2, max_len=64)
        n = m.count_parameters()
        assert 10_000 < n < 10_000_000, f"Params: {n}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
