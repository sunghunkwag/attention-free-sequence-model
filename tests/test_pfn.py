"""Tests for Particle Field Network."""
import pytest
import torch
from architectures.pfn import (ByteToParticle, ParticleKernel, ParticleStep,
                  ParticleDynamics, FieldReadout, PFNLayer, PFN, pfn_tiny)

B, D, P = 2, 32, 8
torch.manual_seed(42)

def _ids(B, L, V=100): return torch.randint(0, V, (B, L))

class TestByteToParticle:
    def test_output_keys(self):
        bp = ByteToParticle(D, P)
        out = bp(_ids(B, 16))
        assert set(out.keys()) == {'charge', 'position', 'mass'}
    def test_shapes(self):
        bp = ByteToParticle(D, P)
        out = bp(_ids(B, 16))
        assert out['charge'].shape == (B, 16, D)
        assert out['position'].shape == (B, 16, P)
        assert out['mass'].shape == (B, 16, 1)
    def test_mass_bounded(self):
        bp = ByteToParticle(D, P)
        out = bp(_ids(B, 16))
        assert (out['mass'] >= 0).all() and (out['mass'] <= 1).all()

class TestParticleKernel:
    def test_shape(self):
        pk = ParticleKernel(D, P, n_kernels=3)
        c, p, m = torch.randn(B,10,D), torch.randn(B,10,P), torch.rand(B,10,1)
        assert pk(c, p, m).shape == (B, 10, D)
    def test_gradient(self):
        pk = ParticleKernel(D, P, n_kernels=3)
        c = torch.randn(1,8,D, requires_grad=True)
        p = torch.randn(1,8,P, requires_grad=True)
        m = torch.rand(1,8,1)
        pk(c, p, m).sum().backward()
        assert c.grad is not None and p.grad is not None
    def test_no_attention(self):
        pk = ParticleKernel(D, P)
        for n, mod in pk.named_modules():
            assert not isinstance(mod, torch.nn.MultiheadAttention)

class TestParticleStep:
    def test_shapes(self):
        ps = ParticleStep(D, P, n_kernels=3)
        c, p, m = torch.randn(B,10,D), torch.randn(B,10,P), torch.rand(B,10,1)
        c2, p2, m2 = ps(c, p, m)
        assert c2.shape == c.shape
        assert p2.shape == p.shape
        assert m2.shape == m.shape
    def test_position_changes(self):
        ps = ParticleStep(D, P, n_kernels=3)
        c, p, m = torch.randn(B,10,D), torch.randn(B,10,P), torch.rand(B,10,1)
        _, p2, _ = ps(c, p, m)
        assert not torch.allclose(p, p2)
    def test_gradient(self):
        ps = ParticleStep(D, P)
        c = torch.randn(1,6,D, requires_grad=True)
        p = torch.randn(1,6,P, requires_grad=True)
        m = torch.rand(1,6,1, requires_grad=True)
        c2, p2, m2 = ps(c, p, m)
        (c2.sum() + p2.sum() + m2.sum()).backward()
        assert c.grad is not None

class TestParticleDynamics:
    def test_shape(self):
        pd = ParticleDynamics(D, P, n_kernels=3, n_steps=3)
        c, p, m = torch.randn(B,8,D), torch.randn(B,8,P), torch.rand(B,8,1)
        c2, p2, m2 = pd(c, p, m)
        assert c2.shape == c.shape

class TestPFN:
    def test_output_shape(self):
        m = pfn_tiny(vocab_size=100)
        assert m(_ids(B, 20, 100)).shape == (B, 20, 100)
    def test_gradient_e2e(self):
        m = pfn_tiny(vocab_size=100)
        ids = _ids(1, 12, 100)
        m(ids).sum().backward()
        assert m.spawn.byte_to_charge.weight.grad is not None
    def test_no_attention(self):
        m = pfn_tiny(vocab_size=100)
        for n, mod in m.named_modules():
            assert not isinstance(mod, torch.nn.MultiheadAttention), n
            assert 'attn' not in n.lower(), n
    def test_param_count(self):
        m = pfn_tiny(vocab_size=100)
        assert 10_000 < m.count_parameters() < 10_000_000
    def test_seq_len_one(self):
        m = pfn_tiny(vocab_size=100)
        assert m(_ids(B, 1, 100)).shape == (B, 1, 100)
    def test_batch_one(self):
        m = pfn_tiny(vocab_size=100)
        assert m(_ids(1, 16, 100)).shape == (1, 16, 100)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
