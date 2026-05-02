import torch
from architectures.afn_v4 import AFN4

def test_afn4_forward_and_grads():
    m = AFN4(vocab_size=64, d_model=16, n_layers=1, nca_steps=1, dilations=(1,), max_len=32)
    x = torch.randint(0, 64, (2, 8))
    y = m(x)
    assert y.shape == (2, 8, 64)
    assert torch.isfinite(y).all()
    y.sum().backward()
    for p in m.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_fourier_mixer_returns_update_only_not_input_residual():
    from architectures.afn_v4 import FourierGateMixer

    mixer = FourierGateMixer(d_model=8, max_len=32)
    x = torch.randn(2, 12, 8)
    with torch.no_grad():
        out = mixer(x)
    assert not torch.allclose(out, x, atol=1e-6), "mixer should return transformed update, not identity+residual"

    with torch.no_grad():
        doubled = mixer(2 * x)
    assert not torch.allclose(doubled - out, x, atol=1e-4), "mixer must not embed hidden +x residual"
