import torch
from architectures.afn_v2 import AFN2

def test_afn2_forward_and_grads():
    m = AFN2(vocab_size=64, d_model=16, n_layers=1, nca_steps=1, dilations=(1,), max_len=32)
    x = torch.randint(0, 64, (2, 8))
    y = m(x)
    assert y.shape == (2, 8, 64)
    assert torch.isfinite(y).all()
    y.sum().backward()
    for p in m.parameters():
        if p.requires_grad:
            assert p.grad is not None
