import torch
from architectures.search_engine import SearchableModel

def test_searchable_model_forward_and_finite():
    blocks = [{"props": ["depthwise_conv_7", "spectral_filter"], "update": "swiglu"}]
    m = SearchableModel(vocab_size=32, d_model=16, blocks=blocks, max_len=32)
    x = torch.randint(0, 32, (2, 8))
    y = m(x)
    assert y.shape == (2, 8, 32)
    assert torch.isfinite(y).all()
    y.sum().backward()
    for p in m.parameters():
        if p.requires_grad:
            assert p.grad is not None
