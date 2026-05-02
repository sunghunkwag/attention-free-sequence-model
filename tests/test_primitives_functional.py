import torch

from architectures.primitives import (
    PROPAGATION_REGISTRY,
    UPDATE_REGISTRY,
    build_module,
    stable_exp_decay_kernel,
)


def test_stable_exp_decay_kernel_is_finite_and_monotone():
    decay = torch.tensor([0.2, 0.5, 0.9], dtype=torch.float32)
    k = stable_exp_decay_kernel(decay, length=8)
    assert k.shape == (8, 3)
    assert torch.isfinite(k).all()
    assert torch.all(k[1:] <= k[:-1])


def test_registry_modules_forward_and_backward():
    x = torch.randn(2, 16, 32, requires_grad=True)
    for name in ["depthwise_conv_7", "spectral_filter", "ema", "gated_shift_medium"]:
        module = build_module(name, 32, PROPAGATION_REGISTRY)
        y = module(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()

    for name in ["swiglu", "highway", "squeeze_excite"]:
        module = build_module(name, 32, UPDATE_REGISTRY)
        y = module(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()

    out = 0.0
    for name in ["depthwise_conv_7", "spectral_filter", "ema", "gated_shift_medium"]:
        out = out + build_module(name, 32, PROPAGATION_REGISTRY)(x).mean()
    for name in ["swiglu", "highway", "squeeze_excite"]:
        out = out + build_module(name, 32, UPDATE_REGISTRY)(x).mean()
    out.backward()
    assert x.grad is not None
