"""
Tests for the best discovered architecture from automated search.

Tests:
  - Shape correctness for various input sizes
  - Gradient flow through all parameters
  - No attention anywhere in the model
  - Edge cases (seq_len=1, batch=1, max sequence length)
  - Parameter count sanity check
"""

import torch
import torch.nn as nn
import pytest

from architectures.best_discovered import BestDiscovered


@pytest.fixture
def model():
    return BestDiscovered(vocab_size=16, d_model=50, max_len=128)


class TestShape:
    def test_basic_shape(self, model):
        x = torch.randint(0, 16, (4, 32))
        out = model(x)
        assert out.shape == (4, 32, 16)

    def test_batch_1(self, model):
        x = torch.randint(0, 16, (1, 48))
        out = model(x)
        assert out.shape == (1, 48, 16)

    def test_seq_len_1(self, model):
        x = torch.randint(0, 16, (2, 1))
        out = model(x)
        assert out.shape == (2, 1, 16)

    def test_seq_len_64(self, model):
        x = torch.randint(0, 16, (2, 64))
        out = model(x)
        assert out.shape == (2, 64, 16)

    def test_max_seq_len(self, model):
        x = torch.randint(0, 16, (1, 128))
        out = model(x)
        assert out.shape == (1, 128, 16)

    def test_benchmark_seq_lens(self, model):
        """Test with all 3 benchmark task sequence lengths."""
        for L in [32, 48, 64]:
            x = torch.randint(0, 16, (2, L))
            out = model(x)
            assert out.shape == (2, L, 16), f"Failed for L={L}"


class TestGradient:
    def test_gradient_flow(self, model):
        x = torch.randint(0, 16, (2, 32))
        out = model(x)
        loss = out.sum()
        loss.backward()

        params_with_grad = 0
        total_params = 0
        for name, p in model.named_parameters():
            if p.requires_grad:
                total_params += 1
                if p.grad is not None and p.grad.abs().sum() > 0:
                    params_with_grad += 1

        assert params_with_grad > 0, "No gradients flowing"
        # At least 50% of parameters should receive gradients
        assert params_with_grad / total_params > 0.5, \
            f"Only {params_with_grad}/{total_params} params received gradients"

    def test_no_nan_gradients(self, model):
        x = torch.randint(0, 16, (4, 48))
        out = model(x)
        loss = out.sum()
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(p.grad).any(), f"Inf gradient in {name}"


class TestAttentionFree:
    def test_no_multihead_attention(self, model):
        for name, m in model.named_modules():
            assert not isinstance(m, nn.MultiheadAttention), \
                f"Found MultiheadAttention at {name}"

    def test_no_softmax_in_routing(self, model):
        """Verify no softmax modules in the model (except possibly output head)."""
        for name, m in model.named_modules():
            if isinstance(m, nn.Softmax):
                assert 'head' in name, \
                    f"Found Softmax at {name} (only allowed in output head)"


class TestParameterCount:
    def test_param_count_reasonable(self, model):
        params = model.count_parameters()
        # Should be within 20% of 105K target
        assert 80_000 < params < 130_000, \
            f"Parameter count {params} is outside expected range"

    def test_param_count_matches_config(self):
        model = BestDiscovered(vocab_size=16, d_model=50, max_len=128)
        params = model.count_parameters()
        assert params == 103500, \
            f"Expected 103500 params, got {params}"


class TestTraining:
    def test_loss_decreases(self):
        model = BestDiscovered(vocab_size=16, d_model=50, max_len=128)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        model.train()
        losses = []
        for _ in range(20):
            x = torch.randint(0, 16, (8, 32))
            targets = torch.randint(0, 16, (8, 32))
            out = model(x)
            loss = torch.nn.functional.cross_entropy(
                out.reshape(-1, 16), targets.reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease over 20 steps
        assert losses[-1] < losses[0], \
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


class TestEdgeCases:
    def test_deterministic_output(self, model):
        model.eval()
        x = torch.randint(0, 16, (2, 32))
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2), "Non-deterministic output in eval mode"

    def test_different_vocab_sizes(self):
        for vocab in [3, 5, 16, 32]:
            model = BestDiscovered(vocab_size=vocab, d_model=50, max_len=128)
            x = torch.randint(0, vocab, (2, 32))
            out = model(x)
            assert out.shape == (2, 32, vocab)
