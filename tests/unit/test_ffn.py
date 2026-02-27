"""Tests for feed-forward network layers."""

import torch
import pytest

from ml.model.ffn import SwiGLU


class TestSwiGLU:
    """Test suite for SwiGLU feed-forward network."""

    def test_output_shape_matches_input(self) -> None:
        """SwiGLU should preserve input shape."""
        ffn = SwiGLU(hidden_size=256)
        x = torch.randn(4, 49, 256)

        output = ffn(x)

        assert output.shape == x.shape

    def test_default_ffn_hidden_size(self) -> None:
        """Default intermediate size should be ~2.67 × hidden_size."""
        hidden_size = 256
        ffn = SwiGLU(hidden_size=hidden_size)

        expected = int(hidden_size * 8 / 3)
        assert ffn.ffn_hidden_size == expected

    def test_custom_ffn_hidden_size(self) -> None:
        """Should allow custom intermediate dimension."""
        ffn = SwiGLU(hidden_size=256, ffn_hidden_size=512)

        assert ffn.ffn_hidden_size == 512

    def test_gradient_flows_through(self) -> None:
        """Gradients should flow through all three projections."""
        ffn = SwiGLU(hidden_size=64, dropout=0.0)
        x = torch.randn(2, 5, 64, requires_grad=True)

        output = ffn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert ffn.gate_proj.weight.grad is not None
        assert ffn.value_proj.weight.grad is not None
        assert ffn.output_proj.weight.grad is not None

    def test_dropout_disabled_during_eval(self) -> None:
        """Dropout should be disabled during evaluation mode."""
        ffn = SwiGLU(hidden_size=64, dropout=0.5)
        x = torch.randn(100, 10, 64)

        ffn.eval()
        with torch.no_grad():
            output1 = ffn(x)
            output2 = ffn(x)

        # Outputs should be identical in eval mode
        assert torch.allclose(output1, output2)

    def test_handles_different_batch_sizes(self) -> None:
        """SwiGLU should work with various batch sizes."""
        ffn = SwiGLU(hidden_size=128)

        for batch_size in [1, 8, 32]:
            x = torch.randn(batch_size, 20, 128)
            output = ffn(x)
            assert output.shape == (batch_size, 20, 128)

    def test_projections_are_bias_free(self) -> None:
        """All linear layers should not have bias terms."""
        ffn = SwiGLU(hidden_size=256)

        assert ffn.gate_proj.bias is None
        assert ffn.value_proj.bias is None
        assert ffn.output_proj.bias is None

    def test_gating_mechanism_works(self) -> None:
        """Output should be affected by both gate and value paths."""
        ffn = SwiGLU(hidden_size=64, dropout=0.0)
        x = torch.randn(2, 5, 64)

        # Zero out gate projection weights -> output should be near zero
        with torch.no_grad():
            ffn.gate_proj.weight.zero_()

        output = ffn(x)

        # Output should be very small (gating causes near-zero values)
        assert output.abs().mean() < 1e-3
