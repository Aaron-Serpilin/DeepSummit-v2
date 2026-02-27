"""Tests for normalization layers."""

import torch
import pytest

from ml.model.normalization import RMSNorm


class TestRMSNorm:
    """Test suite for RMSNorm layer."""

    def test_output_shape_matches_input(self) -> None:
        """RMSNorm should preserve input shape."""
        norm = RMSNorm(dim=256)
        x = torch.randn(4, 49, 256)  # (batch, seq_len, hidden)

        output = norm(x)

        assert output.shape == x.shape

    def test_normalized_rms_is_approximately_one(self) -> None:
        """After normalization, RMS should be close to 1 (before scaling)."""
        norm = RMSNorm(dim=256)
        x = torch.randn(4, 49, 256) * 10  # Large values

        output = norm(x)

        # Compute RMS of output (before weight scaling, weight is ones by default)
        rms = torch.sqrt(output.pow(2).mean(dim=-1))
        # Should be close to 1.0 (the weight is all ones initially)
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)

    def test_weight_parameter_is_learnable(self) -> None:
        """The weight parameter should be trainable."""
        norm = RMSNorm(dim=256)

        assert norm.weight.requires_grad
        assert norm.weight.shape == (256,)

    def test_weight_initialized_to_ones(self) -> None:
        """Weight should be initialized to ones."""
        norm = RMSNorm(dim=128)

        assert torch.allclose(norm.weight, torch.ones(128))

    def test_handles_different_batch_sizes(self) -> None:
        """RMSNorm should work with various batch sizes."""
        norm = RMSNorm(dim=64)

        for batch_size in [1, 8, 32]:
            x = torch.randn(batch_size, 10, 64)
            output = norm(x)
            assert output.shape == (batch_size, 10, 64)

    def test_gradient_flows_through(self) -> None:
        """Gradients should flow through RMSNorm."""
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 5, 64, requires_grad=True)

        output = norm(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert norm.weight.grad is not None

    def test_eps_prevents_division_by_zero(self) -> None:
        """Epsilon should prevent NaN when input is zero."""
        norm = RMSNorm(dim=32, eps=1e-6)
        x = torch.zeros(2, 5, 32)

        output = norm(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
