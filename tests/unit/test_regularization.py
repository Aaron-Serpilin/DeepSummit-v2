"""Tests for regularization layers."""

import torch
import pytest

from ml.model.regularization import DropPath, create_drop_path_schedule


class TestDropPath:
    """Test suite for DropPath (Stochastic Depth)."""

    def test_output_shape_matches_input(self) -> None:
        """DropPath should preserve input shape."""
        drop_path = DropPath(drop_prob=0.1)
        x = torch.randn(4, 49, 256)

        output = drop_path(x)

        assert output.shape == x.shape

    def test_no_drop_during_eval(self) -> None:
        """DropPath should return input unchanged during eval."""
        drop_path = DropPath(drop_prob=0.5)
        x = torch.randn(8, 10, 64)

        drop_path.eval()
        output = drop_path(x)

        assert torch.equal(output, x)

    def test_zero_drop_prob_returns_input(self) -> None:
        """With drop_prob=0.0, should return input unchanged."""
        drop_path = DropPath(drop_prob=0.0)
        x = torch.randn(8, 10, 64)

        drop_path.train()
        output = drop_path(x)

        assert torch.equal(output, x)

    def test_drops_entire_samples_not_features(self) -> None:
        """DropPath should drop entire samples, not individual features.

        If a sample is dropped, all its features should be zero.
        If a sample is kept, all its features should be non-zero (after scaling).
        """
        torch.manual_seed(42)
        drop_path = DropPath(drop_prob=0.5)
        x = torch.ones(100, 5, 32)  # All ones for easy checking

        drop_path.train()
        output = drop_path(x)

        # Check each sample in batch
        for i in range(output.shape[0]):
            sample = output[i]
            # Sample should be either all zeros (dropped) or all scaled (kept)
            is_dropped = (sample == 0).all()
            is_kept = (sample != 0).all()
            assert is_dropped or is_kept

    def test_maintains_expected_value(self) -> None:
        """Expected value should match input (E[output] = x)."""
        torch.manual_seed(42)
        drop_path = DropPath(drop_prob=0.3)
        x = torch.ones(10000, 1, 1)  # Large batch for statistical test

        drop_path.train()
        output = drop_path(x)

        # Mean should be close to 1.0 (the input value)
        assert torch.abs(output.mean() - 1.0) < 0.05

    def test_gradient_flows_when_not_dropped(self) -> None:
        """Gradients should flow through for kept samples."""
        torch.manual_seed(42)
        drop_path = DropPath(drop_prob=0.1)
        x = torch.randn(4, 5, 32, requires_grad=True)

        drop_path.train()
        output = drop_path(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        # At least some gradients should be non-zero (kept samples)
        assert (x.grad != 0).any()

    def test_extra_repr(self) -> None:
        """String representation should include drop_prob."""
        drop_path = DropPath(drop_prob=0.15)

        repr_str = drop_path.extra_repr()

        assert "drop_prob=0.15" in repr_str


class TestCreateDropPathSchedule:
    """Test suite for drop path schedule creation."""

    def test_linear_schedule(self) -> None:
        """Schedule should increase linearly from 0 to drop_path_rate."""
        schedule = create_drop_path_schedule(drop_path_rate=0.1, num_layers=6)

        expected = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
        assert len(schedule) == 6
        for actual, exp in zip(schedule, expected):
            assert abs(actual - exp) < 1e-6

    def test_first_layer_always_zero(self) -> None:
        """First layer should always have drop_prob=0."""
        schedule = create_drop_path_schedule(drop_path_rate=0.2, num_layers=8)

        assert schedule[0] == 0.0

    def test_last_layer_equals_drop_path_rate(self) -> None:
        """Last layer should have drop_prob equal to drop_path_rate."""
        drop_path_rate = 0.15
        schedule = create_drop_path_schedule(drop_path_rate=drop_path_rate, num_layers=4)

        assert abs(schedule[-1] - drop_path_rate) < 1e-6

    def test_single_layer(self) -> None:
        """Single layer should get the full drop_path_rate."""
        schedule = create_drop_path_schedule(drop_path_rate=0.1, num_layers=1)

        assert len(schedule) == 1
        assert schedule[0] == 0.1

    def test_returns_correct_length(self) -> None:
        """Schedule length should match num_layers."""
        for num_layers in [2, 4, 8, 12]:
            schedule = create_drop_path_schedule(drop_path_rate=0.1, num_layers=num_layers)
            assert len(schedule) == num_layers
