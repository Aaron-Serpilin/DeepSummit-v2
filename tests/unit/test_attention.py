"""Tests for multi-head attention."""

import pytest
import torch

from ml.model.attention import MultiHeadAttention


class TestMultiHeadAttention:
    """Test suite for MultiHeadAttention."""

    def test_output_shape_matches_input(self) -> None:
        """Attention output should have same shape as input."""
        attn = MultiHeadAttention(hidden_size=256, num_heads=8)
        x = torch.randn(4, 49, 256)

        output, _ = attn(x)

        assert output.shape == x.shape

    def test_hidden_size_divisible_by_num_heads(self) -> None:
        """Should raise error if hidden_size not divisible by num_heads."""
        with pytest.raises(AssertionError):
            MultiHeadAttention(hidden_size=256, num_heads=7)

    def test_attention_weights_shape(self) -> None:
        """Attention weights should have shape (batch, num_heads, seq_len, seq_len)."""
        attn = MultiHeadAttention(hidden_size=128, num_heads=4, dropout=0.0)
        x = torch.randn(2, 10, 128)

        attn.eval()
        _, attn_weights = attn(x)

        assert attn_weights is not None
        assert attn_weights.shape == (2, 4, 10, 10)

    def test_attention_weights_sum_to_one(self) -> None:
        """Attention weights should sum to 1 across key dimension."""
        attn = MultiHeadAttention(hidden_size=64, num_heads=8, dropout=0.0)
        x = torch.randn(2, 5, 64)

        attn.eval()
        _, attn_weights = attn(x)

        # Sum across key dimension (last dim)
        sums = attn_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_gradient_flows_through_all_projections(self) -> None:
        """Gradients should flow through Q, K, V, and output projections."""
        attn = MultiHeadAttention(hidden_size=64, num_heads=4, dropout=0.0)
        x = torch.randn(2, 5, 64, requires_grad=True)

        output, _ = attn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert attn.q_proj.weight.grad is not None
        assert attn.k_proj.weight.grad is not None
        assert attn.v_proj.weight.grad is not None
        assert attn.out_proj.weight.grad is not None

    def test_projections_are_bias_free(self) -> None:
        """All projection layers should not have bias."""
        attn = MultiHeadAttention(hidden_size=256, num_heads=8)

        assert attn.q_proj.bias is None
        assert attn.k_proj.bias is None
        assert attn.v_proj.bias is None
        assert attn.out_proj.bias is None

    def test_handles_different_sequence_lengths(self) -> None:
        """Should work with varying sequence lengths."""
        attn = MultiHeadAttention(hidden_size=128, num_heads=8)

        for seq_len in [10, 49, 100]:
            x = torch.randn(2, seq_len, 128)
            output, _ = attn(x)
            assert output.shape == (2, seq_len, 128)

    def test_dropout_disabled_during_eval(self) -> None:
        """Dropout should be disabled during evaluation."""
        attn = MultiHeadAttention(hidden_size=64, num_heads=4, dropout=0.5)
        x = torch.randn(2, 10, 64)

        attn.eval()
        with torch.no_grad():
            output1, _ = attn(x)
            output2, _ = attn(x)

        assert torch.allclose(output1, output2)

    def test_attention_is_permutation_equivariant(self) -> None:
        """Permuting input sequence should permute output accordingly."""
        attn = MultiHeadAttention(hidden_size=64, num_heads=4, dropout=0.0)
        x = torch.randn(1, 5, 64)

        attn.eval()
        with torch.no_grad():
            output1, _ = attn(x)

            # Permute sequence: [0,1,2,3,4] -> [1,0,3,2,4]
            perm_indices = torch.tensor([1, 0, 3, 2, 4])
            x_perm = x[:, perm_indices, :]
            output2, _ = attn(x_perm)

            # Permuted output should match permutation of original output
            expected = output1[:, perm_indices, :]
            assert torch.allclose(output2, expected, atol=1e-5)
