"""Tests for transformer encoder components."""

import torch

from ml.model.encoder import TransformerBlock, TransformerEncoder


class TestTransformerBlock:
    """Test suite for TransformerBlock."""

    def test_output_shape_matches_input(self) -> None:
        """Block output should have same shape as input."""
        block = TransformerBlock(hidden_size=256, num_heads=8)
        x = torch.randn(4, 49, 256)

        output, _ = block(x)

        assert output.shape == x.shape

    def test_gradient_flows_through_all_components(self) -> None:
        """Gradients should flow through attention, FFN, and normalization."""
        block = TransformerBlock(hidden_size=64, num_heads=4, dropout=0.0, drop_path_prob=0.0)
        x = torch.randn(2, 5, 64, requires_grad=True)

        output, _ = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert block.norm1.weight.grad is not None
        assert block.norm2.weight.grad is not None
        assert block.attention.q_proj.weight.grad is not None
        assert block.ffn.gate_proj.weight.grad is not None

    def test_residual_connections_work(self) -> None:
        """Output should contain information from input via residual connections."""
        block = TransformerBlock(hidden_size=64, num_heads=4, dropout=0.0, drop_path_prob=0.0)
        x = torch.randn(2, 5, 64)

        # Set all weights to very small values
        with torch.no_grad():
            for param in block.parameters():
                param.mul_(0.01)

        output, _ = block(x)

        # With small weights, output should be close to input due to residuals
        # (not exactly equal because of normalization)
        assert torch.allclose(output, x, atol=0.5)

    def test_drop_path_increases_with_probability(self) -> None:
        """Higher drop_path_prob should result in more zero outputs during training."""
        torch.manual_seed(42)
        x = torch.ones(100, 5, 64)

        # Low drop path probability
        block_low = TransformerBlock(hidden_size=64, num_heads=4, drop_path_prob=0.1)
        block_low.train()
        output_low, _ = block_low(x)
        dropped_low = (output_low == x).all(dim=-1).all(dim=-1).sum()  # Count fully unchanged samples

        # High drop path probability
        torch.manual_seed(42)
        block_high = TransformerBlock(hidden_size=64, num_heads=4, drop_path_prob=0.5)
        block_high.train()
        output_high, _ = block_high(x)
        dropped_high = (output_high == x).all(dim=-1).all(dim=-1).sum()

        # More samples should be dropped with higher probability
        # Note: this is a statistical test, may rarely fail
        assert dropped_high >= dropped_low

    def test_pre_normalization_architecture(self) -> None:
        """Should use pre-normalization (norm before attention/FFN)."""
        block = TransformerBlock(hidden_size=64, num_heads=4)

        # Check that norm layers exist
        assert hasattr(block, 'norm1')
        assert hasattr(block, 'norm2')

        # Verify they are RMSNorm
        from ml.model.normalization import RMSNorm
        assert isinstance(block.norm1, RMSNorm)
        assert isinstance(block.norm2, RMSNorm)

    def test_handles_different_sequence_lengths(self) -> None:
        """Should work with varying sequence lengths."""
        block = TransformerBlock(hidden_size=128, num_heads=8)

        for seq_len in [10, 49, 100]:
            x = torch.randn(2, seq_len, 128)
            output, _ = block(x)
            assert output.shape == (2, seq_len, 128)


class TestTransformerEncoder:
    """Test suite for TransformerEncoder."""

    def test_output_shape_matches_input(self) -> None:
        """Encoder output should have same shape as input."""
        encoder = TransformerEncoder(hidden_size=256, num_layers=6, num_heads=8)
        x = torch.randn(4, 49, 256)

        output, _ = encoder(x)

        assert output.shape == x.shape

    def test_correct_number_of_blocks(self) -> None:
        """Should create the specified number of transformer blocks."""
        num_layers = 6
        encoder = TransformerEncoder(hidden_size=128, num_layers=num_layers)

        assert len(encoder.blocks) == num_layers

    def test_drop_path_schedule_is_linear(self) -> None:
        """Drop path probability should increase linearly across layers."""
        encoder = TransformerEncoder(
            hidden_size=128,
            num_layers=6,
            drop_path_rate=0.1,
        )

        # Extract drop_path probabilities from each block
        drop_probs = [
            block.drop_path1.drop_prob for block in encoder.blocks
        ]

        # Should increase monotonically
        for i in range(len(drop_probs) - 1):
            assert drop_probs[i] <= drop_probs[i + 1]

        # First should be 0, last should be drop_path_rate
        assert drop_probs[0] == 0.0
        assert abs(drop_probs[-1] - 0.1) < 1e-6

    def test_gradient_flows_through_all_layers(self) -> None:
        """Gradients should flow through all transformer blocks."""
        encoder = TransformerEncoder(
            hidden_size=64,
            num_layers=4,
            num_heads=4,
            dropout=0.0,
            drop_path_rate=0.0,
        )
        x = torch.randn(2, 5, 64, requires_grad=True)

        output, _ = encoder(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        # Check first and last block have gradients
        assert encoder.blocks[0].attention.q_proj.weight.grad is not None
        assert encoder.blocks[-1].ffn.gate_proj.weight.grad is not None

    def test_return_attention_weights(self) -> None:
        """Should return attention weights when requested."""
        encoder = TransformerEncoder(
            hidden_size=64,
            num_layers=3,
            num_heads=4,
            dropout=0.0,
        )
        x = torch.randn(2, 5, 64)

        encoder.eval()
        output, attn_weights = encoder(x, return_attention=True)

        assert attn_weights is not None
        assert len(attn_weights) == 3  # One per layer
        for attn in attn_weights:
            if attn is not None:
                assert attn.shape == (2, 4, 5, 5)  # (batch, heads, seq, seq)

    def test_no_attention_weights_by_default(self) -> None:
        """Should not return attention weights by default."""
        encoder = TransformerEncoder(hidden_size=64, num_layers=2)
        x = torch.randn(2, 5, 64)

        output, attn_weights = encoder(x)

        assert attn_weights is None

    def test_final_normalization_applied(self) -> None:
        """Final RMSNorm should be applied after all blocks."""
        encoder = TransformerEncoder(hidden_size=64, num_layers=2)

        assert hasattr(encoder, 'norm')
        from ml.model.normalization import RMSNorm
        assert isinstance(encoder.norm, RMSNorm)

    def test_handles_different_hidden_sizes(self) -> None:
        """Should work with various hidden dimensions."""
        for hidden_size in [64, 128, 256, 512]:
            encoder = TransformerEncoder(
                hidden_size=hidden_size,
                num_layers=2,
                num_heads=8,
            )
            x = torch.randn(2, 10, hidden_size)
            output, _ = encoder(x)
            assert output.shape == (2, 10, hidden_size)

    def test_single_layer_encoder(self) -> None:
        """Should work with a single transformer layer."""
        encoder = TransformerEncoder(
            hidden_size=64,
            num_layers=1,
            num_heads=4,
        )
        x = torch.randn(2, 5, 64)

        output, _ = encoder(x)

        assert output.shape == x.shape
        assert len(encoder.blocks) == 1
