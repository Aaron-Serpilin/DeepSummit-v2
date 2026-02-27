"""Tests for classification head."""

import torch

from ml.model.head import ClassificationHead


class TestClassificationHead:
    """Test suite for ClassificationHead."""

    def test_output_shape(self) -> None:
        """Output should be (batch,) probability values."""
        head = ClassificationHead(hidden_size=256)
        encoder_output = torch.randn(4, 49, 256)

        prob = head(encoder_output)

        assert prob.shape == (4,)

    def test_output_in_valid_probability_range(self) -> None:
        """Output probabilities should be in [0, 1]."""
        head = ClassificationHead(hidden_size=128)
        encoder_output = torch.randn(100, 49, 128)

        prob = head(encoder_output)

        assert (prob >= 0.0).all()
        assert (prob <= 1.0).all()

    def test_extracts_cls_token_from_position_zero(self) -> None:
        """Should only use the [CLS] token at position 0."""
        head = ClassificationHead(hidden_size=64, dropout=0.0)
        head.eval()

        # Create input with distinctive values at position 0
        encoder_output1 = torch.randn(2, 49, 64)

        # Only modify position 0
        encoder_output2 = encoder_output1.clone()
        encoder_output2[:, 1:, :] = torch.randn(2, 48, 64)

        with torch.no_grad():
            prob1 = head(encoder_output1)
            prob2 = head(encoder_output2)

        # Results should be identical since position 0 is the same
        assert torch.allclose(prob1, prob2)

    def test_gradient_flows_through(self) -> None:
        """Gradients should flow back through the classification head."""
        head = ClassificationHead(hidden_size=64)
        encoder_output = torch.randn(2, 10, 64, requires_grad=True)

        prob = head(encoder_output)
        loss = prob.sum()
        loss.backward()

        assert encoder_output.grad is not None
        assert head.dense.weight.grad is not None
        assert head.classifier.weight.grad is not None

    def test_forward_with_logits_returns_both(self) -> None:
        """forward_with_logits should return both logits and probability."""
        head = ClassificationHead(hidden_size=64)
        encoder_output = torch.randn(4, 10, 64)

        logits, prob = head.forward_with_logits(encoder_output)

        assert logits.shape == (4,)
        assert prob.shape == (4,)
        # Probability should be sigmoid of logits
        assert torch.allclose(prob, torch.sigmoid(logits))

    def test_logits_can_be_any_real_value(self) -> None:
        """Logits (pre-sigmoid) should not be constrained to [0, 1]."""
        head = ClassificationHead(hidden_size=64)
        # Use extreme inputs to get extreme logits
        encoder_output = torch.randn(100, 10, 64) * 10

        logits, prob = head.forward_with_logits(encoder_output)

        # Logits can be outside [0, 1]
        assert (logits < 0).any() or (logits > 1).any()
        # But probabilities are always in [0, 1]
        assert (prob >= 0.0).all()
        assert (prob <= 1.0).all()

    def test_dropout_disabled_during_eval(self) -> None:
        """Dropout should be disabled during evaluation."""
        head = ClassificationHead(hidden_size=64, dropout=0.5)
        encoder_output = torch.randn(2, 10, 64)

        head.eval()
        with torch.no_grad():
            prob1 = head(encoder_output)
            prob2 = head(encoder_output)

        assert torch.allclose(prob1, prob2)

    def test_handles_different_batch_sizes(self) -> None:
        """Should work with varying batch sizes."""
        head = ClassificationHead(hidden_size=128)

        for batch_size in [1, 8, 32, 64]:
            encoder_output = torch.randn(batch_size, 49, 128)
            prob = head(encoder_output)
            assert prob.shape == (batch_size,)

    def test_two_layer_architecture(self) -> None:
        """Should have two linear layers (dense + classifier)."""
        head = ClassificationHead(hidden_size=256)

        assert hasattr(head, 'dense')
        assert hasattr(head, 'classifier')
        assert isinstance(head.dense, torch.nn.Linear)
        assert isinstance(head.classifier, torch.nn.Linear)

        # Check dimensions
        assert head.dense.in_features == 256
        assert head.dense.out_features == 256
        assert head.classifier.in_features == 256
        assert head.classifier.out_features == 1
