"""Tests for the DeepSummit unified transformer."""

import torch

from ml.model.transformer import DeepSummitTransformer


class TestDeepSummitTransformer:
    """Test suite for the complete DeepSummit model."""

    # Default categorical vocab sizes from TabularTokenizer
    DEFAULT_VOCAB_SIZES = [151, 101, 21, 5, 4, 15]

    def _create_sample_inputs(
        self,
        batch_size: int = 4,
        device: str = "cpu",
        vocab_sizes: list[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Create sample inputs for testing.

        Args:
            batch_size: Number of samples in batch
            device: Device to create tensors on
            vocab_sizes: Categorical vocabulary sizes (uses default if None)

        Returns:
            Dictionary of input tensors with valid values.
        """
        if vocab_sizes is None:
            vocab_sizes = self.DEFAULT_VOCAB_SIZES

        # Create categorical indices within valid range for each feature
        categorical = torch.zeros(batch_size, 6, dtype=torch.long, device=device)
        for i, vocab_size in enumerate(vocab_sizes):
            categorical[:, i] = torch.randint(0, vocab_size, (batch_size,), device=device)

        return {
            "numeric": torch.randn(batch_size, 8, device=device),
            "categorical": categorical,
            "binary": torch.randint(0, 2, (batch_size, 6), device=device),
            "weather": torch.randn(batch_size, 26, 15, device=device),
            "days_before_summit": torch.arange(26, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .float(),
            "day_of_year": torch.randint(1, 366, (batch_size, 26), device=device).float(),
        }

    def test_forward_output_shapes(self) -> None:
        """Forward pass should return correct output shapes."""
        model = DeepSummitTransformer(hidden_size=128, num_layers=2, num_heads=4)
        inputs = self._create_sample_inputs(batch_size=4)

        output = model(**inputs)

        assert output["probability"].shape == (4,)
        assert output["logits"].shape == (4,)
        assert output["attention_weights"] is None

    def test_probability_in_valid_range(self) -> None:
        """Output probabilities should be in [0, 1]."""
        model = DeepSummitTransformer(hidden_size=64, num_layers=2, num_heads=4)
        inputs = self._create_sample_inputs(batch_size=100)

        output = model(**inputs)
        prob = output["probability"]

        assert (prob >= 0.0).all()
        assert (prob <= 1.0).all()

    def test_predict_method(self) -> None:
        """Predict method should return only probability tensor."""
        model = DeepSummitTransformer(hidden_size=64, num_layers=2, num_heads=4)
        inputs = self._create_sample_inputs(batch_size=4)

        prob = model.predict(**inputs)

        assert isinstance(prob, torch.Tensor)
        assert prob.shape == (4,)
        assert (prob >= 0.0).all()
        assert (prob <= 1.0).all()

    def test_attention_weights_returned_when_requested(self) -> None:
        """Should return attention weights when return_attention=True."""
        model = DeepSummitTransformer(hidden_size=64, num_layers=3, num_heads=4)
        inputs = self._create_sample_inputs(batch_size=2)

        model.eval()
        output = model(**inputs, return_attention=True)

        assert output["attention_weights"] is not None
        assert len(output["attention_weights"]) == 3  # num_layers
        for attn in output["attention_weights"]:
            if attn is not None:
                # (batch, num_heads, seq_len, seq_len)
                assert attn.shape == (2, 4, 47, 47)

    def test_get_attention_weights_method(self) -> None:
        """get_attention_weights should return list of attention tensors."""
        model = DeepSummitTransformer(hidden_size=64, num_layers=2, num_heads=4)
        inputs = self._create_sample_inputs(batch_size=2)

        attn_weights = model.get_attention_weights(**inputs)

        assert isinstance(attn_weights, list)
        assert len(attn_weights) == 2

    def test_gradient_flows_through_all_components(self) -> None:
        """Gradients should flow through tokenizers, encoder, and head."""
        model = DeepSummitTransformer(
            hidden_size=64, num_layers=2, num_heads=4, dropout=0.0, drop_path_rate=0.0
        )
        inputs = self._create_sample_inputs(batch_size=2)
        for key in inputs:
            if inputs[key].dtype == torch.float32:
                inputs[key].requires_grad_(True)

        output = model(**inputs)
        loss = output["probability"].sum()
        loss.backward()

        # Check gradients flow through key components
        assert model.cls_token.grad is not None
        assert model.tabular_tokenizer.numeric_tokenizers[0].weight.grad is not None
        assert model.weather_tokenizer.weather_proj.weight.grad is not None
        assert model.encoder.blocks[0].attention.q_proj.weight.grad is not None
        assert model.head.classifier.weight.grad is not None

    def test_total_sequence_length(self) -> None:
        """Token sequence should be [CLS] + tabular(20) + weather(26) = 47."""
        model = DeepSummitTransformer()

        assert model.TOTAL_TOKENS == 47
        assert model.NUM_TABULAR_TOKENS == 20
        assert model.NUM_WEATHER_TOKENS == 26

    def test_modality_ids_template_shape(self) -> None:
        """Modality IDs template should have correct shape and values."""
        model = DeepSummitTransformer()
        template = model.modality_ids_template

        assert template.shape == (1, 47)
        # First token is [CLS] (modality 0)
        assert template[0, 0] == 0
        # Next 20 are tabular (modality 1)
        assert (template[0, 1:21] == 1).all()
        # Weather tokens have modalities 2, 3, 4
        assert (template[0, 21:28] == 2).all()  # 7-day
        assert (template[0, 28:38] == 3).all()  # 30-day
        assert (template[0, 38:47] == 4).all()  # 90-day

    def test_handles_different_batch_sizes(self) -> None:
        """Model should work with varying batch sizes."""
        model = DeepSummitTransformer(hidden_size=64, num_layers=2, num_heads=4)

        for batch_size in [1, 4, 16, 32]:
            inputs = self._create_sample_inputs(batch_size=batch_size)
            output = model(**inputs)
            assert output["probability"].shape == (batch_size,)

    def test_eval_mode_deterministic(self) -> None:
        """Outputs should be deterministic in eval mode."""
        model = DeepSummitTransformer(
            hidden_size=64, num_layers=2, num_heads=4, dropout=0.5
        )
        inputs = self._create_sample_inputs(batch_size=2)

        model.eval()
        with torch.no_grad():
            prob1 = model.predict(**inputs)
            prob2 = model.predict(**inputs)

        assert torch.allclose(prob1, prob2)

    def test_count_parameters(self) -> None:
        """count_parameters should return parameter counts by component."""
        model = DeepSummitTransformer(hidden_size=64, num_layers=2, num_heads=4)

        counts = model.count_parameters()

        assert "cls_token" in counts
        assert "modality_embedding" in counts
        assert "tabular_tokenizer" in counts
        assert "weather_tokenizer" in counts
        assert "encoder" in counts
        assert "head" in counts
        assert "total" in counts
        assert counts["total"] > 0
        assert counts["total"] == sum(v for k, v in counts.items() if k != "total")

    def test_default_configuration(self) -> None:
        """Default configuration should create a valid model."""
        model = DeepSummitTransformer()
        inputs = self._create_sample_inputs(batch_size=2)

        output = model(**inputs)

        assert output["probability"].shape == (2,)
        assert model.hidden_size == 256
        assert model.num_layers == 6

    def test_custom_categorical_vocab_sizes(self) -> None:
        """Model should accept custom categorical vocabulary sizes."""
        custom_vocab_sizes = [50, 100, 20, 5, 4, 10]
        model = DeepSummitTransformer(
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            categorical_vocab_sizes=custom_vocab_sizes,
        )
        inputs = self._create_sample_inputs(batch_size=2, vocab_sizes=custom_vocab_sizes)

        assert (
            model.tabular_tokenizer.categorical_vocab_sizes == custom_vocab_sizes
        )
        # Verify model works with custom vocab sizes
        output = model(**inputs)
        assert output["probability"].shape == (2,)

    def test_output_matches_logits_sigmoid(self) -> None:
        """Probability should equal sigmoid of logits."""
        model = DeepSummitTransformer(hidden_size=64, num_layers=2, num_heads=4)
        inputs = self._create_sample_inputs(batch_size=4)

        output = model(**inputs)
        logits = output["logits"]
        prob = output["probability"]

        expected_prob = torch.sigmoid(logits)
        assert torch.allclose(prob, expected_prob)

    def test_cls_token_is_learnable(self) -> None:
        """[CLS] token should be a learnable parameter."""
        model = DeepSummitTransformer(hidden_size=128)

        assert model.cls_token.requires_grad
        assert model.cls_token.shape == (1, 1, 128)

    def test_encoder_uses_correct_architecture(self) -> None:
        """Encoder should use configured parameters."""
        model = DeepSummitTransformer(
            hidden_size=128,
            num_layers=4,
            num_heads=8,
            drop_path_rate=0.2,
        )

        assert len(model.encoder.blocks) == 4
        assert model.encoder.blocks[0].attention.num_heads == 8
        # Last block should have drop_path_rate
        assert abs(model.encoder.blocks[-1].drop_path1.drop_prob - 0.2) < 1e-6
