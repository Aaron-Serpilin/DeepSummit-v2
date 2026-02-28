"""DeepSummit Unified Multimodal Transformer.

This module implements the complete DeepSummit model architecture that jointly
attends over expedition metadata and multi-scale historical weather data to
predict mountaineering expedition success probability.

Architecture Overview:
    Inputs:
        - Tabular features: 8 numeric + 6 categorical + 6 binary = 20 tokens
        - Weather features: 7 (7d) + 10 (30d) + 9 (90d) = 26 tokens

    Token Sequence:
        [CLS] + [Tabular x 20] + [Weather x 26] = 47 tokens

    Processing:
        1. Tokenize each modality separately
        2. Add [CLS] token at position 0
        3. Add modality embeddings to distinguish data sources
        4. Apply transformer encoder (joint attention over all tokens)
        5. Extract [CLS] representation for classification
"""

from typing import cast

import torch
import torch.nn as nn

from ml.model.embeddings import ModalityEmbedding
from ml.model.encoder import TransformerEncoder
from ml.model.head import ClassificationHead
from ml.model.tokenizer import TabularTokenizer, WeatherTokenizer


class DeepSummitTransformer(nn.Module):
    """Unified multimodal transformer for expedition success prediction.

    FT-Transformer inspired architecture with per-feature tokenization for
    tabular data and Time2Vec temporal encoding for weather time series.

    The model uses:
    - Pre-normalization (RMSNorm before attention and FFN)
    - SwiGLU feed-forward networks
    - Stochastic depth regularization
    - Flash Attention 2 (via PyTorch's scaled_dot_product_attention)
    - Joint attention across all modalities at every layer

    Args:
        hidden_size: Token embedding dimension (default: 256)
        num_layers: Number of transformer blocks (default: 6)
        num_heads: Number of attention heads (default: 8)
        ffn_hidden_size: FFN intermediate dimension (default: ~2.67 × hidden_size)
        dropout: Dropout probability (default: 0.1)
        drop_path_rate: Maximum stochastic depth rate (default: 0.1)
        numeric_features: Number of numeric tabular features (default: 8)
        categorical_vocab_sizes: Vocabulary sizes for categorical features
        binary_features: Number of binary tabular features (default: 6)
        weather_vars: Number of weather variables per timestep (default: 15)
        time2vec_features: Number of Time2Vec features per temporal dim (default: 32)
    """

    # Number of tokens from each source
    NUM_TABULAR_TOKENS = 20  # 8 numeric + 6 categorical + 6 binary
    NUM_WEATHER_TOKENS = 26  # 7 (7d) + 10 (30d) + 9 (90d)
    TOTAL_TOKENS = 1 + NUM_TABULAR_TOKENS + NUM_WEATHER_TOKENS  # 47 with [CLS]

    # Modality type IDs
    MODALITY_CLS = 0
    MODALITY_TABULAR = 1
    MODALITY_WEATHER_7D = 2
    MODALITY_WEATHER_30D = 3
    MODALITY_WEATHER_90D = 4
    NUM_MODALITIES = 5

    # Type annotation for registered buffer
    modality_ids_template: torch.Tensor

    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_hidden_size: int | None = None,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        numeric_features: int = 8,
        categorical_vocab_sizes: list[int] | None = None,
        binary_features: int = 6,
        weather_vars: int = 15,
        time2vec_features: int = 32,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Learnable [CLS] token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        # Modality embeddings to distinguish token types
        self.modality_embedding = ModalityEmbedding(
            num_modalities=self.NUM_MODALITIES,
            hidden_size=hidden_size,
        )

        # Tabular tokenizer: expedition metadata → 20 tokens
        self.tabular_tokenizer = TabularTokenizer(
            hidden_size=hidden_size,
            numeric_features=numeric_features,
            categorical_vocab_sizes=categorical_vocab_sizes,
            binary_features=binary_features,
        )

        # Weather tokenizer: multi-scale time series → 26 tokens
        self.weather_tokenizer = WeatherTokenizer(
            hidden_size=hidden_size,
            weather_vars=weather_vars,
            time2vec_features=time2vec_features,
        )

        # Transformer encoder: joint attention over all tokens
        self.encoder = TransformerEncoder(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_hidden_size=ffn_hidden_size,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
        )

        # Classification head: [CLS] → probability
        self.head = ClassificationHead(
            hidden_size=hidden_size,
            dropout=dropout,
        )

        # Pre-compute modality ID template for efficiency
        # Shape: (1, 47) - will be expanded to batch size during forward
        self.register_buffer(
            "modality_ids_template",
            self._create_modality_ids_template(),
        )

    def _create_modality_ids_template(self) -> torch.Tensor:
        """Create template of modality IDs for the full sequence.

        Returns:
            Tensor of shape (1, 47) with modality type IDs.
        """
        modality_ids = []

        # [CLS] token
        modality_ids.append(self.MODALITY_CLS)

        # Tabular tokens (20)
        modality_ids.extend([self.MODALITY_TABULAR] * self.NUM_TABULAR_TOKENS)

        # Weather tokens by scale
        modality_ids.extend([self.MODALITY_WEATHER_7D] * 7)   # 7-day: 7 tokens
        modality_ids.extend([self.MODALITY_WEATHER_30D] * 10)  # 30-day: 10 tokens
        modality_ids.extend([self.MODALITY_WEATHER_90D] * 9)   # 90-day: 9 tokens

        return torch.tensor(modality_ids, dtype=torch.long).unsqueeze(0)

    def forward(
        self,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
        binary: torch.Tensor,
        weather: torch.Tensor,
        days_before_summit: torch.Tensor,
        day_of_year: torch.Tensor,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        """Forward pass through the DeepSummit transformer.

        Args:
            numeric: Numeric tabular features (batch, 8)
            categorical: Categorical feature indices (batch, 6)
            binary: Binary features as 0/1 (batch, 6)
            weather: Weather data (batch, 26, 15) - 26 timesteps, 15 variables
            days_before_summit: Days before summit for each timestep (batch, 26)
            day_of_year: Day of year for each timestep (batch, 26)
            return_attention: If True, return attention weights for visualization

        Returns:
            Dictionary containing:
                - probability: Success probability (batch,)
                - logits: Raw logits before sigmoid (batch,)
                - attention_weights: List of attention weights per layer (if requested)
        """
        batch_size = numeric.shape[0]
        device = numeric.device

        # 1. Tokenize tabular features → (batch, 20, hidden_size)
        tabular_tokens = self.tabular_tokenizer(numeric, categorical, binary)

        # 2. Tokenize weather features → (batch, 26, hidden_size)
        weather_tokens = self.weather_tokenizer(
            weather, days_before_summit, day_of_year
        )

        # 3. Expand [CLS] token to batch → (batch, 1, hidden_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # 4. Concatenate all tokens: [CLS] + tabular + weather
        # Shape: (batch, 47, hidden_size)
        tokens = torch.cat([cls_tokens, tabular_tokens, weather_tokens], dim=1)

        # 5. Add modality embeddings
        # Expand template to batch size: (1, 47) → (batch, 47)
        modality_ids = self.modality_ids_template.expand(batch_size, -1).to(device)
        modality_emb = self.modality_embedding(modality_ids)
        tokens = tokens + modality_emb

        # 6. Apply transformer encoder
        encoded, attention_weights = self.encoder(
            tokens, return_attention=return_attention
        )

        # 7. Classification from [CLS] token
        logits, probability = self.head.forward_with_logits(encoded)

        return {
            "probability": probability,
            "logits": logits,
            "attention_weights": attention_weights,
        }

    def predict(
        self,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
        binary: torch.Tensor,
        weather: torch.Tensor,
        days_before_summit: torch.Tensor,
        day_of_year: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience method for inference - returns only probability.

        Args:
            numeric: Numeric tabular features (batch, 8)
            categorical: Categorical feature indices (batch, 6)
            binary: Binary features as 0/1 (batch, 6)
            weather: Weather data (batch, 26, 15)
            days_before_summit: Days before summit for each timestep (batch, 26)
            day_of_year: Day of year for each timestep (batch, 26)

        Returns:
            Success probability (batch,)
        """
        output = self.forward(
            numeric=numeric,
            categorical=categorical,
            binary=binary,
            weather=weather,
            days_before_summit=days_before_summit,
            day_of_year=day_of_year,
            return_attention=False,
        )
        return cast(torch.Tensor, output["probability"])

    def get_attention_weights(
        self,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
        binary: torch.Tensor,
        weather: torch.Tensor,
        days_before_summit: torch.Tensor,
        day_of_year: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Get attention weights from all layers for interpretability.

        Useful for understanding which tokens (features/timesteps) the model
        attends to when making predictions.

        Args:
            numeric: Numeric tabular features (batch, 8)
            categorical: Categorical feature indices (batch, 6)
            binary: Binary features as 0/1 (batch, 6)
            weather: Weather data (batch, 26, 15)
            days_before_summit: Days before summit for each timestep (batch, 26)
            day_of_year: Day of year for each timestep (batch, 26)

        Returns:
            List of attention weight tensors, one per layer.
            Each tensor has shape (batch, num_heads, seq_len, seq_len).
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(
                numeric=numeric,
                categorical=categorical,
                binary=binary,
                weather=weather,
                days_before_summit=days_before_summit,
                day_of_year=day_of_year,
                return_attention=True,
            )
        attention_weights = output["attention_weights"]
        if attention_weights is None:
            return []
        return cast(list[torch.Tensor], attention_weights)

    def count_parameters(self) -> dict[str, int]:
        """Count parameters by component for analysis.

        Returns:
            Dictionary with parameter counts per component and total.
        """
        counts = {
            "cls_token": self.cls_token.numel(),
            "modality_embedding": sum(
                p.numel() for p in self.modality_embedding.parameters()
            ),
            "tabular_tokenizer": sum(
                p.numel() for p in self.tabular_tokenizer.parameters()
            ),
            "weather_tokenizer": sum(
                p.numel() for p in self.weather_tokenizer.parameters()
            ),
            "encoder": sum(p.numel() for p in self.encoder.parameters()),
            "head": sum(p.numel() for p in self.head.parameters()),
        }
        counts["total"] = sum(counts.values())
        return counts
