"""Tokenization layers for converting features into transformer tokens.

This module implements FT-Transformer style per-feature tokenization for tabular
data and a weather tokenizer for multi-scale temporal weather data.
"""

import torch
import torch.nn as nn

from ml.model.embeddings import Time2Vec


class TabularTokenizer(nn.Module):
    """Tokenize tabular features (expedition metadata).

    Converts expedition features into token embeddings using per-feature tokenization:
    - Numeric features (8): Each gets its own Linear(1, hidden_size)
    - Categorical features (6): Each gets its own Embedding(vocab_size, hidden_size)
    - Binary features (6): Each gets its own Embedding(2, hidden_size)

    Total output: 20 tokens per expedition.

    Args:
        hidden_size: Token embedding dimension
        numeric_features: Number of numeric features
        categorical_vocab_sizes: List of vocabulary sizes for each categorical feature
        binary_features: Number of binary features
    """

    def __init__(
        self,
        hidden_size: int,
        numeric_features: int = 8,
        categorical_vocab_sizes: list[int] | None = None,
        binary_features: int = 6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.numeric_features = numeric_features
        self.binary_features = binary_features

        # Default categorical vocab sizes for DeepSummit features
        if categorical_vocab_sizes is None:
            categorical_vocab_sizes = [
                151,  # nationality (~150 + UNK)
                101,  # route1 (~100 + UNK)
                21,   # himal (~20 + UNK)
                5,    # season (4 + UNK)
                4,    # style (3 + UNK)
                15,   # peakid (14 + UNK)
            ]
        self.categorical_vocab_sizes = categorical_vocab_sizes

        # Numeric tokenizers: one Linear(1, H) per feature
        self.numeric_tokenizers = nn.ModuleList([
            nn.Linear(1, hidden_size) for _ in range(numeric_features)
        ])

        # Categorical embeddings: one Embedding(vocab, H) per feature
        # Index 0 reserved for [UNK] token
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, hidden_size)
            for vocab_size in categorical_vocab_sizes
        ])

        # Binary embeddings: one Embedding(2, H) per feature
        self.binary_embeddings = nn.ModuleList([
            nn.Embedding(2, hidden_size) for _ in range(binary_features)
        ])

    def forward(
        self,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
        binary: torch.Tensor,
    ) -> torch.Tensor:
        """Tokenize tabular features.

        Args:
            numeric: Numeric features of shape (batch, 8)
            categorical: Categorical feature indices of shape (batch, 6)
            binary: Binary features of shape (batch, 6)

        Returns:
            Token embeddings of shape (batch, 20, hidden_size)
        """
        tokens: list[torch.Tensor] = []

        # Tokenize numeric features
        for i, tokenizer in enumerate(self.numeric_tokenizers):
            # Extract single feature and add feature dimension
            feature = numeric[:, i : i + 1]  # (batch, 1)
            token = tokenizer(feature)  # (batch, hidden_size)
            tokens.append(token)

        # Tokenize categorical features
        for i, embedding in enumerate(self.categorical_embeddings):
            feature = categorical[:, i]  # (batch,)
            token = embedding(feature)  # (batch, hidden_size)
            tokens.append(token)

        # Tokenize binary features
        for i, embedding in enumerate(self.binary_embeddings):
            feature = binary[:, i]  # (batch,)
            token = embedding(feature)  # (batch, hidden_size)
            tokens.append(token)

        # Stack all tokens: (batch, 20, hidden_size)
        tokens_stacked: torch.Tensor = torch.stack(tokens, dim=1)

        return tokens_stacked


class WeatherTokenizer(nn.Module):
    """Tokenize multi-scale weather time series.

    Converts weather data from multiple temporal scales into token embeddings:
    - 7-day scale: 7 timesteps (daily resolution)
    - 30-day scale: 10 timesteps (3-day aggregates)
    - 90-day scale: 9 timesteps (10-day aggregates)

    Each timestep has 15 weather variables (temperature, wind, precipitation, etc.)
    that are projected to hidden_size and combined with Time2Vec positional encoding.

    Total output: 26 tokens per expedition (7 + 10 + 9).

    Args:
        hidden_size: Token embedding dimension
        weather_vars: Number of weather variables per timestep (default: 15)
        time2vec_features: Number of Time2Vec features per temporal dimension
    """

    def __init__(
        self,
        hidden_size: int,
        weather_vars: int = 15,
        time2vec_features: int = 32,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.weather_vars = weather_vars

        # Single shared projection for all weather timesteps
        self.weather_proj = nn.Linear(weather_vars, hidden_size)

        # Time2Vec encoders for two temporal dimensions
        # These will be concatenated, so each gets half the features
        self.time2vec_days_before = Time2Vec(time2vec_features)
        self.time2vec_day_of_year = Time2Vec(time2vec_features)

        # The total temporal encoding dim is 2 * time2vec_features
        temporal_dim = 2 * time2vec_features

        # Project temporal encoding to hidden_size for adding to weather tokens
        self.temporal_proj = nn.Linear(temporal_dim, hidden_size)

    def forward(
        self,
        weather: torch.Tensor,
        days_before_summit: torch.Tensor,
        day_of_year: torch.Tensor,
    ) -> torch.Tensor:
        """Tokenize weather time series.

        Args:
            weather: Weather data of shape (batch, 26, 15)
                26 timesteps: 7 (daily) + 10 (3-day) + 9 (10-day)
                15 variables per timestep
            days_before_summit: Days before summit for each timestep (batch, 26)
            day_of_year: Day of year for each timestep (batch, 26)

        Returns:
            Token embeddings of shape (batch, 26, hidden_size)
        """
        batch_size, num_timesteps, _ = weather.shape

        # Project weather variables to hidden_size
        weather_tokens = self.weather_proj(weather)  # (batch, 26, hidden_size)

        # Encode temporal positions with Time2Vec
        # days_before_summit: (batch, 26) -> (batch, 26, time2vec_features)
        time_enc_before = self.time2vec_days_before(days_before_summit)

        # day_of_year: (batch, 26) -> (batch, 26, time2vec_features)
        time_enc_year = self.time2vec_day_of_year(day_of_year)

        # Concatenate both temporal encodings
        temporal_encoding = torch.cat([time_enc_before, time_enc_year], dim=-1)
        # (batch, 26, 2 * time2vec_features)

        # Project temporal encoding to hidden_size
        temporal_tokens = self.temporal_proj(temporal_encoding)
        # (batch, 26, hidden_size)

        # Add temporal encoding to weather tokens
        tokens: torch.Tensor = weather_tokens + temporal_tokens

        return tokens
