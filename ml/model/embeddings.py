"""Positional encoding implementations for the DeepSummit transformer.

This module implements Time2Vec, a learnable time representation that captures
both periodic and non-periodic patterns in temporal data.
"""

import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    """Time2Vec: Learning a Vector Representation of Time.

    Time2Vec learns a time representation that captures both linear trends
    and periodic patterns:

        Time2Vec(t) = [ω₀·t + φ₀, sin(ω₁·t + φ₁), ..., sin(ωₖ·t + φₖ)]
                      └─────────┘ └──────────────────────────────────┘
                       linear          k periodic components
                      (trend)          (cyclical patterns)

    Advantages over fixed sinusoidal encoding:
    - Learnable frequencies adapt to task-specific temporal patterns
    - Linear component captures non-periodic trends
    - More expressive for irregular temporal patterns

    In DeepSummit, we encode two temporal dimensions:
    - days_before_summit: Relative position (0-89 days)
    - day_of_year: Absolute calendar position (1-365 days)

    Args:
        num_features: Number of output features (should divide evenly into hidden_size)
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.num_features = num_features

        # Linear component: ω₀·t + φ₀
        self.linear_weight = nn.Parameter(torch.randn(1))
        self.linear_bias = nn.Parameter(torch.randn(1))

        # Periodic components: sin(ωᵢ·t + φᵢ) for i=1...(num_features-1)
        num_periodic = num_features - 1
        self.periodic_weights = nn.Parameter(torch.randn(num_periodic))
        self.periodic_biases = nn.Parameter(torch.randn(num_periodic))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode time values.

        Args:
            t: Time values of shape (...,) representing temporal positions

        Returns:
            Time embeddings of shape (..., num_features)
        """
        # Ensure t has the right shape for broadcasting
        if t.dim() == 0:
            t = t.unsqueeze(0)

        # Linear component
        linear = self.linear_weight * t + self.linear_bias
        linear = linear.unsqueeze(-1)  # (..., 1)

        # Periodic components
        # t: (...,) -> (..., 1) for broadcasting
        t_expanded = t.unsqueeze(-1)
        periodic = torch.sin(self.periodic_weights * t_expanded + self.periodic_biases)
        # periodic: (..., num_periodic)

        # Concatenate linear and periodic components
        encoding = torch.cat([linear, periodic], dim=-1)
        # encoding: (..., num_features)

        return encoding


class ModalityEmbedding(nn.Module):
    """Modality type embeddings for distinguishing token types.

    Each token gets a modality-specific embedding added to it, allowing the
    transformer to understand which "data source" each token comes from.

    Modality types:
        0: Special tokens ([CLS], [SEP])
        1: Tabular features (expedition metadata)
        2: 7-day weather (tactical summit conditions)
        3: 30-day weather (acclimatization period)
        4: 90-day weather (seasonal context)

    Args:
        num_modalities: Number of distinct modality types (default: 5)
        hidden_size: Embedding dimension
    """

    def __init__(self, num_modalities: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_modalities, hidden_size)

    def forward(self, modality_ids: torch.Tensor) -> torch.Tensor:
        """Get modality embeddings.

        Args:
            modality_ids: Modality type indices of shape (batch, seq_len)

        Returns:
            Modality embeddings of shape (batch, seq_len, hidden_size)
        """
        embeddings: torch.Tensor = self.embedding(modality_ids)
        return embeddings
