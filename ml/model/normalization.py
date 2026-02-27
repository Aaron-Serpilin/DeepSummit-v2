"""Normalization layers for the DeepSummit transformer.

This module implements RMSNorm (Root Mean Square Layer Normalization),
a simplified alternative to LayerNorm used in modern transformers like LLaMA.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    RMSNorm simplifies LayerNorm by removing mean centering:
    - LayerNorm: (x - mean) / std * gamma + beta
    - RMSNorm:   x / RMS(x) * gamma

    Benefits:
    - 10-15% faster than LayerNorm (skips mean computation)
    - Same or better performance in practice
    - Used by LLaMA, Mistral, and modern LLMs

    Args:
        dim: The dimension to normalize over (typically hidden_size)
        eps: Small constant for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Normalized tensor of same shape
        """
        # Compute RMS: sqrt(mean(x^2))
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return (x / rms) * self.weight
