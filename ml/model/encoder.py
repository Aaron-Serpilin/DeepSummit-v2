"""Transformer encoder implementation for the DeepSummit model.

This module implements the transformer encoder stack with pre-normalization,
SwiGLU FFN, and stochastic depth regularization.
"""

import torch
import torch.nn as nn

from ml.model.attention import MultiHeadAttention
from ml.model.ffn import SwiGLU
from ml.model.normalization import RMSNorm
from ml.model.regularization import DropPath, create_drop_path_schedule


class TransformerBlock(nn.Module):
    """Single transformer encoder block with pre-normalization.

    Architecture (Pre-RMSNorm):
        x = x + DropPath(Attention(RMSNorm(x)))
        x = x + DropPath(FFN(RMSNorm(x)))

    Pre-normalization improves training stability compared to post-norm.

    Args:
        hidden_size: Token embedding dimension
        num_heads: Number of attention heads
        ffn_hidden_size: FFN intermediate dimension (default: ~2.67 × hidden_size)
        dropout: Dropout probability
        drop_path_prob: DropPath probability for this block
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        ffn_hidden_size: int | None = None,
        dropout: float = 0.1,
        drop_path_prob: float = 0.0,
    ) -> None:
        super().__init__()

        # Pre-normalization layers
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)

        # Multi-head self-attention
        self.attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
        )

        # SwiGLU feed-forward network
        self.ffn = SwiGLU(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            dropout=dropout,
        )

        # Stochastic depth (DropPath)
        self.drop_path1 = DropPath(drop_path_prob)
        self.drop_path2 = DropPath(drop_path_prob)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply transformer block.

        Args:
            x: Input tokens of shape (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask

        Returns:
            Tuple of:
                - Output tokens of shape (batch, seq_len, hidden_size)
                - Attention weights (only during eval)
        """
        # Multi-head self-attention with pre-norm and residual
        residual = x
        x = self.norm1(x)
        attn_output, attn_weights = self.attention(x, attention_mask)
        x = residual + self.drop_path1(attn_output)

        # Feed-forward network with pre-norm and residual
        residual = x
        x = self.norm2(x)
        ffn_output = self.ffn(x)
        x = residual + self.drop_path2(ffn_output)

        return x, attn_weights


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder blocks.

    Implements the full transformer encoder with:
    - Multiple transformer blocks with increasing drop path probability
    - Final RMSNorm after all blocks
    - Optional attention weight collection for visualization

    Args:
        hidden_size: Token embedding dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads per block
        ffn_hidden_size: FFN intermediate dimension
        dropout: Dropout probability
        drop_path_rate: Maximum drop path probability (for final layer)
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_hidden_size: int | None = None,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Create linear drop path schedule
        drop_path_probs = create_drop_path_schedule(drop_path_rate, num_layers)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                ffn_hidden_size=ffn_hidden_size,
                dropout=dropout,
                drop_path_prob=drop_path_probs[i],
            )
            for i in range(num_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Apply transformer encoder.

        Args:
            x: Input tokens of shape (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask
            return_attention: If True, collect and return attention weights from all layers

        Returns:
            Tuple of:
                - Encoded tokens of shape (batch, seq_len, hidden_size)
                - List of attention weights per layer (if return_attention=True)
        """
        attention_weights: list[torch.Tensor] | None = [] if return_attention else None

        # Apply each transformer block
        for block in self.blocks:
            x, attn = block(x, attention_mask)
            if return_attention and attn is not None and attention_weights is not None:
                attention_weights.append(attn)

        # Final normalization
        x = self.norm(x)

        return x, attention_weights
