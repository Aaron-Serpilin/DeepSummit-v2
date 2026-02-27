"""Feed-forward network implementations for the DeepSummit transformer.

This module implements SwiGLU, a gated activation function that outperforms
standard GELU in transformer architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network.

    SwiGLU uses a gating mechanism with Swish (SiLU) activation:
        SwiGLU(x) = (Swish(x·W₁) ⊙ (x·W₂)) · W₃

    This outperforms standard GELU FFN in transformer benchmarks and is used
    in PaLM, LLaMA, and other modern LLMs.

    Architecture:
        Input (H) → Gate projection (H → ffn_hidden)
                  ↓
                  Swish activation
                  ↓
                  Element-wise multiply with Value projection (H → ffn_hidden)
                  ↓
                  Output projection (ffn_hidden → H)

    Args:
        hidden_size: Input/output dimension
        ffn_hidden_size: Intermediate dimension (typically ~2.67 × hidden_size)
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Default expansion factor: 8/3 ≈ 2.67 (keeps param count similar to 4H GELU FFN)
        if ffn_hidden_size is None:
            ffn_hidden_size = int(hidden_size * 8 / 3)

        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size

        # Three linear projections for SwiGLU
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.value_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.output_proj = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation.

        Args:
            x: Input tensor of shape (..., hidden_size)

        Returns:
            Output tensor of shape (..., hidden_size)
        """
        # Gate path: apply Swish (SiLU) activation
        gate = F.silu(self.gate_proj(x))

        # Value path: no activation
        value = self.value_proj(x)

        # Element-wise gating
        hidden = gate * value

        # Output projection with dropout
        output = self.output_proj(hidden)
        output = self.dropout(output)

        return output
