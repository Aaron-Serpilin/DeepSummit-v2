"""Multi-head attention implementation for the DeepSummit transformer.

This module implements standard multi-head self-attention with Flash Attention 2
support for memory-efficient computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with Flash Attention 2 support.

    Implements scaled dot-product attention across multiple heads:
        Attention(Q, K, V) = softmax(QK^T / √d_k) V

    Uses PyTorch's scaled_dot_product_attention which automatically enables
    Flash Attention 2 when available (PyTorch 2.2+).

    Args:
        hidden_size: Input/output dimension
        num_heads: Number of attention heads
        dropout: Dropout probability for attention weights
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply multi-head self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Optional mask of shape (batch, seq_len) or (batch, seq_len, seq_len)
                True values are masked (not attended to)

        Returns:
            Tuple of:
                - Output tensor of shape (batch, seq_len, hidden_size)
                - Attention weights of shape (batch, num_heads, seq_len, seq_len) or None
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, hidden_size)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention: (batch, seq_len, hidden) -> (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with Flash Attention 2 (automatic in PyTorch 2.2+)
        # LEARN: scaled_dot_product_attention automatically uses Flash Attention when:
        # - PyTorch >= 2.2
        # - CUDA is available
        # - Data types are fp16 or bf16
        # It provides 2-4x speedup and O(N) memory instead of O(N²) for attention matrix
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,  # We use self-attention, not causal (autoregressive) attention
        )
        # attn_output: (batch, num_heads, seq_len, head_dim)

        # Reshape back: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        # Output projection
        output = self.out_proj(attn_output)

        # For attention visualization, compute weights manually (not used in forward pass)
        # This only runs during eval when we need attention weights
        if not self.training:
            with torch.no_grad():
                scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
                if attention_mask is not None:
                    scores = scores.masked_fill(attention_mask, float('-inf'))
                attn_weights = F.softmax(scores, dim=-1)
        else:
            attn_weights = None

        return output, attn_weights
