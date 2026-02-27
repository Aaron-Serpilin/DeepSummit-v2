"""Regularization layers for the DeepSummit transformer.

This module implements DropPath (Stochastic Depth), which randomly drops
entire residual branches during training to improve generalization.
"""

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.

    Randomly drops entire residual branches during training with probability `drop_prob`.
    During inference, no dropping occurs.

    This regularization technique helps prevent overfitting in deep transformers by:
    - Creating an implicit ensemble of shallower networks during training
    - Encouraging earlier layers to learn robust representations

    Typically used with a linear schedule: drop probability increases with layer depth.

    Args:
        drop_prob: Probability of dropping the branch (0.0 = no drop, 1.0 = always drop)
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DropPath.

        Args:
            x: Input tensor from residual branch

        Returns:
            - During training: x with probability (1 - drop_prob), else 0
            - During inference: x (no dropping)
        """
        # No drop during eval or if drop_prob is 0
        if not self.training or self.drop_prob == 0.0:
            return x

        # Compute keep probability
        keep_prob = 1.0 - self.drop_prob

        # Generate random tensor with shape (batch_size, 1, 1, ...) to broadcast
        # This ensures entire samples are dropped, not individual features
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = (random_tensor < keep_prob).float()

        # Scale by keep_prob to maintain expected value
        # Without scaling: E[output] = keep_prob * x
        # With scaling: E[output] = x (maintains same scale as no dropout)
        output = x * random_tensor / keep_prob

        return output

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"drop_prob={self.drop_prob}"


def create_drop_path_schedule(
    drop_path_rate: float, num_layers: int
) -> list[float]:
    """Create linear drop path schedule for transformer layers.

    Drop probability increases linearly with layer depth:
        layer_0: drop_prob = 0.0
        layer_i: drop_prob = drop_path_rate × (i / (num_layers - 1))
        layer_N: drop_prob = drop_path_rate

    This schedule regularizes deeper layers more heavily while preserving
    early layer representations.

    Args:
        drop_path_rate: Maximum drop probability (for final layer)
        num_layers: Total number of transformer layers

    Returns:
        List of drop probabilities, one per layer

    Example:
        >>> create_drop_path_schedule(drop_path_rate=0.1, num_layers=6)
        [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
    """
    if num_layers == 1:
        return [drop_path_rate]

    return [drop_path_rate * i / (num_layers - 1) for i in range(num_layers)]
