"""Classification head for the DeepSummit transformer.

This module implements the final classification layer that converts the
transformer's [CLS] token representation into a summit success probability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """Classification head for binary prediction.

    Extracts the [CLS] token embedding and projects it to a probability:
        cls_token → Linear(H, H) → GELU → Dropout → Linear(H, 1) → Sigmoid

    The two-layer design with non-linearity allows the head to learn
    a more complex decision boundary than a simple linear classifier.

    Args:
        hidden_size: Input dimension (transformer hidden size)
        dropout: Dropout probability between layers
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        # Two-layer MLP with GELU activation
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """Compute summit success probability from encoder output.

        Args:
            encoder_output: Transformer output of shape (batch, seq_len, hidden_size)
                The [CLS] token is expected at position 0.

        Returns:
            Probability tensor of shape (batch,) in range [0, 1]
        """
        # Extract [CLS] token (position 0)
        cls_token = encoder_output[:, 0, :]  # (batch, hidden_size)

        # Two-layer classification head
        x = self.dense(cls_token)  # (batch, hidden_size)
        x = F.gelu(x)
        x = self.dropout(x)
        logits = self.classifier(x)  # (batch, 1)

        # Convert logits to probability
        probability = torch.sigmoid(logits).squeeze(-1)  # (batch,)

        return probability

    def forward_with_logits(self, encoder_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute both logits and probability (useful for loss computation).

        Args:
            encoder_output: Transformer output of shape (batch, seq_len, hidden_size)

        Returns:
            Tuple of:
                - logits: Raw logits of shape (batch,)
                - probability: Sigmoid probability of shape (batch,)
        """
        cls_token = encoder_output[:, 0, :]

        x = self.dense(cls_token)
        x = F.gelu(x)
        x = self.dropout(x)
        logits = self.classifier(x).squeeze(-1)  # (batch,)

        probability = torch.sigmoid(logits)

        return logits, probability
