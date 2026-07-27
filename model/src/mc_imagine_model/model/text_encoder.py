"""
Lightweight text/prompt encoder.
"""

import torch
import torch.nn as nn


class PromptEncoder(nn.Module):
    """
    Encodes text prompt tokens into continuous embeddings.
    """

    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int) -> None:
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # TODO: Implement embedding layer and transformer encoder

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens (torch.Tensor): Tensor of prompt tokens.
            
        Returns:
            torch.Tensor: Encoded text embeddings.
        """
        # TODO: Implement forward pass
        return torch.zeros(tokens.size(0), self.embed_dim)
