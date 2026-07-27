"""
Coordinate/positional encoding module.
"""

import torch
import torch.nn as nn


class CoordinateEncoder(nn.Module):
    """
    Encodes chunk coordinates into continuous positional embeddings.
    """

    def __init__(self, embed_dim: int) -> None:
        """
        Args:
            embed_dim (int): Dimensionality of the resulting embedding.
        """
        super().__init__()
        self.embed_dim = embed_dim
        # TODO: Implement coordinate encoding network

    def forward(self, chunk_x: torch.Tensor, chunk_z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            chunk_x (torch.Tensor): X coordinates.
            chunk_z (torch.Tensor): Z coordinates.
            
        Returns:
            torch.Tensor: Positional embeddings.
        """
        # TODO: Implement forward pass
        return torch.zeros(chunk_x.size(0), self.embed_dim)


class SeedEncoder(nn.Module):
    """
    Encodes a world seed into a latent noise vector.
    """

    def __init__(self, embed_dim: int) -> None:
        """
        Args:
            embed_dim (int): Dimensionality of the resulting embedding.
        """
        super().__init__()
        self.embed_dim = embed_dim
        # TODO: Implement seed encoding logic (e.g., hash to vector)

    def forward(self, seed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seed (torch.Tensor): World seeds.
            
        Returns:
            torch.Tensor: Seed noise vectors.
        """
        # TODO: Implement forward pass
        return torch.zeros(seed.size(0), self.embed_dim)
