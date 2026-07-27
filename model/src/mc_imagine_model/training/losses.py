"""
Custom loss functions for Mc-Imagine model training.
"""

import torch
import torch.nn as nn
from typing import Dict


class TerrainLoss(nn.Module):
    """
    Combines heightmap MSE and block volume cross-entropy.
    """

    def __init__(self) -> None:
        super().__init__()
        # TODO: Initialize MSELoss and CrossEntropyLoss

    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        # TODO: Compute combined terrain loss
        return torch.tensor(0.0, requires_grad=True)


class BiomeLoss(nn.Module):
    """
    Cross-entropy loss for biome prediction.
    """

    def __init__(self) -> None:
        super().__init__()
        # TODO: Initialize CrossEntropyLoss

    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        # TODO: Compute biome loss
        return torch.tensor(0.0, requires_grad=True)


class StructureLoss(nn.Module):
    """
    Binary cross-entropy + localization loss for structure placement.
    """

    def __init__(self) -> None:
        super().__init__()
        # TODO: Initialize BCEWithLogitsLoss

    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        # TODO: Compute structure loss
        return torch.tensor(0.0, requires_grad=True)


class CombinedLoss(nn.Module):
    """
    Weighted sum of all loss components.
    """

    def __init__(self, terrain_weight: float = 1.0, biome_weight: float = 1.0, structure_weight: float = 1.0) -> None:
        super().__init__()
        self.terrain_loss = TerrainLoss()
        self.biome_loss = BiomeLoss()
        self.structure_loss = StructureLoss()
        
        self.terrain_weight = terrain_weight
        self.biome_weight = biome_weight
        self.structure_weight = structure_weight

    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss_t = self.terrain_loss(preds, targets)
        loss_b = self.biome_loss(preds, targets)
        loss_s = self.structure_loss(preds, targets)
        
        total_loss = (
            self.terrain_weight * loss_t +
            self.biome_weight * loss_b +
            self.structure_weight * loss_s
        )
        return total_loss
