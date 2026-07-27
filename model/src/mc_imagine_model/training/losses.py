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
    Binary cross-entropy + localization loss for structure_markers (structure_support == "basic").
    """

    def __init__(self) -> None:
        super().__init__()
        # TODO: Initialize BCEWithLogitsLoss

    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        # TODO: Compute structure loss
        return torch.tensor(0.0, requires_grad=True)


class StructureGraphLoss(nn.Module):
    """
    Loss for structure_graph_nodes/edges (structure_support == "intricate").

    Node/edge ids are categorical (indices into the structure template library — see
    docs/model-spec.md "Structure Template Library"), so this is cross-entropy per node slot plus
    cross-entropy per edge cell, not a regression loss like TerrainLoss. Padding slots
    (room_type_id == 0) should be masked out rather than penalized, since max_rooms is a fixed upper
    bound and most structures won't use all of it.
    """

    def __init__(self) -> None:
        super().__init__()
        # TODO: Initialize masked CrossEntropyLoss for nodes and edges

    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        # TODO: Compute masked structure-graph loss
        return torch.tensor(0.0, requires_grad=True)


class MacroFieldLoss(nn.Module):
    """
    Loss for MacroFieldNet (model/macro_net.py), which trains separately from ImagineNet — it's a
    different model operating at macro-region granularity, not one of ImagineNet's heads, so this
    is NOT folded into CombinedLoss below. Only relevant for models with
    capabilities.requires_macro_field = true.

    Combines: MSE over region_heightfield (regression, like TerrainLoss), cross-entropy over
    flavor_zone_ids weighted by flavor_zone_weights (categorical + soft blend target), and binary
    cross-entropy over structure_candidate's has_structure flag plus a localization loss for its
    offset_x/offset_z — structurally similar to StructureLoss's marker localization, but at region
    scale instead of chunk scale.
    """

    def __init__(self) -> None:
        super().__init__()
        # TODO: Initialize MSELoss (heightfield), CrossEntropyLoss (flavor zones),
        # BCEWithLogitsLoss (structure candidate flag) + localization loss (offset)

    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        # TODO: Compute combined macro-field loss
        return torch.tensor(0.0, requires_grad=True)


class CombinedLoss(nn.Module):
    """
    Weighted sum of all loss components. structure_graph_weight only contributes when training a
    structure_support == "intricate" model (see StructureGraphLoss); leave it at 0 for lower tiers.
    """

    def __init__(
        self,
        terrain_weight: float = 1.0,
        biome_weight: float = 1.0,
        structure_weight: float = 1.0,
        structure_graph_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.terrain_loss = TerrainLoss()
        self.biome_loss = BiomeLoss()
        self.structure_loss = StructureLoss()
        self.structure_graph_loss = StructureGraphLoss()

        self.terrain_weight = terrain_weight
        self.biome_weight = biome_weight
        self.structure_weight = structure_weight
        self.structure_graph_weight = structure_graph_weight

    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss_t = self.terrain_loss(preds, targets)
        loss_b = self.biome_loss(preds, targets)
        loss_s = self.structure_loss(preds, targets)
        loss_sg = self.structure_graph_loss(preds, targets)

        total_loss = (
            self.terrain_weight * loss_t +
            self.biome_weight * loss_b +
            self.structure_weight * loss_s +
            self.structure_graph_weight * loss_sg
        )
        return total_loss
