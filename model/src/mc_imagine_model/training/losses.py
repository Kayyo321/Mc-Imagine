"""
Custom loss functions for Mc-Imagine model training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from mc_imagine_model.spec_constants import HEIGHT_LOSS_SCALE, SLOPE_LOSS_SCALE


class TerrainLoss(nn.Module):
    """
    Combines heightmap regression, heightmap *slope* regression, column-profile classification, and
    water-level regression — supervising the three tensors `TerrainHead` (model/heads.py) actually
    predicts for this tier. Per docs/poc-plan.md: "TerrainLoss = MSE(heightmap) +
    CrossEntropy(profile)"; the water-level MSE term is added on top of that literal definition since
    `TerrainHead` also predicts a per-column water level and it needs real supervision to be useful
    (rather than training on nothing), at a modest weight so it doesn't dominate the heightmap
    signal. The slope term is the docs/phase2-plan.md Phase 2 addition described below.

    Expects `preds` to contain "heightmap" [B,16,16], "profile_logits" [B,num_profiles,16,16],
    "water_level" [B,16,16]; `targets` to contain "heightmap" [B,16,16] float, "profile_id"
    [B,16,16] long, "water_level" [B] float (broadcast across all 16x16 columns — see
    data/dataset.py, which stores one water_level scalar per macro-region).
    """

    # Raw height MSE is O(10^2-10^3) in block units while cross-entropy over 8-12 classes is O(1-2),
    # so the terms must be normalized before they are summed. The scale matters enormously and is
    # easy to get wrong in a way that looks principled: normalizing by the head's own tanh amplitude
    # (192) was measured to leave terrain contributing 0.052 against biome's 2.48 — a 48x imbalance
    # under which training optimizes classification and ignores terrain shape entirely.
    #
    # HEIGHT_SCALE is therefore a typical terrain *deviation*, and SLOPE_SCALE a typical per-block
    # *step* (~1 block by definition) — normalizing a block-to-block gradient by 192 shrank the slope
    # term to ~2.7e-5, i.e. the relief term this class exists for was doing nothing. Both come from
    # spec_constants, where the measurements behind them are recorded.
    HEIGHT_SCALE = HEIGHT_LOSS_SCALE
    SLOPE_SCALE = SLOPE_LOSS_SCALE

    def __init__(self, water_weight: float = 0.25, slope_weight: float = 1.0) -> None:
        """
        Args:
            water_weight: weight of the per-column water-level MSE.
            slope_weight: weight of the finite-difference slope MSE, relative to the (identically
                normalized) absolute-height MSE. 1.0 means "care about getting the shape right
                exactly as much as getting the elevation right".
        """
        super().__init__()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.water_weight = water_weight
        self.slope_weight = slope_weight

    @staticmethod
    def _slopes(height: torch.Tensor) -> tuple:
        """First-order finite differences of a [B,16,16] `[z][x]` heightmap along x and along z.

        Returns `(d_x [B,16,15], d_z [B,15,16])`. Deliberately plain forward differences rather than
        a Sobel/conv filter: no padding is involved, so this introduces no border artifacts and
        nothing has to be excluded from the mean.
        """
        d_x = height[:, :, 1:] - height[:, :, :-1]   # last dim is X
        d_z = height[:, 1:, :] - height[:, :-1, :]   # middle dim is Z
        return d_x, d_z

    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        pred_h = preds["heightmap"] / self.HEIGHT_SCALE
        target_h = targets["heightmap"] / self.HEIGHT_SCALE

        height_loss = self.mse(pred_h, target_h)

        # --- slope / gradient term (docs/phase2-plan.md Phase 2) -------------------------------
        # Absolute-height MSE alone is exactly what let the Day-1 model win by being *smooth*: a
        # flat surface at the correct conditional mean height is a strong local optimum for MSE,
        # and measured output std was 0.02-3.73 against ground-truth per-chunk std up to 8.75. A
        # flat prediction can be near-optimal on height while being *maximally* wrong on slope, so
        # supervising the finite differences directly prices in relief that absolute error barely
        # notices, and closes off the smooth-but-correct-mean solution.
        #
        # Slopes are normalized by SLOPE_SCALE (~1 block per block), NOT by HEIGHT_SCALE: a
        # block-to-block step is a fundamentally different quantity from an absolute elevation, and
        # measuring this term under the old height normalization put it at ~2.7e-5 — three orders of
        # magnitude below the cross-entropies, so the relief supervision was inert. Computed on raw
        # block units so `slope_weight` is a meaningful knob.
        pred_dx, pred_dz = self._slopes(preds["heightmap"] / self.SLOPE_SCALE)
        target_dx, target_dz = self._slopes(targets["heightmap"] / self.SLOPE_SCALE)
        slope_loss = 0.5 * (self.mse(pred_dx, target_dx) + self.mse(pred_dz, target_dz))

        profile_loss = self.ce(preds["profile_logits"], targets["profile_id"])
        target_water = targets["water_level"].view(-1, 1, 1).expand_as(preds["water_level"])
        water_loss = self.mse(preds["water_level"] / self.HEIGHT_SCALE, target_water / self.HEIGHT_SCALE)
        return (
            height_loss
            + self.slope_weight * slope_loss
            + profile_loss
            + self.water_weight * water_loss
        )


class BiomeLoss(nn.Module):
    """
    Cross-entropy loss for biome prediction, over the quarter-resolution `biome_grid` classes.
    Expects `preds["biome_logits"]` [B,num_biomes,4,4] and `targets["biome_grid"]` [B,4,4] long.
    """

    def __init__(self) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.ce(preds["biome_logits"], targets["biome_grid"])


class StructureLoss(nn.Module):
    """
    Binary cross-entropy + localization loss for structure_markers (structure_support == "basic").

    Not used by this PoC (`imaginator-low_intensity-no_structures` declares
    `structure_support: "none"`) — left as an explicit stub; `CombinedLoss` below never calls this
    when `structure_weight == 0` (the default, and the only value this PoC trains with).
    """

    def __init__(self) -> None:
        super().__init__()
        # TODO (out of scope for this PoC): Initialize BCEWithLogitsLoss

    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        # TODO (out of scope for this PoC): Compute structure loss
        return torch.tensor(0.0, requires_grad=True)


class StructureGraphLoss(nn.Module):
    """
    Loss for structure_graph_nodes/edges (structure_support == "intricate").

    Node/edge ids are categorical (indices into the structure template library — see
    docs/model-spec.md "Structure Template Library"), so this is cross-entropy per node slot plus
    cross-entropy per edge cell, not a regression loss like TerrainLoss. Padding slots
    (room_type_id == 0) should be masked out rather than penalized, since max_rooms is a fixed upper
    bound and most structures won't use all of it.

    Not used by this PoC — see StructureLoss's docstring above.
    """

    def __init__(self) -> None:
        super().__init__()
        # TODO (out of scope for this PoC): Initialize masked CrossEntropyLoss for nodes and edges

    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        # TODO (out of scope for this PoC): Compute masked structure-graph loss
        return torch.tensor(0.0, requires_grad=True)


class MacroFieldLoss(nn.Module):
    """
    Loss for MacroFieldNet (model/macro_net.py), which trains separately from ImagineNet — it's a
    different model operating at macro-region granularity, not one of ImagineNet's heads, so this
    is NOT folded into CombinedLoss below. Only relevant for models with
    capabilities.requires_macro_field = true — not this PoC's model.

    Combines: MSE over region_heightfield (regression, like TerrainLoss), cross-entropy over
    flavor_zone_ids weighted by flavor_zone_weights (categorical + soft blend target), and binary
    cross-entropy over structure_candidate's has_structure flag plus a localization loss for its
    offset_x/offset_z — structurally similar to StructureLoss's marker localization, but at region
    scale instead of chunk scale.
    """

    def __init__(self) -> None:
        super().__init__()
        # TODO (out of scope for this PoC): Initialize MSELoss (heightfield), CrossEntropyLoss
        # (flavor zones), BCEWithLogitsLoss (structure candidate flag) + localization loss (offset)

    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        # TODO (out of scope for this PoC): Compute combined macro-field loss
        return torch.tensor(0.0, requires_grad=True)


class CombinedLoss(nn.Module):
    """
    Weighted sum of all loss components. `structure_weight`/`structure_graph_weight` default to
    `0.0` per docs/poc-plan.md ("structure_weight = structure_graph_weight = 0") since this PoC only
    trains `structure_support: "none"` models — when both are `0` (the only configuration this PoC
    ever uses), `StructureLoss`/`StructureGraphLoss` are never even invoked, since `preds` won't
    contain real `structure_markers`/`structure_graph_*` tensors to feed them (see
    `model/imagine_net.py`, which returns zeroed placeholders for those keys at this tier).
    """

    def __init__(
        self,
        terrain_weight: float = 1.0,
        biome_weight: float = 1.0,
        structure_weight: float = 0.0,
        structure_graph_weight: float = 0.0,
        slope_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.terrain_loss = TerrainLoss(slope_weight=slope_weight)
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
        total_loss = self.terrain_weight * loss_t + self.biome_weight * loss_b

        if self.structure_weight > 0.0:
            total_loss = total_loss + self.structure_weight * self.structure_loss(preds, targets)
        if self.structure_graph_weight > 0.0:
            total_loss = total_loss + self.structure_graph_weight * self.structure_graph_loss(preds, targets)
        return total_loss
