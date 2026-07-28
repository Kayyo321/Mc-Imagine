"""
Output head modules.
"""

from typing import Tuple

import torch
import torch.nn as nn

from mc_imagine_model.spec_constants import (
    HEIGHT_AMPLITUDE,
    HEIGHT_CENTER,
    NUM_BIOMES,
    NUM_PROFILES,
)


class TerrainHead(nn.Module):
    """
    Output head for terrain generation. Per docs/poc-plan.md Phase 5, this predicts (from the
    decoder's final 16x16 feature map, `in_channels` deep):
      - heightmap: 1 channel, squashed to `HEIGHT_CENTER + HEIGHT_AMPLITUDE*tanh(x)`;
      - column-profile logits: `num_profiles` channels (8 by default — see spec_constants.py);
      - per-column water level: 1 channel, squashed identically since it is also a Y-coordinate.

    **Why the constants come from `spec_constants` and are never written literally here**
    (docs/phase2-plan.md §0/§1b): Day 1 hardcoded `63 + 96*tanh(x)`, a hard ceiling of 159, while
    `data/world_generator.py` rendered terrain up to 235 and 14% of regions exceeded 159. The head
    physically could not represent its own targets, so it pinned at the ceiling — where tanh's
    gradient is ~0 — and *every* local variation died with it: the "towering snow-capped peaks"
    prompt came out at height 158.84-158.98, std 0.03. Flat. The range is now
    `[HEIGHT_MIN, HEIGHT_MAX]` = [-96, 288], the generator clips into a strictly narrower band
    ([TERRAIN_CLIP_MIN, TERRAIN_CLIP_MAX]) so no target ever sits at the asymptote, and both facts
    are asserted at training startup. Sharing one definition is what keeps those three places from
    drifting apart again.

    `block_volume` is *not* predicted here — see `data/dataset.py`'s docstring and
    `export/export_onnx.py`: it's a deterministic `torch.where`-style expansion of
    (heightmap, profile, water_level) computed once, inside the exported graph.
    """

    def __init__(self, in_channels: int = 128, num_profiles: int = NUM_PROFILES) -> None:
        super().__init__()
        self.height_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.profile_conv = nn.Conv2d(in_channels, num_profiles, kernel_size=1)
        self.water_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features (torch.Tensor): [B, in_channels, 16, 16] decoder features.

        Returns:
            tuple: (heightmap [B,16,16] float, profile_logits [B,num_profiles,16,16] float,
                    water_level [B,16,16] float)
        """
        heightmap = HEIGHT_CENTER + HEIGHT_AMPLITUDE * torch.tanh(self.height_conv(features).squeeze(1))
        profile_logits = self.profile_conv(features)
        water_level = HEIGHT_CENTER + HEIGHT_AMPLITUDE * torch.tanh(self.water_conv(features).squeeze(1))
        return heightmap, profile_logits, water_level


class BiomeHead(nn.Module):
    """
    Output head for biome grid prediction. Maps the decoder's 16x16 feature map down to the
    4x4 quarter-resolution grid `biome_grid` needs (docs/model-spec.md: "one entry per 4x4x4
    volume") via a stride-4 4x4 conv — a learned downsample rather than pooling, so the head can
    weight which of the 16 columns in each 4x4 block matters most for that block's biome label.
    """

    def __init__(self, in_channels: int = 128, num_biomes: int = NUM_BIOMES) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_biomes, kernel_size=4, stride=4)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): [B, in_channels, 16, 16] decoder features.

        Returns:
            torch.Tensor: biome_logits, [B, num_biomes, 4, 4]
        """
        return self.conv(features)


class StructureHead(nn.Module):
    """
    Output head for structure_markers (capabilities.structure_support == "basic"). Scattered
    single-piece motifs — see docs/model-spec.md "Output Tensors".

    Explicit stub: `imaginator-low_intensity-no_structures` (the only model this PoC trains)
    declares `structure_support: "none"`, so this head is never instantiated or trained — see
    docs/poc-plan.md's scope ("StructureHead/StructureGraphHead as explicit stubs — do not
    implement them").
    """

    def __init__(self) -> None:
        super().__init__()
        # TODO (out of scope for this PoC): Implement structure head architecture

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: structure_markers, float32[N, 5] (type_id, x, y, z, probability)
        """
        # TODO (out of scope for this PoC): Implement forward pass
        return torch.zeros(0)


class StructureGraphHead(nn.Module):
    """
    Output head for structure_graph_nodes/edges/origin (capabilities.structure_support == "intricate").

    Unlike StructureHead's flat marker list, this composes a bounded room graph from a curated
    template library (mod/common/src/main/resources/structures/, per docs/model-spec.md "Structure
    Template Library") rather than predicting raw voxels — the model chooses which rooms, how they
    connect, and which pre-authored room/redstone template fills each role. This is the head that
    makes the "castles with hidden rooms and redstone escape-room puzzles" prompt tractable: it only
    ever needs to learn pacing/composition, never redstone circuit design.

    IMPORTANT: this head's raw output is *preference weights* over the template library (which
    room/connector templates the prompt makes likely), not a directly-emitted final graph. A free
    sequence decoder could otherwise propose a structurally invalid graph (a hidden door connecting
    to a room with no matching connector socket) with no way to guarantee otherwise. The mod-side
    (or a training-time reference implementation of) Wave Function Collapse consumes these
    preference weights and resolves the actual structure_graph_nodes/edges, guaranteeing every
    connection is socket-valid by construction — see docs/model-spec.md "Foundation Architecture"
    and PROJECT.md "High Intensity Detail & Intricate Structures — Full Mechanics". This module only
    needs to learn taste; WFC supplies correctness.

    Structurally closer to a small sequence/graph decoder (predicting a fixed room-graph grammar,
    max_rooms tokens long) than to TerrainHead/BiomeHead's dense regression.

    Explicit stub: not used by `imaginator-low_intensity-no_structures` (structure_support: "none") —
    see docs/poc-plan.md's scope.
    """

    def __init__(self, max_rooms: int = 12) -> None:
        super().__init__()
        self.max_rooms = max_rooms
        # TODO (out of scope for this PoC): Implement autoregressive/graph decoder architecture

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            tuple:
                structure_graph_nodes: int32[max_rooms, 4] (room_type_id, size_class, loot_tier, redstone_template_id)
                structure_graph_edges: int32[max_rooms, max_rooms] (0 = no connection, else connector_type_id)
                structure_origin: int32[3] (chunk-relative x, y, z anchor)
        """
        # TODO (out of scope for this PoC): Implement forward pass
        return torch.zeros(0), torch.zeros(0), torch.zeros(0)
