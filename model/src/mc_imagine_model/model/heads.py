"""
Output head modules.
"""

import torch
import torch.nn as nn


class TerrainHead(nn.Module):
    """
    Output head for terrain generation (heightmap + block volume).
    """

    def __init__(self) -> None:
        super().__init__()
        # TODO: Implement terrain head architecture

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            tuple: (heightmap, block_volume)
        """
        # TODO: Implement forward pass
        return torch.zeros(0), torch.zeros(0)


class BiomeHead(nn.Module):
    """
    Output head for biome grid prediction.
    """

    def __init__(self) -> None:
        super().__init__()
        # TODO: Implement biome head architecture

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: biome_grid
        """
        # TODO: Implement forward pass
        return torch.zeros(0)


class StructureHead(nn.Module):
    """
    Output head for structure_markers (capabilities.structure_support == "basic").
    Scattered single-piece motifs — see docs/model-spec.md "Output Tensors".
    """

    def __init__(self) -> None:
        super().__init__()
        # TODO: Implement structure head architecture

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: structure_markers, float32[N, 5] (type_id, x, y, z, probability)
        """
        # TODO: Implement forward pass
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
    """

    def __init__(self, max_rooms: int = 12) -> None:
        super().__init__()
        self.max_rooms = max_rooms
        # TODO: Implement autoregressive/graph decoder architecture

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            tuple:
                structure_graph_nodes: int32[max_rooms, 4] (room_type_id, size_class, loot_tier, redstone_template_id)
                structure_graph_edges: int32[max_rooms, max_rooms] (0 = no connection, else connector_type_id)
                structure_origin: int32[3] (chunk-relative x, y, z anchor)
        """
        # TODO: Implement forward pass
        return torch.zeros(0), torch.zeros(0), torch.zeros(0)
