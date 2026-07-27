"""
Macro-region model definition.

Backs macro.onnx (see docs/model-spec.md "Runtime Architecture" / "Two-Model File Structure").
Only relevant for models declaring capabilities.requires_macro_field = true — the
low-intensity/no-structures reference model never trains or exports this graph.

Unlike ImagineNet (imagine_net.py), which runs once per chunk, MacroFieldNet runs once per
32x32-chunk macro-region (matching vanilla Minecraft's .mca region file boundary), lazily, the
first time any chunk inside that region is requested. Its job is establishing cross-chunk
coherence — large-scale landform continuity and spatially contiguous "flavor zone" style regions —
that no amount of independent per-chunk inference can produce on its own.
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

# TODO: Import actual modules once implemented
# from .text_encoder import PromptEncoder  # shared foundation with ImagineNet, see docs/model-spec.md "Foundation Architecture"


class MacroFieldNet(nn.Module):
    """
    MacroFieldNet architecture for generating per-macro-region context conditioned on text + seed.

    Shares its PromptEncoder foundation with ImagineNet (both should use the same frozen
    MiniLM-class text encoder) but is otherwise a much smaller/cheaper model than ImagineNet,
    since it only needs to produce a coarse landform + a handful of flavor-zone/structure-candidate
    values per region, not per-block detail.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Args:
            config (dict): Model configuration parameters, including max_flavor_zones (how many
                candidate zone slots region_heightfield's flavor_zone_ids/flavor_zone_weights
                outputs carry — see docs/model-spec.md, fixed at 4 in the current spec).
        """
        super().__init__()
        self.config = config

        # TODO: Initialize shared PromptEncoder (same checkpoint as ImagineNet's)
        # TODO: Region-coordinate positional encoding (analogous to CoordinateEncoder in positional.py,
        #       but at region scale rather than chunk scale)
        # TODO: SeedEncoder (shared with ImagineNet, or a separate instance — same "seed -> noise
        #       vector" contract either way, see SeedEncoder's docstring in positional.py)

        # TODO: region_heightfield head -> float32[33, 33]
        # TODO: flavor_zone head -> flavor_zone_ids int32[4], flavor_zone_weights float32[4]
        # TODO: structure_candidate head -> int32[3] (has_structure, offset_x, offset_z), gated by
        #       capabilities.structure_spacing_regions at inference/placement time (the seeded
        #       spacing check itself is NOT learned — see docs/model-spec.md "Structure Placement
        #       Determinism"; this head only needs to learn whether/where a structure fits
        #       narratively once placement has already been decided by seeded math)

    def forward(
        self, prompt_tokens: torch.Tensor, region_x: torch.Tensor, region_z: torch.Tensor, seed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            tuple:
                region_heightfield: float32[33, 33]
                flavor_zone_ids: int32[4]
                flavor_zone_weights: float32[4]
                structure_candidate: int32[3]
        """
        # TODO: Implement forward pass
        return torch.zeros(0), torch.zeros(0), torch.zeros(0), torch.zeros(0)
