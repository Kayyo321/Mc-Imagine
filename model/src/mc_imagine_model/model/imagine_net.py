"""
Core model definition.

This module defines the main ImagineNet architecture, which is a conditional generation model for
Minecraft chunks. Backs model.onnx (see docs/model-spec.md "Input Tensors: model.onnx" / "Output
Tensors: model.onnx") — the per-chunk graph every model ships, regardless of tier. See
model/macro_net.py for the separate, per-macro-region MacroFieldNet that models declaring
capabilities.requires_macro_field = true also ship — not used by
`imaginator-low_intensity-no_structures`, the only model this PoC trains.

Architecture (docs/poc-plan.md Phase 5):
  1. `PromptEncoder` (frozen MiniLM, text_encoder.py) -> 384-dim prompt embedding.
  2. `SeedEncoder` (positional.py) -> 64-dim seed embedding.
  3. concat(prompt, seed) -> small MLP -> 128-dim fused conditioning vector.
  4. `CoordinateEncoder` (positional.py) -> 36-channel Fourier feature grid, (16+2*halo)^2 spatial.
  5. Broadcast the fused conditioning vector across that grid, concat channel-wise, and run it
     through 4 valid-padded (no zero-padding) 3x3 conv + GELU layers at 128 channels — each valid
     conv shrinks the spatial size by 2, so 4 of them take 24x24 -> 16x16 exactly.
  6. `TerrainHead` / `BiomeHead` (heads.py) consume the final 16x16, 128-channel feature map.

The valid-padding-only conv stack is what makes docs/poc-plan.md §1b's seamless-border guarantee
hold: since every value in the 24x24 input grid is a pure function of global block coordinate (see
CoordinateEncoder), and no zero-padding is ever introduced, every value in the 16x16 output is
*also* a pure function of global coordinate — independent of which chunk requested it.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from mc_imagine_model.model.heads import BiomeHead, TerrainHead
from mc_imagine_model.model.positional import CoordinateEncoder, SeedEncoder
from mc_imagine_model.model.text_encoder import PromptEncoder
from mc_imagine_model.spec_constants import NUM_BIOMES, NUM_PROFILES

HALO = 4
PATCH = 16
CONV_CHANNELS = 128
FUSION_HIDDEN = 256
FUSION_OUT = 128
NUM_CONV_LAYERS = 4  # 24x24 -> 16x16 via 4 valid 3x3 convs (each shrinks by 2)


class ImagineNet(nn.Module):
    """
    ImagineNet architecture for generating Minecraft world chunks conditioned on text.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the ImagineNet model.

        Args:
            config (dict): Model configuration parameters, including which capability tier this
                instance targets (config["intensity"], config["structure_support"],
                config["requires_macro_field"]) so __init__ knows which optional heads/inputs to
                wire up — e.g. StructureGraphHead only for structure_support == "intricate". This
                PoC only ever configures/trains `intensity: "low"`, `structure_support: "none"`,
                `requires_macro_field: false` — the higher tiers' heads are stubs (heads.py) never
                instantiated here.
        """
        super().__init__()
        self.config = config

        checkpoint_dir = config.get("checkpoint_dir")
        self.text_encoder = PromptEncoder(checkpoint_dir=checkpoint_dir)
        self.coord_encoder = CoordinateEncoder(halo=config.get("halo", HALO), patch=config.get("patch", PATCH))
        self.seed_encoder = SeedEncoder(
            num_freqs=config.get("seed_freqs", 32), embed_dim=config.get("seed_embed_dim", 64)
        )

        prompt_dim = self.text_encoder.embed_dim
        seed_dim = self.seed_encoder.embed_dim
        fusion_hidden = config.get("fusion_hidden", FUSION_HIDDEN)
        fusion_out = config.get("fusion_out", FUSION_OUT)
        self.fusion = nn.Sequential(
            nn.Linear(prompt_dim + seed_dim, fusion_hidden),
            nn.GELU(),
            nn.Linear(fusion_hidden, fusion_out),
            nn.GELU(),
        )

        conv_channels = config.get("conv_channels", CONV_CHANNELS)
        num_conv_layers = config.get("num_conv_layers", NUM_CONV_LAYERS)
        in_ch = fusion_out + self.coord_encoder.out_channels
        layers = []
        c = in_ch
        for _ in range(num_conv_layers):
            layers.append(nn.Conv2d(c, conv_channels, kernel_size=3, padding=0))
            layers.append(nn.GELU())
            c = conv_channels
        self.conv_stack = nn.Sequential(*layers)

        self.terrain_head = TerrainHead(in_channels=conv_channels, num_profiles=config.get("num_profiles", NUM_PROFILES))
        self.biome_head = BiomeHead(in_channels=conv_channels, num_biomes=config.get("num_biomes", NUM_BIOMES))
        # structure_head / structure_graph_head intentionally never instantiated:
        # structure_support == "none" for this PoC's only trained model.

    def forward(
        self,
        prompt_tokens: torch.Tensor,
        chunk_x: torch.Tensor,
        chunk_z: torch.Tensor,
        seed: torch.Tensor,
        intensity_scale: Optional[torch.Tensor] = None,
        structure_request: Optional[torch.Tensor] = None,
        macro_local_height: Optional[torch.Tensor] = None,
        flavor_zone_ids: Optional[torch.Tensor] = None,
        flavor_zone_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the ImagineNet model. Optional args mirror docs/model-spec.md's
        conditional input list for model.onnx — present only for models whose capabilities
        require them. None of them apply to `imaginator-low_intensity-no_structures` (intensity
        "low" has no `intensity_scale` knob; `structure_support: "none"` never sets
        `structure_request`; `requires_macro_field: false` means no macro inputs) — they're
        accepted and ignored here purely so this signature matches the general contract other
        tiers would extend.

        Args:
            prompt_tokens (torch.Tensor): Tokenized text prompts, [B, max_tokens] int.
            chunk_x (torch.Tensor): X coordinates of chunks, [B] int.
            chunk_z (torch.Tensor): Z coordinates of chunks, [B] int.
            seed (torch.Tensor): World seeds, [B] int64.

        Returns:
            dict: heightmap [B,16,16], profile_logits [B,num_profiles,16,16], water_level [B,16,16],
                biome_logits [B,num_biomes,4,4] — the tensors this tier actually trains — plus
                zeroed placeholders for block_volume/biome_grid/structure_* fields, kept only for
                interface parity with the full (all-tier) dict shape other code may expect.
        """
        prompt_emb = self.text_encoder(prompt_tokens)  # [B, 384]
        seed_emb = self.seed_encoder(seed)  # [B, 64]
        fused = self.fusion(torch.cat([prompt_emb, seed_emb], dim=-1))  # [B, fusion_out]

        coord_feats = self.coord_encoder(chunk_x, chunk_z)  # [B, 36, size, size]
        b, _, h, w = coord_feats.shape
        fused_grid = fused.view(b, -1, 1, 1).expand(b, fused.shape[-1], h, w)

        x = torch.cat([coord_feats, fused_grid], dim=1)
        feats = self.conv_stack(x)  # [B, conv_channels, 16, 16]

        heightmap, profile_logits, water_level = self.terrain_head(feats)
        biome_logits = self.biome_head(feats)

        return {
            "heightmap": heightmap,
            "profile_logits": profile_logits,
            "water_level": water_level,
            "biome_logits": biome_logits,
            "block_volume": torch.zeros(0),
            "biome_grid": torch.zeros(0),
            "structure_markers": torch.zeros(0),
            "structure_graph_nodes": torch.zeros(0),
            "structure_graph_edges": torch.zeros(0),
            "structure_origin": torch.zeros(0),
        }
