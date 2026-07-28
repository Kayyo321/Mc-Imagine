"""
Single source of truth for the small fixed vocabularies shared across the data pipeline, the model
heads, and the exporter: the block palette `block_volume` indexes into, the biome palette
`biome_grid` indexes into, and the 8-class surface-profile table `TerrainHead`'s column-profile
logits predict over.

Keeping these in one module (rather than duplicating literal lists in `data/world_generator.py`,
`data/labeler.py`, `data/dataset.py`, `model/heads.py`, and `export/package_mcim.py`) is what
guarantees the trained model, the exported `block_volume` expansion, and the packaged manifest's
`capabilities.block_palette`/`capabilities.biome_palette` all agree on the same integer ids — a
mismatch here would be the "silently wrong" class of bug docs/model-spec.md warns about for axis
order, just one level removed.

Per docs/model-spec.md's `imaginator-low_intensity-no_structures` worked example, index 0 of
`block_palette` is always air (air is implicit in the worked example's list; we make it explicit
here since `block_volume` needs an actual "nothing here" id for the space above the terrain).
`biome_palette` mirrors the worked example's 8 biomes exactly.
"""

from typing import List, Tuple

# --- block_palette (block_volume indexes into this) ---------------------------------------------
BLOCK_PALETTE: List[str] = [
    "minecraft:air",          # 0 - always air, per docs/model-spec.md capabilities.block_palette
    "minecraft:bedrock",      # 1
    "minecraft:stone",        # 2
    "minecraft:deepslate",    # 3
    "minecraft:dirt",         # 4
    "minecraft:grass_block",  # 5
    "minecraft:sand",         # 6
    "minecraft:water",        # 7
    "minecraft:red_sand",     # 8  - mesa / badlands surface
    "minecraft:mud",          # 9  - swamp surface
    "minecraft:gravel",       # 10 - eroded / seafloor surface
    "minecraft:snow_block",   # 11 - snowy surface
    "minecraft:podzol",       # 12 - taiga forest floor
]
AIR_ID = 0
BEDROCK_ID = 1
STONE_ID = 2
DEEPSLATE_ID = 3
WATER_ID = 7

# --- biome_palette (biome_grid indexes into this) — verbatim from docs/model-spec.md example -----
BIOME_PALETTE: List[str] = [
    "minecraft:plains",        # 0
    "minecraft:desert",        # 1
    "minecraft:forest",        # 2
    "minecraft:snowy_plains",  # 3
    "minecraft:swamp",         # 4
    "minecraft:savanna",       # 5
    "minecraft:taiga",         # 6
    "minecraft:ocean",         # 7
]
NUM_BIOMES = len(BIOME_PALETTE)

# --- surface profile table (TerrainHead's 8-class column-profile logits) -------------------------
# Each entry: (name, subsurface_block_id, surface_block_id) — both index into BLOCK_PALETTE.
# subsurface fills (h-4, h); surface fills exactly h (see export/export_onnx.py's block_volume
# expansion and docs/model-spec.md's Output Tensors section).
PROFILE_TABLE: List[Tuple[str, int, int]] = [
    ("grass", 4, 5),   # 0 - dirt subsurface, grass_block surface (plains/forest/savanna default)
    ("sand", 6, 6),    # 1 - desert dunes
    ("snow", 4, 11),   # 2 - snowy surface over dirt
    ("mesa", 8, 8),    # 3 - red sand mesa / badlands
    ("stone", 2, 2),   # 4 - exposed rock (mountain peaks, steep slopes)
    ("mud", 9, 9),     # 5 - swamp
    ("gravel", 10, 10),  # 6 - eroded / seafloor
    ("podzol", 4, 12),   # 7 - taiga forest floor
]
NUM_PROFILES = len(PROFILE_TABLE)

# Default profile id most representative of each biome id above (used to bias procedural sampling
# in data/world_generator.py and to keep data/labeler.py's captions consistent with the terrain
# world_generator actually renders).
BIOME_DEFAULT_PROFILE: List[int] = [
    0,  # plains -> grass
    1,  # desert -> sand
    0,  # forest -> grass (podzol(7) also used for patchy variation, see world_generator)
    2,  # snowy_plains -> snow
    5,  # swamp -> mud
    0,  # savanna -> grass
    7,  # taiga -> podzol
    6,  # ocean -> gravel seafloor
]

# The 6 held-out prompts docs/poc-plan.md's Phase 5 gate renders at a fixed seed and judges by eye:
# "they must be visibly, unmistakably different from each other and each must match its caption."
# Deliberately never seen verbatim during training (data/labeler.py's templates produce paraphrases
# built from the same vocabulary, not these exact strings) — see training/train.py's per-epoch dump
# and the Phase 5 gate script.
GATE_PROMPTS: List[str] = [
    "vast deep valleys, beautiful mountains overlooking water-filled craters",
    "endless flat desert dunes",
    "towering snow-capped peaks",
    "shallow tropical archipelago",
    "eroded red mesa plateaus",
    "gentle rolling grassland",
]
GATE_SEED = 12345
