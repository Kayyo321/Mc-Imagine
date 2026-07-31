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

# --- biome_palette (biome_grid indexes into this) ------------------------------------------------
# Indices 0-7 are the original set from docs/model-spec.md's worked example. 8-11 were added in
# Phase 2 (docs/phase2-plan.md §0): the Day-1 PoC had no badlands-family biome at all, so
# `mesa_plateaus` terrain was forced to label itself `savanna`, and *vanilla* savanna lake
# decoration — which by design runs on top of AI terrain — scattered water pools across an
# otherwise-correct red mesa. The AI terrain was right; only the biome label was wrong.
# `stony_peaks`/`beach` are cheap wins that remove similar mislabels for peaks and shorelines.
BIOME_PALETTE: List[str] = [
    "minecraft:plains",           # 0
    "minecraft:desert",           # 1
    "minecraft:forest",           # 2
    "minecraft:snowy_plains",     # 3
    "minecraft:swamp",            # 4
    "minecraft:savanna",          # 5
    "minecraft:taiga",            # 6
    "minecraft:ocean",            # 7
    "minecraft:badlands",         # 8  - mesa/plateau country (Phase 2)
    "minecraft:eroded_badlands",  # 9  - heavily weathered mesa (Phase 2)
    "minecraft:stony_peaks",      # 10 - exposed rock above the treeline (Phase 2)
    "minecraft:beach",            # 11 - shoreline between land and ocean (Phase 2)
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
    3,  # badlands -> mesa (red sand)
    3,  # eroded_badlands -> mesa (red sand)
    4,  # stony_peaks -> stone
    1,  # beach -> sand
]
assert len(BIOME_DEFAULT_PROFILE) == NUM_BIOMES, "BIOME_DEFAULT_PROFILE must cover every biome id"

# --- height representation (Phase 2 — docs/phase2-plan.md §1b) -----------------------------------
# TerrainHead squashes height as `HEIGHT_CENTER + HEIGHT_AMPLITUDE * tanh(x)`. Day 1 used
# `63 + 96*tanh` => a hard ceiling of 159, while the generator produced terrain up to 235: the
# "towering snow-capped peaks" prompt pinned at 158.84-158.98 (std 0.03) because tanh had saturated
# and the gradient had gone to zero, killing *all* local relief. These constants are the single
# source of truth shared by heads.py (the squash), world_generator.py (the clip range it renders
# into), and train.py (a startup assertion that no target is outside the representable band), so
# that class of silent saturation cannot come back.
HEIGHT_CENTER = 96.0
HEIGHT_AMPLITUDE = 192.0
HEIGHT_MIN = HEIGHT_CENTER - HEIGHT_AMPLITUDE  # -96
HEIGHT_MAX = HEIGHT_CENTER + HEIGHT_AMPLITUDE  # 288
# Terrain is generated inside a slightly narrower band so targets never sit at the tanh asymptote
# (where gradients vanish); tanh(±2.3) ≈ ±0.98 of full scale.
TERRAIN_CLIP_MIN = -48.0
TERRAIN_CLIP_MAX = 252.0

# --- the surface-anchored occupancy band (Phase 4 — docs/phase4-plan.md §1.2) --------------------
# Everything before v1.1.2 derived `block_volume` from one predicted height per column via
# `y <= h -> solid` (export/export_onnx.py). That is a total order on each column: exactly one
# air/solid transition, by construction, in the graph, forever — so overhangs, arches, undercut
# cliffs and walk-in cave mouths were not unimplemented but *unrepresentable*. v1.1.2 replaces that
# comparison with a per-cell occupancy band, and these constants are its geometry.
#
# The band is ANCHORED TO THE PREDICTED HEIGHT, not to absolute world y. An absolute band would have
# to span the whole 384-block world height to work at every elevation, which is the 98,304-cell
# problem the band exists to avoid; anchoring means the head only ever models "the interesting 64
# blocks", wherever they are, and spends no capacity learning that y=-30 is usually stone.
#
#   band index i in [0, BAND_HEIGHT)  <->  world y = round(h) + BAND_BOTTOM_OFFSET + i
#
# i.e. index 0 is the BOTTOM of the band and index BAND_HEIGHT-1 is its top, so band index runs the
# same direction as world y. Below the band: solid rock, deterministically, exactly as before (and
# vanilla's carvers still cut deep caves down there, as they always have). Above it: air, or water
# per docs/phase4-plan.md §1.5's openness rule.
#
# WHY 64. It has to cover the tallest natural void a player can walk into plus the full horizontal
# reach of an undercut, and it is the term that sets the occupancy head's output size. 64 is the
# starting value; docs/phase4-plan.md §7's Gate 0 measures the ground truth's cavity-height
# distribution and either confirms it or grows the band. DO NOT tune this after training starts —
# it changes the tensor shape of every band array, every shard, and the head itself.
#
# WHY +2 AT THE TOP. The surface cell itself is `h`, and the band needs headroom above it so a
# column can legitimately place solid material above the anchor (the lip of an overhang seen from
# the column beneath it) rather than having the representation clip it away.
BAND_HEIGHT = 64
BAND_BOTTOM_OFFSET = -61  # world y of band index 0, relative to the column's predicted height
BAND_TOP_OFFSET = 2       # world y of band index BAND_HEIGHT-1, relative to the same height
assert BAND_TOP_OFFSET - BAND_BOTTOM_OFFSET + 1 == BAND_HEIGHT, (
    "band offsets and BAND_HEIGHT must describe the same 64 cells: an off-by-one here silently "
    "misaligns the generator's target band against the head's output band, which trains fine and "
    "produces terrain shifted vertically by one block against its own heightmap"
)

# THE BAND CAN HANG OFF THE BOTTOM OF THE WORLD, AND THAT IS EXPECTED. Terrain is generated down to
# TERRAIN_CLIP_MIN (-48), so a column anchored there puts its band bottom at -48 - 61 = -109, below
# Minecraft's floor at y=-64. The band is a *relative* window and is deliberately NOT clamped to fit
# the world: clamping the anchor `h` upward so the whole band fits (h >= -3) would forbid every deep
# canyon floor the Phase 2 work exists to produce, and clamping the band's *extent* per column would
# make its cell count position-dependent, which the fixed [64,16,16] tensor shape cannot express.
# Instead, consumers must drop band cells whose world y falls outside [MIN_Y, MAX_Y]:
#   - export/export_onnx.py writes band cells into the volume only where the world y is in range;
#   - data/world_generator.py renders the band as solid wherever it hangs below the world floor,
#     so the target and the emitted volume agree about those cells being unreachable rock.
# The top is safe by construction: BAND_TOP_OFFSET is 2 and export already clamps `h` to MAX_Y - 2.

# --- loss normalization scales (Phase 2 follow-up — MEASURED, not guessed) -----------------------
# These are deliberately NOT `HEIGHT_AMPLITUDE`. Normalizing a loss term by the head's full output
# range is principled-sounding and quantitatively disastrous, because it scales each term by how
# much the head *could* move rather than by how much the quantity actually varies. Measured on the
# regenerated dataset with an untrained net (6 batches of 64):
#
#     term                     normalized by 192      contribution
#     height MSE                     0.0367            negligible
#     slope MSE                      0.000027          ~zero
#     water MSE (x0.25)              0.0153            negligible
#     profile CE (8 classes)         2.1106            dominant
#     biome CE (12 classes)          2.4842            dominant
#     => terrain : biome  =  0.021   (a 48x imbalance)
#
# Training under that balance optimizes classification and effectively ignores terrain shape — and
# the slope term added specifically to force relief contributed ~0.0007 of the gradient, i.e. it did
# nothing at all. The scales below are chosen from what each quantity actually is:
#   HEIGHT_LOSS_SCALE - a typical terrain deviation from the regional mean, not the head's range.
#   SLOPE_LOSS_SCALE  - a typical per-block height step, which is ~1 block by definition. Dividing a
#                       block-to-block gradient by 192 is what annihilated the term.
# With these, the terms land at height ~1.3, slope ~1.0, water ~0.55, profile ~2.1, biome ~2.5 —
# same order of magnitude, so `terrain_weight`/`biome_weight`/`slope_weight` in the configs are
# meaningful knobs instead of fighting a 48x scale gap.
HEIGHT_LOSS_SCALE = 32.0
SLOPE_LOSS_SCALE = 1.0

# --- seed -> world offset (Phase 2 — docs/phase2-plan.md §1a) ------------------------------------
# The Day-1 generator reseeded its Perlin gradient table per macro-region
# (`seed*7919 + region_x*104729 + region_z*15485863`), making every region an *independent
# pseudorandom field*. The network only ever sees (prompt, seed, global coords) and cannot invert a
# hash, so MSE drove it to the conditional mean: flat terrain at the right average height. The fix
# is one canonical noise field for the whole project, sampled at `global_coord + seed_offset` — a
# deterministic, continuous, *learnable* function of coordinates, which additionally makes landforms
# continuous across region boundaries so a valley can span hundreds of blocks.
#
# Constants are chosen so every intermediate stays int64-exact and every result is < 2**24 (hence
# exactly representable in float32, which is what the exported ONNX graph computes in). The torch
# mirror of this function lives in model/positional.py and MUST stay bit-identical to it.
SEED_FOLD = 1_000_003        # folds arbitrary int64 world seeds into a safe magnitude first
SEED_MIX = 2_654_435_761     # Knuth multiplicative hash constant
SEED_HASH_MOD = 4_294_967_296

# --- the spatial period of the whole world (Phase 2 follow-up) -----------------------------------
# WORLD_PERIOD MUST equal the largest wavelength in `model/positional.py`'s
# `CoordinateEncoder.DEFAULT_WAVELENGTHS` — currently 2048 — and every shorter wavelength in that
# tuple must divide it (2048, 1024, ... 8 all do). That is not a stylistic preference: the encoder's
# feature vector phi(x, z) is built exclusively from sin/cos at those wavelengths, so phi is
# *exactly periodic with period 2048 blocks in both axes*. Two positions 2048 blocks apart are
# literally the same input to the network.
#
# The canonical noise field is therefore made exactly 2048-periodic too (see
# `data/world_generator.py`'s tileable Perlin). If it were not, hundreds of distinct world locations
# would map to identical phi while carrying *different* terrain — directly conflicting regression
# targets, which MSE resolves by predicting their mean, i.e. flat terrain. That is precisely the
# Day-1 failure mode docs/phase2-plan.md §0 diagnosed, and the canonical-field change of §1a does
# not fix it on its own: the field must additionally be *consistent with* the encoder's period.
#
# The second, equally important consequence is coverage. Training regions are laid out over
# 64 x 512 = 32768 blocks; against a 65536-block offset window that is ~3% sparse sampling of a
# 65536^2 domain. Folded onto one 2048^2 domain, the same 600-8000 regions cover it 30x+ over. That
# is the difference between a learnable function and a memorization problem.
#
# KNOWN CONSEQUENCE, ACCEPTED DELIBERATELY: terrain now repeats every 2048 blocks. Walk far enough
# in a straight line and you re-enter the same landscape. This is a documented trade of unbounded
# variety for learnability — an unlearnable target is worth nothing regardless of how varied it is.
# To extend the period later, add longer wavelengths to `CoordinateEncoder.DEFAULT_WAVELENGTHS`
# (e.g. 8192, 4096) and raise WORLD_PERIOD to the new largest wavelength. The two MUST stay equal;
# raising one without the other silently reintroduces the aliasing this constant exists to prevent.
WORLD_PERIOD = 2048

# Seed offsets span exactly one period. A larger range adds *zero* variety — an offset of
# `WORLD_PERIOD + k` samples the identical field as an offset of `k` — while creating the aliasing
# described above, so 65_536 (the previous value) was strictly worse than 2048 in both directions.
# This still yields 2048 x 2048 ~= 4.2 million distinct world translations.
OFFSET_RANGE = WORLD_PERIOD


def seed_to_offset(seed: int) -> Tuple[float, float]:
    """Maps a world seed to the `(offset_x, offset_z)` translation of the canonical noise field.

    This is what makes "same seed => same world, different seed => different layout, same character"
    true: a different seed samples a different translation of one shared field, which is essentially
    how vanilla Minecraft seeds already behave.

    Both offsets land in `[0, WORLD_PERIOD)` — one full period of the canonical field, beyond which
    translations merely repeat (see WORLD_PERIOD above).
    """
    s = int(seed) % SEED_FOLD
    h = (s * SEED_MIX) % SEED_HASH_MOD
    ox = h % OFFSET_RANGE
    oz = (h // OFFSET_RANGE) % OFFSET_RANGE
    return float(ox), float(oz)


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
