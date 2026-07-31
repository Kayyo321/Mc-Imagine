# Mc-Imagine Model Format Specification (.mcim)

**Version:** 0.6.0

## Overview
The `.mcim` file format is the standard distributable format for Mc-Imagine world generation models. An
`.mcim` file is a ZIP archive containing one or more trained ONNX graphs, tokenizer configuration, and a
metadata manifest. As of 0.4.0, a model may ship **up to three ONNX graphs** — a per-chunk graph (always
present), an optional per-macro-region graph, and an optional per-detail-pass graph — see "Runtime
Architecture" below for why.

This is the **canonical** tensor and manifest contract for the project — `PROJECT.md` references this
document rather than redefining the schema, and `model/src/mc_imagine_model/` (training pipeline) and
`mod/common/src/main/java/.../ModelLoader.java`/`ModelSession.java` (runtime loader) must both stay
consistent with what's written here. If a training or loading change requires a shape/field change, update
this file first and bump `format_version`.

## Capability Tiers
Models are not one-size-fits-all — a model's manifest declares what kind of prompts it can actually satisfy,
and both the mod's chunk generator and the world-creation UI use this declaration to decide what's possible.
Two independent axes matter most in practice:

- **`intensity`**: `"low"` | `"medium"` | `"high"` — **attention to detail**, not overall creative
  wildness. It governs how much fine-grained work the model puts into what it generates:
  - `"low"` — broad-strokes generation. Coarse heightmaps, a narrow block palette, minimal terrain
    texturing; if structures are supported at all, they're placed with minimal dressing.
  - `"medium"` — moderate fidelity. More varied block texturing, secondary terrain features (small outcrops,
    foliage clusters, shoreline detail), and structures (if supported) gain secondary dressing (torches,
    banners, rubble) without added structural complexity.
  - `"high"` — maximal fidelity. Dense environmental detail, layered terrain texturing, and — for models that
    also support structures — fully dressed rooms with environmental storytelling detail, not just the bare
    room shape.
- **`structure_support`**: `"none"` | `"basic"` | `"intricate"` — whether, and how, the model handles
  user-requested structures at all. Independent of `intensity`: a model can be high-intensity terrain with no
  structures, or lower-intensity terrain paired with intricate structures.
  - `"none"` — no structure output at all; vanilla Minecraft structure generation (villages, ruins, etc.)
    runs untouched on top of the model's terrain.
  - `"basic"` — the model emits `structure_markers` (see below): scattered single-piece motifs (a ruin, a
    small shrine) at chosen positions, still coexisting with vanilla structures.
  - `"intricate"` — the model emits a full `structure_graph` (see below): a composed, multi-room structure
    with narrative pacing, connections, and loot/redstone assignment. This is the tier that makes a prompt
    like *"castles in deep forests, abandoned, with hidden rooms, redstone escape-room puzzles, and loot
    worth the effort"* achievable — see the worked example at the bottom of this document.

These two example model IDs are used throughout project docs as the concrete reference points for the two
ends of this spectrum:
- `imaginator-low_intensity-no_structures`
- `imaginator-high_intensity-intricate_structures`

---

## Runtime Architecture: Two-Tier Lazy Generation

This is the central design decision governing how a `.mcim` model actually drives world generation, and it
resolves a real tension:

1. Minecraft worlds are effectively unbounded (±30,000,000 blocks per axis). **Generating "the whole world"
   at world-creation time is both impossible and pointless** — a player will only ever visit a vanishing
   fraction of it. Vanilla Minecraft itself never does this: chunks generate lazily, the first time a player
   approaches them, and are cached (to disk, in vanilla's case) once generated. Mc-Imagine preserves this
   exact behavior rather than replacing it.
2. But naive, fully-independent per-chunk inference — which is all the pipeline does today — cannot produce
   spatially coherent large-scale landforms (a valley system that actually reads as one continuous valley
   across dozens of chunks) or spatially contiguous style regions (a prompt asking for "multiple biomes that
   contain different flavors of a dark fantasy world" needs those flavors to occupy coherent patches, not
   flicker chunk-to-chunk). Some information has to be shared across chunks that haven't been generated yet.

**Resolution: a model may ship a second, much cheaper ONNX graph, `macro.onnx`, alongside the existing
per-chunk `model.onnx`.** `macro.onnx` operates at **macro-region** granularity: 32×32 chunks (512×512
blocks) — deliberately chosen to match vanilla Minecraft's own `.mca` region-file boundary, so "a
macro-region" maps onto a partition the engine already uses, rather than an arbitrary new unit.

- `macro.onnx` is a pure, deterministic function of `(prompt, seed, region_x, region_z)` — exactly like
  vanilla's seeded noise functions, it is **lazily evaluable at any region coordinate**, with no dependency on
  what's been generated before it or around it. It is never run for "the whole world" — it runs once per
  macro-region, **the first time any chunk inside that region is requested**, and its output is cached
  (mod-side `macroCache`, alongside the existing `chunkCache` — see `PROJECT.md` §"Caching & Registries").
- `model.onnx` is unchanged in spirit — invoked once per chunk, on demand, exactly when the chunk loads — but
  now additionally conditioned on a small slice of its containing macro-region's cached output, sliced and
  passed in by the mod (not recomputed by `model.onnx` itself).
- Models that don't need cross-chunk coherence (`intensity: "low"` paired with `structure_support: "none"`)
  may skip `macro.onnx` entirely — set `capabilities.requires_macro_field: false` and omit the file. This
  keeps the accessible, lightweight tier genuinely lightweight: no second graph to ship, load, or evaluate.
- **World-creation time work stays minimal**: validate/tag the prompt against the selected model's
  capabilities (§"Model Selection & Prompt Validation" in `PROJECT.md`), load the model's graph(s), and
  *optionally* warm the `macro.onnx` cache for a small radius (1–2 macro-regions) around the spawn point only
  — purely so the player's first steps don't hit a cold-cache stall. Nothing resembling "the whole world," or
  even "the whole starting region," is ever generated up front. Every chunk beyond that tiny warm radius is
  generated lazily, in the background, exactly when the chunk-loading system asks for it — matching the
  async dispatch (`CompletableFuture`/`Executor`) `ImagineChunkGenerator.fillFromNoise` already uses today.

This gives global coherence **without** global generation: every chunk gets a deterministic, consistent
answer to "which style region am I in, how close to a boundary, what's the regional landform" without ever
computing more than the small number of macro-regions a player has actually walked through.

## Structure Placement Determinism

Vanilla Minecraft places rare structures (villages, strongholds) using a seeded, jittered per-region grid
with `spacing`/`separation` parameters, computable independently of generation order. `intricate`-tier
structures work the same way, driven by `macro.onnx`'s `structure_candidate` output:

- `hash(seed, region_x, region_z)` against `capabilities.structure_spacing_regions` (default `4`) decides
  whether a given macro-region contains a candidate structure site at all — on average one every 4
  macro-regions (≈ every 128×128 chunks: rare and special, deliberately, not "a castle in every chunk"), the
  same spacing/separation idea vanilla structure sets already use.
- A second hash jitters the candidate's position within the region (`structure_candidate`'s `offset_x`/`offset_z`).
- Because this is pure seeded math, **any chunk, visited in any order, can independently determine whether it
  overlaps a candidate structure's footprint** — no dependency on the structure's "root" chunk having been
  generated first. This is what lets `applyCarvers` and vanilla structure generation correctly skip a
  reserved footprint even when a player approaches a castle from the side, long before its gatehouse chunk is
  ever loaded.
- The room graph itself (`structure_graph_nodes`/`_edges`/`_origin`) is decoded only once per candidate site —
  the first time any chunk touching it is requested (`structure_request: 1`) — and the compiled result
  (footprint + placed rooms) is cached in a mod-side structure registry, keyed by candidate-site id. Every
  subsequent chunk touching the same footprint reads the cached compiled structure; `StructureGraphHead`
  never runs twice for the same castle.

## Detail Passes (How "Intensity" Actually Runs)

`intensity` is deliberately **not** implemented as "swap in a bigger neural network for higher tiers" — that
would mean the accessible `"low"` tier and the demanding `"high"` tier need entirely different architectures
and hardware budgets, defeating the point of a single shared contract with `requirements.min_ram_mb` scaling
smoothly across tiers.

Instead, `capabilities.detail_passes` (an integer — conventionally `0` for `"low"`, `1` for `"medium"`, `2`
for `"high"`) counts a fixed number of small, **additive decoration passes** that run *after* the base
`model.onnx` call (and after structure compilation, for `intricate` models):

- Each pass scatters micro-features — loose rock, roots, moss, cobwebs, rubble, hanging vines, ambient
  debris — onto blocks the base pass (or `StructureCompiler`) already placed. Passes are purely additive:
  they never alter the base heightmap, block-volume shape, or room layout, only dress it.
- A pass may be backed by a small optional third ONNX graph, `detail.onnx` (present only if the model author
  chose an ML-driven decorator; see "Detail Graph" below), **or** by the mod's built-in procedural scatter
  routine, seeded off chunk coordinates and driven by lightweight tags the base pass already output. Either
  way cost scales linearly and cheaply with tier, instead of requiring a categorically larger base model for
  `"high"`.
- The same mechanism dresses `intricate`-tier structures: after `StructureCompiler` stamps a room template,
  `detail_passes` runs the same scatter logic inside that room's volume (cobwebs in a forgotten library,
  scorch marks by a trap doorway) — the difference between a `"high"` intensity castle and a `"medium"` one
  is entirely in this layer, not in the room graph or template library itself.

## Foundation Architecture

Rather than designing every component from a blank page, each piece of `ImagineNet` is deliberately built on
an established, proven foundation, chosen specifically for the "must run per chunk, in real time, on a
player's own machine" constraint:

- **Text encoder (`PromptEncoder`)**: a MiniLM-class pretrained sentence encoder — the architecture family
  distributed as `sentence-transformers/all-MiniLM-L6-v2` (6 transformer layers, 384-dim embeddings, ~22M
  parameters) — replaces training a tokenizer + transformer from scratch. Frozen initially, ONNX-exportable
  at low opset requirements, comfortably within the `min_ram_mb` budgets declared in the examples below. An
  LLM-scale encoder is explicitly ruled out by the real-time-per-chunk constraint; a pretrained model already
  understands language broadly, so training only has to teach the mapping from that embedding space to
  terrain/structure output, not language itself.
- **Terrain/base generation (`TerrainHead`/`BiomeHead`)**: a convolutional, single-forward-pass generator in
  the lineage of **TOAD-GAN → World-GAN** — GAN research that trains directly on Minecraft block-token grids
  (World-GAN specifically), rather than pixel images. Chosen over diffusion-style architectures precisely
  because it's single-pass: a multi-step sampler is the wrong computational profile for a per-chunk,
  real-time workload. Weights aren't reused directly (different conditioning inputs, different tensor
  contract than either paper used) — the architecture family and training approach are.
- **Structure composition (`StructureGraphHead`)**: does not freely generate an unconstrained graph. It
  produces *preference weights* over the template library (which room/connector templates the prompt makes
  likely — "abandoned," "hidden," "escape-room," "worth the effort") that drive a **Wave Function Collapse**
  (WFC) constraint-propagation pass. WFC guarantees every placed template's connector sockets validly match
  its neighbors by construction — no dangling hidden doors, no impossible adjacencies — instead of hoping a
  free-generation sequence model never emits an invalid graph. The model only has to learn *taste*; structural
  validity is the algorithm's job, not something it has to learn perfectly from limited data.

## File Structure
A typical `.mcim` archive contains:
```
model.mcim
├── manifest.json
├── model.onnx        (per-chunk graph — always present)
├── macro.onnx         (per-macro-region graph — present iff capabilities.requires_macro_field is true)
├── detail.onnx         (per-detail-pass graph — optional; present only for ML-driven decoration, see "Detail Passes")
└── tokenizer.json
```

## `manifest.json` Schema
The manifest provides the mod with essential information about how to load and use the model.

- **`format_version`**: (String) Version of the .mcim spec (e.g., "0.6.0").
- **`model`**:
  - `id`: (String) Stable machine identifier (e.g. `"imaginator-high_intensity-intricate_structures"`).
  - `name`: (String) Display name of the model.
  - `version`: (String) Model version.
  - `author`: (String) Creator of the model.
  - `description`: (String) Brief description of what the model generates well.
  - `license`: (String) License under which the model is distributed.
- **`capabilities`**:
  - `intensity`: (String) `"low"`, `"medium"`, or `"high"` — attention to detail in generated output. See "Capability Tiers" above.
  - `terrain`: (Boolean) True if the model predicts terrain heightmap/volume.
  - `biomes`: (Boolean) True if the model predicts biomes.
  - `caves`: (Boolean) True if the model itself authors voids in `block_volume` — overhangs, arches, undercuts, cave mouths — as opposed to emitting a heightfield volume. Meaningful from 0.6.0; every model through 0.5.0 declared `false`. **Read it as "the model chose these voids", not "this world has caves":** vanilla's carvers run after generation and cut caves and ravines through whatever solid mass the model wrote regardless of this flag, which is a deliberate design win and predates the flag being usable. A `false` model still yields a world with caves in it; a `true` model yields one whose surface geometry is not a function of a heightmap.
  - `structure_support`: (String) `"none"`, `"basic"`, or `"intricate"` — see "Capability Tiers" above. Supersedes the old boolean `structures` field; a manifest with `structure_support` omitted is treated as `"none"`.
  - `prompt_tags`: (List of Strings) Semantic tags the model can act on (e.g. `"terrain"`, `"biome_blend"`, `"structures"`, `"loot"`, `"redstone"`). The world-creation UI tags the player's prompt with the same vocabulary and warns/strips any clause whose tag isn't in this list rather than failing outright.
  - `block_palette`: (List of Strings) Namespaced IDs of blocks the model can output, indexed by `block_volume`'s integer ids (index `0` is always air).
  - `biome_palette`: (List of Strings, added in 0.5.0) Namespaced biome IDs (e.g. `"minecraft:desert"`) that `biome_grid` indexes into — the exact mirror of `block_palette`, for the same integer-id-to-registry-entry resolution. Optional; a manifest omitting it cannot have `biome_grid` consumed by a `BiomeSource` and the mod treats biome assignment as vanilla-default.
  - `max_prompt_tokens`: (Integer) Maximum context length for the text prompt.
  - `requires_macro_field`: (Boolean) Whether this model ships and requires `macro.onnx` for cross-chunk coherence. `false` for the accessible `"low"`/`"none"` tier, which may omit the second graph entirely.
  - `detail_passes`: (Integer) Number of additive decoration passes run after the base call — see "Detail Passes" above. Conventionally `0`/`1`/`2` for `"low"`/`"medium"`/`"high"`.
  - `max_rooms`: (Integer, `structure_support: "intricate"` only) Fixed upper bound on room-graph size — see "Output Tensors" below.
  - `template_library_version`: (String, `structure_support: "intricate"` only) Version of the structure template library this model was trained against.
  - `structure_spacing_regions`: (Integer, `structure_support: "intricate"` only) Average macro-regions between candidate structure sites — see "Structure Placement Determinism" above.
  - `flavor_zones`: (List of Objects, optional) Named style-region table for models supporting multi-flavor terrain within one aesthetic, referenced by `macro.onnx`'s `flavor_zone_ids` output. Each entry: `{"id": Integer, "name": String, "block_bias": [String]}`.
- **`requirements`**:
  - `min_ram_mb`: (Integer) Minimum system RAM required.
  - `recommended_vram_mb`: (Integer) Recommended GPU VRAM for optimal performance.
  - `onnx_opset`: (Integer) The ONNX opset version the model was exported with.
- **`io`**: Declares input/output tensor names per graph, so a `.mcim` can be sanity-checked before loading.
  - `chunk`: `{ "input_names": [...], "output_names": [...] }` — for `model.onnx`, always present.
  - `macro`: `{ "input_names": [...], "output_names": [...] }` — for `macro.onnx`, present iff `capabilities.requires_macro_field` is true.
  - `detail`: `{ "input_names": [...], "output_names": [...] }` — for `detail.onnx`, present iff that optional graph ships.

## Input Tensors: `model.onnx` (per chunk)
- `prompt_tokens`: `int32[1, max_tokens]` - Tokenized text prompt.
- `chunk_x`: `int32[1]` - X coordinate of the chunk being generated.
- `chunk_z`: `int32[1]` - Z coordinate of the chunk being generated.
- `seed`: `int64[1]` - The world seed.
- `intensity_scale`: `float32[1]` - (Present iff `intensity` is `"medium"` or `"high"`.) Runtime-tunable detail strength within the model's declared tier.
- `structure_request`: `int32[1]` - (Present iff `structure_support == "intricate"`.) Set by the mod from the containing macro-region's cached `structure_candidate` output — `1` only for chunks overlapping a candidate structure footprint.
- `macro_local_height`: `float32[2, 2]` - (Present iff `requires_macro_field`.) The 4 nearest samples from the containing macro-region's `region_heightfield`, sliced by the mod before the call, for the model to interpolate large-scale landform from.
- `flavor_zone_ids`: `int32[4]` - (Present iff `requires_macro_field`.) Passthrough of the containing macro-region's candidate flavor zones.
- `flavor_zone_weights`: `float32[4]` - (Present iff `requires_macro_field`.) Passthrough blend weights, summing to 1, for smooth style blending near zone boundaries.

## Input Tensors: `macro.onnx` (per macro-region — only present iff `requires_macro_field`)
- `prompt_tokens`: `int32[1, max_tokens]` - Same tokenized prompt as the chunk graph.
- `seed`: `int64[1]` - The world seed.
- `region_x`: `int32[1]` - Macro-region X coordinate (1 region = 32×32 chunks = 512×512 blocks, aligned to vanilla's `.mca` region files).
- `region_z`: `int32[1]` - Macro-region Z coordinate.

## Output Tensors: `macro.onnx`
- `region_heightfield`: `float32[33, 33]` - Coarse, large-scale landform samples spanning the region plus a 1-sample margin (33 = 32 chunk-boundaries + 1 for continuity), which `model.onnx` interpolates and layers local detail on top of.
- `flavor_zone_ids`: `int32[4]` - Up to 4 candidate flavor zones influencing this region (indices into `capabilities.flavor_zones`).
- `flavor_zone_weights`: `float32[4]` - Corresponding blend weights (sum to 1).
- `structure_candidate`: `int32[3]` - `(has_structure: 0|1, offset_x, offset_z)` — see "Structure Placement Determinism" above.

## Output Tensors: `model.onnx` (per chunk)
Every model must produce the terrain tensors. Structure tensors are additive and only required at the
`structure_support` tier the manifest declares — a `"none"` model omits both; a `"basic"` model adds
`structure_markers`; an `"intricate"` model adds the three `structure_graph_*` tensors instead.

**Axis order (pinned in 0.5.0 — previously ambiguous):**
- `heightmap`: indexed `[z][x]` — matches the mod's existing `heightmap[z*16 + x]` read.
- `block_volume`: indexed `[x][y][z]`, with `y` running from `minY = -64` (index `0`) to `maxY = 319` (index `383`).
- `biome_grid`: indexed `[x][y][z]` at quarter resolution (one entry per 4×4×4 volume; `y` likewise from `minY`).

A transposed axis order is the most likely silent failure in the whole pipeline — implementations on both
sides (training export, mod loader) must conform to this exactly, not to whatever felt natural locally.

**All tiers:**
- `heightmap`: `int32[16, 16]` - **The topmost solid cell of `block_volume` for that column** (redefined in 0.6.0 — see below). `[z][x]`.
- `block_volume`: `int32[16, 384, 16]` - 3D grid of block state IDs mapped to `capabilities.block_palette`. `[x][y][z]`. **Not derivable from the other outputs** (see below).
- `biome_grid`: `int32[4, 96, 4]` - Biome distribution per 4x4x4 volume, mapped to `capabilities.biome_palette`. `[x][y][z]` at quarter resolution.

**`block_volume` is authoritative, and is no longer derivable (0.6.0).** Through 0.5.0 every shipped model
expanded `block_volume` from `(heightmap, profile, water_level)` with a fixed rule — `y <= h` is solid, one
transition per column, in the graph, by construction. Three separate docstrings in the training pipeline
assumed that derivability. It no longer holds: a 0.6.0 model predicts per-cell occupancy directly, so a
column may contain **any number of alternating solid and air runs**. Consumers must read `block_volume`
cell-by-cell and must not reconstruct it from `heightmap`. This is what makes overhangs, natural arches,
undercut cliffs and walk-in cave mouths representable at all; under the old rule they were not merely
unimplemented but *unexpressible*.

**`heightmap` means "topmost solid", not "the height the model predicted" (0.6.0).** The two coincided
before and do not now: a model that internally predicts an anchor height and then carves material away
above it would disagree with itself. The mod uses `heightmap` for `getBaseHeight`, for priming
`WORLD_SURFACE_WG`/`OCEAN_FLOOR_WG`, and hence for vanilla decoration and structure placement — so if the
two definitions drift, decoration floats in the air and structures bury themselves. Exporters **must emit
`heightmap` by reading back the volume they just produced**, not by rounding an internal prediction, which
makes the guarantee true by construction. Where a column somehow contains no solid cell at all, emit the
model's own anchor height as the floor rather than an arbitrary sentinel.

Note what does *not* change: both tensors keep their shape, their dtype and their pinned axis order. This is
a **semantic** break, which is exactly the kind that passes every shape check and fails in the world.

**`structure_support: "basic"` only:**
- `structure_markers`: `float32[N, 5]` - List of potential single-piece structures (type_id, x, y, z, probability). Each marker stamps one hand-authored NBT template (a ruin, a shrine) at the given position — no internal composition.

**`structure_support: "intricate"` only:**
A full structure is not raw voxels — an ONNX regression model cannot be trusted to emit *functionally
correct* redstone from scratch. Instead, the model composes a bounded-size **room graph**: it chooses which
rooms to include, how they connect, and which pre-authored piece fills each role, with a Wave Function
Collapse pass (see "Foundation Architecture") guaranteeing the result is structurally valid. The actual room
and redstone-contraption content lives in a curated template library shipped with the mod
(`mod/common/src/main/resources/structures/`), not inside the ONNX model — the model's job is narrative
composition (pacing, connectivity, difficulty/reward), not circuit design.
- `structure_graph_nodes`: `int32[MAX_ROOMS, 4]` - Fixed-size, zero-padded array of room records: `(room_type_id, size_class, loot_tier, redstone_template_id)`. `room_type_id` and `redstone_template_id` index into the template library manifests below; `room_type_id == 0` marks an unused/padding slot.
- `structure_graph_edges`: `int32[MAX_ROOMS, MAX_ROOMS]` - Adjacency matrix; `0` means no connection, any other value is a `connector_type_id` (corridor, hidden door, trapdoor, locked door) indexing the same template library.
- `structure_origin`: `int32[3]` - Chunk-relative `(x, y, z)` anchor where the structure's root room is placed; the `StructureCompiler` (mod-side, see `PROJECT.md` §"Structure Compiler") lays out the remaining rooms from there using each template's declared footprint and the edge connections.
- `MAX_ROOMS` is a fixed constant declared per-model in `capabilities.max_rooms` (default 12) so the tensor shape is static, as ONNX requires.

## Detail Graph: `detail.onnx` (optional — only if a model uses ML-driven decoration)
Called once per configured `detail_passes`, after the base chunk pass (and after structure compilation, for
chunks touching a structure footprint). If a model doesn't ship `detail.onnx`, the mod's built-in procedural
decorator handles decoration instead, using tags from the base pass — `detail.onnx` is an optional upgrade,
not a requirement, even when `detail_passes > 0`.

**Inputs:**
- `prompt_tokens`: `int32[1, max_tokens]`
- `chunk_x`, `chunk_z`: `int32[1]`
- `seed`: `int64[1]`
- `pass_index`: `int32[1]` - Which detail pass this is (`0`-indexed, up to `detail_passes - 1`).
- `block_volume`: `int32[16, 384, 16]` - The chunk's current block state (base pass output, plus any prior detail passes already applied), so each pass can condition on what's already there.

**Outputs:**
- `detail_overrides`: `float32[N, 4]` - Sparse additive placements: `(x, y, z, block_id)`. Only non-air, additive changes — this pass never removes or reshapes what's already placed, only dresses it.

## Structure Template Library
`room_type_id` and `redstone_template_id` are not learned content — they're indices into a **hand-authored**,
versioned library of NBT structure pieces and redstone contraption blueprints shipped alongside the mod
(planned location: `mod/common/src/main/resources/structures/rooms/` and
`mod/common/src/main/resources/structures/redstone/`, each entry with a stable integer id and a declared
footprint/connector-socket set, analogous to a `manifest.json` for structures). Connector sockets are what the
Wave Function Collapse pass (see "Foundation Architecture") propagates constraints over — every template
declares which connector types it can accept on which faces, so WFC can guarantee valid adjacency. A model can
only ever reference ids that exist in the library version it was trained against — recorded in
`capabilities.template_library_version`, so a `.mcim` and the mod build it runs against can be checked for
compatibility the same way `format_version` is checked today.

## Tokenizer Format
`tokenizer.json` follows the standard HuggingFace Tokenizers format, providing vocabulary, merges, and normalization rules needed to convert the player's text prompt into `prompt_tokens`. For this reference model
family, `tokenizer.json` is **BERT-family WordPiece**, uncased, matching the MiniLM foundation
(`docs/model-spec.md` §"Foundation Architecture" / `sentence-transformers/all-MiniLM-L6-v2`'s own tokenizer):
lowercase + accent-stripped input, greedy longest-match-first subword splitting with `##` continuation
prefixes, special tokens `[PAD] = 0`, `[CLS] = 101`, `[SEP] = 102`. `prompt_tokens` is
`[CLS] token... [SEP]` padded with `0` to `max_prompt_tokens`. The mod's Java `PromptTokenizer` and the
training pipeline's Python tokenizer must both implement this exact scheme — a mismatch here is invisible at
runtime (no crash, just silently wrong embeddings) and is validated by a golden-vector test (a shared set of
`(prompt, token_ids)` pairs both implementations must reproduce exactly).

## Versioning Policy
Breaking changes to the manifest schema or expected tensor shapes will result in a minor version bump (e.g., 0.3.0 to 0.4.0). The mod will warn players if they attempt to load an incompatible format version.

## Versioning
- **0.6.0**: `block_volume` is no longer derivable from `(heightmap, profile, water_level)` — a model
  predicts per-cell occupancy and a column may hold any number of solid/air runs; `heightmap` is redefined
  as the topmost solid cell of `block_volume` (both in "Output Tensors" above); `capabilities.caves`
  becomes meaningful. No tensor shape, dtype or axis order changes.
  **0.5.0 manifests remain loadable and 0.5.0 models remain correct.** A heightfield volume is a valid
  volume — one solid run per column is a legal special case of "any number of runs" — so a 0.5.0 model's
  output is bit-for-bit unchanged under a 0.6.0 loader, and nothing in the mod branches on
  `format_version` to produce terrain. The redefinition of `heightmap` is likewise compatible in that
  direction: for a heightfield volume the topmost solid cell *is* the predicted height, so 0.5.0 models
  already satisfied the 0.6.0 definition; only models that carve above their own anchor could have
  distinguished the two, and none existed before 0.6.0.
  Following the precedent 0.5.0 set for 0.4.0, the version is recorded and logged so an operator can tell
  which contract a loaded model honours — which matters here precisely because a 0.6.0 model that fails
  and falls back to heightfield terrain looks exactly like a working 0.5.0 model.
- **0.5.0**: Pinned axis order for `heightmap`/`block_volume`/`biome_grid` (previously ambiguous — see
  "Output Tensors" above); added optional `capabilities.biome_palette`; documented the tokenizer scheme
  (BERT-family WordPiece) authoritatively rather than leaving it to "standard HuggingFace format" alone.
  **0.4.0 manifests remain loadable**: every field introduced in 0.5.0 is optional and the loader defaults
  it (`biome_palette` absent ⇒ `biome_grid` is extracted but not consumed by a `BiomeSource`; axis order was
  always intended to be as pinned here, so no 0.4.0 manifest's data changes meaning, only the spec's
  precision about it does).

---

## Example: `imaginator-low_intensity-no_structures`
Handles pure terrain-shaping prompts, e.g. *"Vast deep valleys, beautiful mountains overlooking
water-filled craters."* Broad-strokes terrain — shape without fine texturing — and no structure output;
vanilla structure generation is left fully in control of anything built on top of this model's terrain. This
is the accessible tier: `requires_macro_field: false` and `detail_passes: 0` mean only `model.onnx` ever
loads — no second or third graph, no macro-region cache, minimal RAM footprint. Cross-chunk continuity relies
on the terrain being intentionally conservative (small heightmap deltas) rather than active coherence
machinery — acceptable at this tier because the prompts it targets don't ask for coordinated large-scale
landforms or spatial style zones the way the high-intensity example below does.

The manifest below illustrates the v1.1.2 / 0.6.0 configuration with `format_version: "0.6.0"` and `caves: true`.

```json
{
  "format_version": "0.6.0",
  "model": {
    "id": "imaginator-low_intensity-no_structures",
    "name": "Imaginator - Low Intensity",
    "version": "1.1.2",
    "author": "Mc-Imagine Team",
    "description": "Conservative terrain shaping from natural-landscape prompts. No structure generation.",
    "license": "CC-BY-4.0"
  },
  "capabilities": {
    "intensity": "low",
    "terrain": true,
    "biomes": true,
    "caves": true,
    "structure_support": "none",
    "prompt_tags": ["terrain", "biome_blend"],
    "block_palette": ["minecraft:stone", "minecraft:dirt", "minecraft:grass_block", "minecraft:water", "minecraft:sand"],
    "biome_palette": ["minecraft:plains", "minecraft:desert", "minecraft:forest", "minecraft:snowy_plains", "minecraft:swamp", "minecraft:savanna", "minecraft:taiga", "minecraft:ocean"],
    "max_prompt_tokens": 128,
    "requires_macro_field": false,
    "detail_passes": 0
  },
  "requirements": {
    "min_ram_mb": 2048,
    "recommended_vram_mb": 1024,
    "onnx_opset": 17
  },
  "io": {
    "chunk": {
      "input_names": ["prompt_tokens", "chunk_x", "chunk_z", "seed"],
      "output_names": ["heightmap", "block_volume", "biome_grid"]
    }
  }
}
```

## Example: `imaginator-high_intensity-intricate_structures`
Everything the low-intensity model does, but with dense environmental detail on terrain (layered texturing,
secondary features), spatially coherent multi-flavor style zones across large landforms, *and* the
`structure_graph` output. `requires_macro_field: true` and `detail_passes: 2` mean this model ships all three
graphs (`model.onnx`, `macro.onnx`, `detail.onnx`) and relies on the full two-tier lazy generation system
described in "Runtime Architecture" above. This model is built to handle two different kinds of prompts —
worked out below.

### Worked example 1: multi-biome dark fantasy terrain
Prompt: *"Vast valleys with multiple biomes that contain different flavors of a beautiful, dark fantasy type
world."* No structures requested here — this exercises the terrain/flavor-zone machinery on its own.

- The valley macro-shape comes from `macro.onnx`'s `region_heightfield`: the prompt embedding biases the
  landform generator toward broad, continuous valley systems (low-frequency large-scale shape), evaluated
  lazily per macro-region as the player explores in any direction — never computed for a bounded "whole
  world."
- The "multiple biomes, different flavors of one dark fantasy world" requirement is exactly what
  `flavor_zone_ids`/`flavor_zone_weights` exist for: this model declares 4 flavor zones sharing one aesthetic,
  and `macro.onnx` assigns each macro-region a blend across up to 4 of them, so zones form coherent,
  smoothly-blended patches across the valley rather than flickering chunk-to-chunk:

| Flavor Zone | `id` | `block_bias` (sample) | Vibe |
|---|---|---|---|
| Blighted Moor | 1 | `coarse_dirt`, `cobweb`, `spruce_log` (bare, scattered) | Fog-choked lowland |
| Shadowbark Forest | 2 | `dark_oak_log`, `dark_oak_leaves`, `podzol`, `moss_block` | Dense, light-starved canopy |
| Ashen Wastes | 3 | `blackstone`, `basalt`, `polished_blackstone` | Scorched volcanic valley floor |
| Moonlit Fen | 4 | `mud`, `mangrove_roots`, `lily_pad`, `glow_lichen` | Eerie moonlit wetland |

  At `intensity: "high"`, both `detail_passes` layer in per-zone ambient dressing (cobwebs and dead
  undergrowth in the Blighted Moor, glow lichen clusters in the Moonlit Fen) on top of the base terrain —
  the same detail-pass mechanism used inside structures, just applied to open terrain here.

### Worked example 2: the castle
Prompt: *"Castles can be found in deep forests, that look abandoned, with hidden rooms, redstone
contraptions creating an escape-room-esc theme with great loot at the end of the puzzles."*

`macro.onnx`'s `structure_candidate` output (per "Structure Placement Determinism" above) marks a sparse set
of macro-regions — on average one every `structure_spacing_regions: 4` regions — as candidate castle sites.
The first chunk a player loads that overlaps one triggers `structure_request: 1`, and `StructureGraphHead`
(guided by Wave Function Collapse over the template library) resolves the room graph once, cached from then on:

| Node (room) | `room_type_id` | `redstone_template_id` | `loot_tier` | Connects to |
|---|---|---|---|---|
| 0 — Overgrown gatehouse | `ruined_gatehouse` | — | 0 | 1 |
| 1 — Entry hall (hidden door) | `great_hall` | `hidden_lever_door` | 0 | 0, 2 |
| 2 — Trapped corridor | `trap_corridor` | `tripwire_dart_volley` | 1 | 1, 3 |
| 3 — Puzzle chamber | `piston_puzzle_room` | `piston_sequence_lock` | 2 | 2, 4 |
| 4 — Treasure vault | `treasure_vault` | — | 3 (best loot table) | 3 |

The model never designs the `hidden_lever_door` or `piston_sequence_lock` circuits themselves — those are
fixed, tested templates in the structure template library, and WFC guarantees every connector between rooms
above is socket-valid. What the model learns is pacing and structure: how many rooms, what order of
difficulty, where the payoff goes, and which prompt language ("hidden," "escape-room," "worth the effort")
maps to which template/loot choices. At `intensity: "high"`, the two configured `detail_passes` then dress
each room afterward — cobwebs and dust in the gatehouse, scorch marks by the dart volley, disturbed rubble in
the vault — without altering the room graph itself.

```json
{
  "format_version": "0.5.0",
  "model": {
    "id": "imaginator-high_intensity-intricate_structures",
    "name": "Imaginator - High Intensity / Intricate Structures",
    "version": "1.0.0",
    "author": "Mc-Imagine Team",
    "description": "Expressive multi-flavor terrain generation plus composed multi-room structures with narrative pacing, hidden connections, and loot/redstone assignment.",
    "license": "CC-BY-4.0"
  },
  "capabilities": {
    "intensity": "high",
    "terrain": true,
    "biomes": true,
    "caves": true,
    "structure_support": "intricate",
    "prompt_tags": ["terrain", "biome_blend", "structures", "loot", "redstone"],
    "max_prompt_tokens": 128,
    "requires_macro_field": true,
    "detail_passes": 2,
    "max_rooms": 12,
    "template_library_version": "0.1.0",
    "structure_spacing_regions": 4,
    "flavor_zones": [
      { "id": 1, "name": "blighted_moor", "block_bias": ["minecraft:coarse_dirt", "minecraft:cobweb", "minecraft:spruce_log"] },
      { "id": 2, "name": "shadowbark_forest", "block_bias": ["minecraft:dark_oak_log", "minecraft:dark_oak_leaves", "minecraft:podzol", "minecraft:moss_block"] },
      { "id": 3, "name": "ashen_wastes", "block_bias": ["minecraft:blackstone", "minecraft:basalt", "minecraft:polished_blackstone"] },
      { "id": 4, "name": "moonlit_fen", "block_bias": ["minecraft:mud", "minecraft:mangrove_roots", "minecraft:lily_pad", "minecraft:glow_lichen"] }
    ],
    "block_palette": ["minecraft:stone", "minecraft:cobbled_deepslate", "minecraft:mossy_stone_bricks", "minecraft:dirt", "minecraft:grass_block", "minecraft:water"],
    "biome_palette": ["minecraft:dark_forest", "minecraft:swamp", "minecraft:basalt_deltas", "minecraft:mangrove_swamp"],
    "max_prompt_tokens": 128
  },
  "requirements": {
    "min_ram_mb": 8192,
    "recommended_vram_mb": 4096,
    "onnx_opset": 17
  },
  "io": {
    "chunk": {
      "input_names": ["prompt_tokens", "chunk_x", "chunk_z", "seed", "intensity_scale", "structure_request", "macro_local_height", "flavor_zone_ids", "flavor_zone_weights"],
      "output_names": ["heightmap", "block_volume", "biome_grid", "structure_graph_nodes", "structure_graph_edges", "structure_origin"]
    },
    "macro": {
      "input_names": ["prompt_tokens", "seed", "region_x", "region_z"],
      "output_names": ["region_heightfield", "flavor_zone_ids", "flavor_zone_weights", "structure_candidate"]
    },
    "detail": {
      "input_names": ["prompt_tokens", "chunk_x", "chunk_z", "seed", "pass_index", "block_volume"],
      "output_names": ["detail_overrides"]
    }
  }
}
```
