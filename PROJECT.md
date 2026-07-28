# Project: Mc-Imagine Phase 1 (Model Loader & ONNX Integration)

## Architecture
Mc-Imagine is a Minecraft mod that generates world chunks using deep learning models exported to ONNX format.
Phase 1 establishes the pipeline from model loading to in-game chunk generation:
1. `ModelLoader.java`: Scans model directory for `.mcim` archives, extracts `manifest.json` (ZIP archive), and parses metadata into `ModelInfo`.
2. `ModelSession.java`: Wraps ONNX Runtime (`OrtEnvironment`, `OrtSession`), loads `.onnx` model binary from `.mcim` archive, executes inference with chunk coordinates, seed, and prompt, returning `ChunkOutput`.
3. `ImagineChunkGenerator.java`: Minecraft ChunkGenerator implementation that invokes `ModelSession` during noise/column generation and populates chunk blocks from returned heightmap/volume tensors.
4. Python dummy model generator (`model/`): PyTorch model script exporting `model.onnx`, packaged into `dummy-test-v1.mcim` inside `mod/fabric/run/mcimagine/models/`.

Phase 1 proves the pipeline works end-to-end with a single dummy model, single-graph, single-chunk-at-a-time.
It does **not** yet address how different models with different creative scope should integrate with vanilla
Minecraft systems, nor how cross-chunk coherence (a continuous valley, spatially contiguous biome "flavors,"
a castle whose footprint spans chunks visited in any order) is possible when every chunk is inferred
independently. Those two problems — capability tiering and runtime generation architecture — are what the
rest of this document designs. Read in this order:
1. **"Model Capability Tiers & World-Generation Integration"** — what a model declares about itself, and how the mod branches on it.
2. **"Runtime Generation Lifecycle: Lazy Chunk Streaming"** — *when* and *how* inference actually happens: on-the-fly per chunk, never "the whole world at creation."
3. **"High Intensity Detail & Intricate Structures — Full Mechanics"** — how `intensity` and `structure_support` are actually implemented, end to end.
4. **"Model Training Pipeline"** — how a model satisfying all of the above actually gets made, and what it's built on.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Codebase Exploration | Map project layout, existing classes, Gradle structure | None | DONE |
| M2 | Gradle Dependency & Test Model | Add ONNX Runtime to gradle; create Python dummy model script & `dummy-test-v1.mcim` | M1 | DONE |
| M3 | ModelLoader & ModelSession | Implement ZIP/manifest parsing in `ModelLoader` & ONNX runtime inference in `ModelSession` | M2 | DONE |
| M4 | ImagineChunkGenerator Wiring | Connect `ImagineChunkGenerator` to `ModelSession` and place blocks from tensors | M3 | DONE |
| M5 | Build, Verification & Audit | Verify gradle build, run unit/E2E test validation, forensic audit check | M4 | DONE |
| M6 | Capability Manifest Schema | Extend `ModelInfo`/manifest.json with `capabilities` block; `ModelLoader` parses block palette + tier | M5 | DONE |
| M7 | Prompt Capability Validation | Prompt tagger classifies requested prompt against model capability tags at world-creation time | M6 | DONE |
| M8 | Tiered Generation Hooks | Wire `applyCarvers`/`buildSurface`/structure reservation to branch on model tier instead of no-op | M6 | DONE |
| M9 | Structure Compiler (Tier 3) | Implement room-graph decoding + template/loot/redstone composition for `intricate_structures` models | M8 | PLANNED |
| M10 | Macro-Field Runtime | Implement `MacroCache`, per-macro-region `macro.onnx` invocation, and chunk-side slicing (`macro_local_height`/`flavor_zone_*`) — supersedes the earlier open "seam handling" question, see "Runtime Generation Lifecycle" | M6 | PLANNED |
| M11 | Cache Bounding & Batched Inference | Replace unbounded `chunkCache` with bounded/evicting cache; add bounded `MacroCache`/`StructureRegistry`; batch adjacent chunk inference calls | M5, M10 | PLANNED |
| M12 | Training Data Bring-up | Implement `world_generator.py` + `chunk_extractor.py` (procedural/mined sourcing) and `labeler.py` (auto-captioning) to produce real `McImagineDataset` samples | M6 | DONE (procedural half only — `chunk_extractor.py`'s `.mca` path is deferred, see docs/poc-plan.md Phase 8) |
| M13 | ImagineNet Architecture | Implement `PromptEncoder` (MiniLM-class foundation), `CoordinateEncoder`/`SeedEncoder`, `TerrainHead`/`BiomeHead`, and train `imaginator-low_intensity-no_structures` end-to-end | M12 | DONE |
| M14 | Structure Template Library | Author the initial `mod/common/src/main/resources/structures/` room + redstone template set, including WFC connector-socket declarations, that `structure_graph` output will reference | M9 | PLANNED |
| M15 | Structure-Graph Model | Add `StructureGraphHead`/`StructureGraphLoss` + WFC constraint pass, train `imaginator-high_intensity-intricate_structures` against the M14 library — the castle benchmark | M13, M14 | PLANNED |
| M16 | Structure Placement Determinism | Implement seeded `hash(seed, region_x, region_z)` candidate-site placement (`structure_spacing_regions`) and the `StructureRegistry` cache — makes structure discovery order-independent | M10 | PLANNED |
| M17 | Detail-Pass Pipeline | Implement `detail.onnx` invocation (or the built-in procedural decorator fallback) and `detail_overrides` application, gated by `capabilities.detail_passes` | M13 | PLANNED |
| M18 | Flavor-Zone Terrain Model | Train `macro.onnx` for `imaginator-high_intensity-intricate_structures` to emit coherent `flavor_zone_ids`/`flavor_zone_weights` — the dark-fantasy-valley benchmark | M13, M17 | PLANNED |
| M19 | Foundation Model Integration | Swap `PromptEncoder` for a frozen MiniLM-class checkpoint; adopt a TOAD-GAN/World-GAN-lineage architecture for `TerrainHead` | M13 | PLANNED |

## Interface Contracts
### `ModelLoader` ↔ `ModelInfo`
- `ModelLoader.loadModels(Path dir)`: Reads `.mcim` files as ZIP, extracts `manifest.json`, parses to `ModelInfo` (name, version, id, metadata, onnxFilename).

### `ModelSession` ↔ `ImagineChunkGenerator`
- `ModelSession.init(byte[] onnxBytes)`: Initializes `OrtEnvironment` and `OrtSession`.
- `ModelSession.generateChunk(int chunkX, int chunkZ, long seed, int[] promptTokens)`: Executes ONNX inference, returns `ChunkOutput` containing float[][] heightmap / block volume tensors.
- **Planned extension**: `ModelSession` needs to manage up to three `OrtSession` instances per loaded model (`model.onnx` always; `macro.onnx`/`detail.onnx` iff the manifest declares them), not one — see "Runtime Generation Lifecycle" below for how each is invoked and cached.

## Code Layout
- `mod/build.gradle`, `mod/common/build.gradle`: Gradle build configuration
- `mod/common/src/main/java/.../ModelLoader.java`: Model archive loader & manifest parser
- `mod/common/src/main/java/.../ModelSession.java`: ONNX Runtime session & inference engine
- `mod/common/src/main/java/.../ImagineChunkGenerator.java`: Minecraft chunk generator integration
- `mod/common/src/main/java/.../generation/` *(planned, M10/M16)*: Home for the new `MacroCache`, `StructureRegistry`, and `StructureCompiler` classes described below
- `mod/common/src/main/resources/structures/` *(planned, M14)*: Hand-authored NBT room templates (`rooms/`) and redstone contraption blueprints (`redstone/`), each declaring WFC connector sockets — the template library `structure_graph` output indexes into; see `docs/model-spec.md` §"Structure Template Library"
- `model/generate_dummy_model.py`: Standalone PyTorch dummy model definition & exporter (the current, working, deliberately minimal placeholder used to validate the mod pipeline end-to-end)
- `model/src/mc_imagine_model/`: The real training package the dummy script stands in for — see "Model Training Pipeline" below for how its modules map to the design
- `docs/model-spec.md`: **Canonical** `.mcim` format spec — manifest schema, capability tiers, runtime architecture, and tensor I/O contract for all three possible graphs. Both the mod (`ModelLoader`/`ModelSession`) and the training package must conform to this document, not the other way around
- `mod/fabric/run/mcimagine/models/dummy-test-v1.mcim`: Sample test model archive containing `manifest.json` and `model.onnx`

---

## Model Capability Tiers & World-Generation Integration (Phase 2 Design)

The core idea: **a model's manifest declares what it is capable of, and the generator's behavior — which
vanilla systems it defers to, what tensors it expects, how prompts are validated — branches on that
declaration.** A model is not just a heightmap function; it's a contract about creative scope. Two example
models illustrate the two ends of that contract:

- **`imaginator-low_intensity-no_structures`** — conservative terrain shaping only. Handles prompts like
  *"vast deep valleys, beautiful mountains overlooking water-filled craters."* Produces a heightmap, a block
  volume, and a biome hint, from a single ONNX graph. Does not attempt structures at all; vanilla structure
  generation (villages, ruins, etc.) is left fully in control of anything built on top of the terrain it
  shapes.
- **`imaginator-high_intensity-intricate_structures`** — everything the low-intensity model does, but with
  denser environmental detail on terrain (layered texturing, secondary features — not just larger/wilder
  shapes), spatially coherent multi-flavor style zones across large landforms (e.g. *"vast valleys with
  multiple biomes that contain different flavors of a beautiful, dark fantasy type world"*), *and* a
  structure-composition stage that can satisfy a prompt like *"castles in deep forests, abandoned, with
  hidden rooms, redstone escape-room puzzles, and loot worth the effort."* This tier ships three ONNX graphs
  (`model.onnx`, `macro.onnx`, `detail.onnx`) working together — the full mechanics are in "Runtime
  Generation Lifecycle" and "High Intensity Detail & Intricate Structures" below.

`intensity` (`"low"` | `"medium"` | `"high"`) and `structure_support` (`"none"` | `"basic"` | `"intricate"`)
are independent axes, not one combined scale — see `docs/model-spec.md` §"Capability Tiers" for the full
definition of each. `intensity` means *attention to detail* specifically (block-palette variety, secondary
terrain features, structure dressing), not how large or wild the terrain shapes are.

### 1. Manifest Capability Schema
The full schema lives in `docs/model-spec.md` (§"Capability Tiers" and §"`manifest.json` Schema") — that
document is canonical; this section only describes how `ModelInfo.java` needs to change to consume it. Add a
`ModelCapabilities` record and a `capabilities()` accessor to `ModelInfo`, mirroring the manifest's
`capabilities` block field-for-field (`intensity`, `structure_support`, `prompt_tags`, `block_palette`,
`requires_macro_field`, `detail_passes`, `max_rooms`, `template_library_version`,
`structure_spacing_regions`, `flavor_zones`). `ModelLoader.parseModelInfo` already defensively parses optional
manifest fields (see the `getString` helper pattern) — the same pattern extends naturally to an optional
nested object, defaulting to `intensity: "low"` / `structure_support: "none"` / `requires_macro_field: false`
for legacy manifests (like the current dummy model) that don't declare a `capabilities` block at all.

The two worked examples from `docs/model-spec.md` — `imaginator-low_intensity-no_structures` and
`imaginator-high_intensity-intricate_structures` — are the concrete deliverables the rest of this section is
building toward, including full worked examples for both the dark-fantasy-valley prompt and the castle prompt.

This also finally gives the **block palette** a home outside code: today `ImagineChunkGenerator.getBlockStateFromId`
is a hardcoded `switch` over 8 block IDs, which means every model is forced to agree on the same tiny palette.
Moving the palette into the manifest lets a `high_intensity` model use richer materials (deepslate variants,
mossy bricks, custom log types) without touching Java.

### 2. Model Selection & Prompt Validation
At world-creation time (the custom UI already referenced in the recent commit history), the user picks a
model *and* writes a prompt. Before generation starts:
- A lightweight prompt tagger (regex/keyword rules to start; could later be a small classifier head on the
  same ONNX model) extracts semantic tags from the prompt — `terrain`, `structures`, `redstone`, etc.
- Tags are checked against the selected model's `capabilities.prompt_tags`.
- Unsupported clauses are **not** a hard failure — they're stripped before tokenization and surfaced to the
  user as a warning ("this model doesn't support structures; the castle description will be ignored, only
  terrain shaping will apply"). This keeps world creation from being an all-or-nothing gate and matches how
  a low-intensity model gracefully "can't" do what a high-intensity one can, without crashing.
- This is also the *only* generation-adjacent work that happens synchronously at world creation — see
  "Runtime Generation Lifecycle" for why actual terrain/structure inference never happens here.

### 3. Generation Step Integration, Branched by Tier
Today `applyCarvers`, `buildSurface`, and `spawnOriginalMobs` in `ImagineChunkGenerator` are empty no-ops, and
`fillFromNoise` writes the AI's block volume directly with no vanilla layer on top. That's fine for the dummy
model but doesn't express the actual design intent. Proposed branching, keyed off `ModelInfo.capabilities()`:

| Vanilla step | `no_structures` tier | `intricate_structures` tier |
|---|---|---|
| `fillFromNoise` | AI heightmap/volume authoritative (current behavior) | Same, but conditioned on macro-region context and layered with `detail_passes` (below) |
| `applyCarvers` | Vanilla cave/ravine carvers run normally on top of the AI stone volume | Carvers **skip** any chunk column reserved by a placed structure's bounding volume (via `StructureRegistry`, see below) |
| `buildSurface` | Vanilla surface rules still apply (snow layers, biome-correct top blocks) over the AI heightmap | Same, except inside structure footprints, where the structure compiler owns surface blocks |
| Vanilla structure generation (villages, ruins, etc.) | Left fully enabled — the model never reserves space | Disabled inside any AI-reserved footprint; enabled elsewhere |
| Structure placement | None — `structure_markers` stays empty as it is today | Driven by the structure compiler (§4) and the seeded placement scheme in "High Intensity Detail & Intricate Structures" |

This turns the currently-empty methods into real integration points instead of leaving AI output as a
complete replacement for vanilla generation — the AI *augments* rather than *replaces*, except where its
tier explicitly claims ownership (structure footprints).

### 4. Structure Compiler (`intricate_structures` tier)
An ONNX model realistically cannot emit "arbitrary working redstone" as raw voxels — there's no way to
guarantee the output is functionally correct (pistons wired to the right levers, hoppers pointed the right
way). The tractable version of "the AI designs an escape-room castle" is the same trick vanilla Minecraft
already uses for villages: **composition from a curated template library, with the AI choosing and arranging
pieces rather than authoring voxels from scratch — and a Wave Function Collapse pass guaranteeing the
composition is structurally valid**, not just hoping the model never proposes an invalid graph (see "High
Intensity Detail & Intricate Structures" for how WFC fits in).

- Three output tensors (only present for `structure_support: "intricate"` models — see
  `docs/model-spec.md` §"Output Tensors" for exact shapes): `structure_graph_nodes` (fixed-size, zero-padded
  room records — type, size class, loot tier, redstone template id), `structure_graph_edges` (adjacency
  matrix of connector types between rooms), and `structure_origin` (chunk-relative anchor point).
- Room-type ids and connector ids index into a **hand-authored library** of NBT structure templates and
  redstone contraption templates (piston doors, tripwire traps, dispenser puzzles) shipped with the mod —
  the same mechanism Minecraft's own structure/jigsaw system uses. The model's job is composition (which
  pieces, how many, how connected, what loot table per treasure room), not raw circuit design.
- A new `StructureCompiler` class walks the decoded graph, computes a bounding volume for the whole
  structure, reserves that footprint (feeding the "disabled" cells back into §3's carver/vanilla-structure
  skip logic via `StructureRegistry`), stamps each room's template at its graph-assigned offset, runs
  `detail_passes` decoration inside it, and wires loot tables/redstone templates per the room's assigned type.
- This is what lets `imaginator-high_intensity-intricate_structures` satisfy "hidden rooms... redstone
  escape-room puzzles... loot worth the effort" without requiring the model to understand redstone logic —
  it only needs to understand *narrative structure* (room types, connections, difficulty/reward pacing),
  which is a much more learnable task.

### 5. Tensor I/O Contract
The exact, versioned tensor shapes for all three possible graphs (`model.onnx`, `macro.onnx`, `detail.onnx`)
live in `docs/model-spec.md` §"Input Tensors" / §"Output Tensors" — this section only notes what changes in
`ModelSession` to consume the fuller contract (it currently does this ad hoc for a single graph: checking
`session.getInputNames()` for optional inputs, and permissively coercing whatever shape/dtype comes back in
`extractHeightmap`). Per-chunk inputs beyond the four already wired (`prompt_tokens`/`chunk_x`/`chunk_z`/`seed`):
`intensity_scale` (float scalar, present for `"medium"`/`"high"` intensity models), `structure_request` (int
flag, `"intricate"` structure support only, set from the macro-region's cached candidate-site data — see
"Structure Placement Determinism" below), and — iff `requires_macro_field` — `macro_local_height`,
`flavor_zone_ids`, `flavor_zone_weights`, all sliced by the mod from the cached `macro.onnx` output for the
chunk's containing region. New outputs beyond `heightmap`/`block_volume`/`biome_grid`: `structure_markers`
(`"basic"` structure support) or `structure_graph_nodes`/`structure_graph_edges`/`structure_origin`
(`"intricate"`), decoded by the `StructureCompiler` (§4).

Two things worth fixing while extending this: `block_volume` IDs should resolve via the manifest's
`block_palette` instead of the current hardcoded `switch` in `getBlockStateFromId`, and `biome_grid` is
currently extracted by `ModelSession` but never consumed by `ImagineChunkGenerator` — either wire it into a
custom `BiomeSource` or explicitly drop it for tier 1. Shape/range validation should also happen once per
tensor before use (clamp heights to `[minY, maxY]`, reject mismatched lengths) rather than only defensively
branching on Java type as `extractHeightmap` does today.

### 6. Coordinate System & Seam Handling — Decided
Every chunk is currently inferred independently — `generateChunk` has no awareness of neighboring chunks, so
adjacent 16×16 heightmaps can (and eventually will) disagree at the border, producing visible seams. This
used to be an open question between two options; **it's now decided**: the model can ship an optional
`macro.onnx` graph that produces a shared, low-resolution landform/style context per 32×32-chunk macro-region
(matching vanilla's own `.mca` region boundary), which every chunk inside that region conditions on. This
guarantees agreement at chunk borders by construction — a classic super-resolution pattern — and, critically,
is *lazily evaluable*, so it doesn't require pre-computing anything for a bounded area the way a naive
"generate the whole region upfront" approach would. Full design, including why this is preferred over the
weaker "just pass neighbor edge data" alternative, is in **"Runtime Generation Lifecycle"** below and
`docs/model-spec.md` §"Runtime Architecture".

### 7. Error Handling & Fallback Chain
Layer this rather than a single try/catch:
- **Load time** (`ModelLoader`): beyond today's manifest-parsing fallback (`parseModelInfo` already falls
  back to defaults on a corrupt/empty archive), run a one-time trial inference with dummy inputs when a model
  is discovered — for each graph it declares (`model.onnx` always, `macro.onnx`/`detail.onnx` if present) —
  and mark it unavailable in the model-selection UI rather than deferring failure to the first real chunk
  request.
- **Capability mismatch**: handled at prompt-validation time (§2), not as an error — unsupported clauses are
  stripped and warned about, generation proceeds.
- **Chunk time**: `ModelSession.generateFallbackChunkOutput` and `ImagineChunkGenerator.generateFallbackChunkOutput`
  currently duplicate the same sine-wave terrain fallback in two places — worth consolidating into one shared
  utility so the two callers (`ModelSession.inferChunk`'s catch block, and `ImagineChunkGenerator` when
  `modelSession` is null) can't drift apart. The same fallback should trigger if `macro.onnx` fails for a
  region — chunks in that region fall back to independent generation (losing coherence, not correctness).
- **Tensor validation**: reject/clamp out-of-range values (§5) before they reach block placement, rather than
  trusting raw model output the way `extractHeightmap`/`extractIntArray` currently do.

### 8. Performance & Caching
- `ImagineChunkGenerator.chunkCache` is currently an unbounded `ConcurrentHashMap<Long, ChunkOutput>` — every
  chunk ever generated in a session stays resident. Over a long play session or with players exploring far,
  this is an unbounded memory leak. Replace with a size- or radius-bounded evicting cache (e.g., Caffeine),
  keyed the same way, evicting chunks far from any loaded player. The planned `MacroCache`
  (region-coordinate → `macro.onnx` output) and `StructureRegistry` (candidate-site id → compiled structure)
  need the same bounding, at a much smaller entry count each since regions and structures are far sparser
  than chunks.
- `fillFromNoise` already dispatches per-chunk via `CompletableFuture`/`Executor` — extend this to **batch**
  multiple pending chunk requests into a single `session.run()` call with a batched leading dimension, which
  matters most on GPU execution providers where per-call launch overhead dominates for single-chunk batches.
- Speculative pre-generation: trigger `generateOrGetChunkOutput` ahead of actual chunk-load calls for chunks
  in a player's likely travel direction, so inference latency is hidden behind normal movement rather than
  causing a hitch on first load. The same applies one level up: `MacroCache` should speculatively warm the
  macro-region a player is approaching before they cross into it.
- The **only** eager generation work at world-creation time is an optional small-radius `MacroCache` warm-up
  around spawn (1–2 macro-regions) — see "Runtime Generation Lifecycle" for the full reasoning.
- Execution provider (CPU vs. CUDA/DirectML) should be a configurable setting given how widely Minecraft
  client hardware varies — not hardcoded to CPU as `OrtSession.SessionOptions()` defaults to today.

---

## Runtime Generation Lifecycle: Lazy Chunk Streaming

This section directly answers the question of *when* AI inference actually happens: **on the fly, in the
background, exactly as each chunk loads — never "the whole world" at world-creation time.** Here's the
reasoning and the full lifecycle.

### Why not generate the whole world at creation?
A Minecraft world is effectively unbounded — ±30,000,000 blocks per axis, on the order of 3.5 million chunks
per axis. Pre-generating that is not a performance tradeoff, it's simply impossible, and even a "reasonably
large" bounded pre-generation (say, a 10,000×10,000 block starting area) would spend enormous compute on
terrain the vast majority of players will never visit. Vanilla Minecraft has never worked this way — chunks
generate lazily as a player (or their loaded render distance) approaches, and are cached (to disk) once
generated, never regenerated. Mc-Imagine deliberately preserves this model rather than replacing it: AI
inference stands in for vanilla noise-based generation at the same point in the same lazy pipeline, not
before it.

### Why not naive independent per-chunk inference either?
This is what the pipeline does today, and it has a real, currently-undocumented cost: if every 16×16 chunk is
inferred with zero knowledge of its neighbors, there's nothing forcing a valley to actually continue as one
valley for 40 chunks, or a "dark fantasy" flavor zone to occupy a coherent patch of the map instead of
re-rolling its interpretation of "dark fantasy" every 16 blocks. Naive per-chunk inference is lazy in the
right way (never over-generates) but incoherent at any distance beyond one chunk.

### The resolution: two cache tiers, both lazy, neither "the whole world"
1. **Macro tier** (new): the first time any chunk in an unvisited 32×32-chunk macro-region is requested, the
   mod calls `macro.onnx` once for that region (if the model declares `requires_macro_field: true`) and
   caches the result in `MacroCache`. This is *itself* lazy — it never runs for a region nobody has
   approached — but because it operates at region scale, one inference call establishes shared context
   (landform, flavor-zone blend, candidate structure site) for up to 1,024 chunks at once.
2. **Chunk tier** (existing, extended): `fillFromNoise` calls `model.onnx` once per chunk, exactly as it does
   today, but now with a `macro_local_height`/`flavor_zone_ids`/`flavor_zone_weights` slice pulled from the
   already-cached macro-region result (a cheap array lookup, not a new inference call) mixed in. Result is
   cached in `chunkCache` as before.
3. **Structure tier** (new): if the chunk overlaps a candidate structure site (determined by pure seeded math
   against the cached macro-region's `structure_candidate` output — see "High Intensity Detail & Intricate
   Structures" below), `structure_request: 1` is set for that `model.onnx` call, `StructureGraphHead`
   resolves the room graph, and `StructureCompiler` compiles + caches it in `StructureRegistry`, keyed by
   candidate-site id, not by chunk — so every other chunk touching the same footprint (visited in any order)
   reads the cached compiled structure instead of re-invoking generation.
4. **Detail tier** (new): after the base chunk pass (and structure compilation, if applicable),
   `capabilities.detail_passes` additional lightweight passes layer in decoration, applied and discarded per
   chunk (not separately cached — cheap enough to redo if a chunk is ever regenerated, unlike the macro/chunk
   tiers which are expensive enough to be worth caching).

Every one of these four tiers is triggered by the same event: **a chunk being requested by Minecraft's own
chunk-loading system.** Nothing generates ahead of that trigger except the small optional spawn warm-up below.

### What actually happens at world-creation time
Deliberately minimal — just enough that the first few seconds of play don't stall:
1. Parse and tag the player's prompt; validate against the selected model's `capabilities.prompt_tags`,
   warning about (and stripping) anything unsupported (§"Model Selection & Prompt Validation" above).
2. Load the selected model's graph(s) into `ModelSession` — `model.onnx` always, `macro.onnx`/`detail.onnx`
   if the manifest declares them.
3. *Optionally*, warm `MacroCache` for the 1–2 macro-regions surrounding the spawn point (a handful of cheap
   macro-region inference calls, not per-chunk generation) — purely to avoid a cold-cache stall the moment the
   player takes their first few steps. This is bounded, small, and the *only* eager work in the whole system.

Nothing else. No chunks are generated. No structures are placed. The world "exists" in the sense that its
generation rules (prompt + seed + model) are fixed and deterministic, exactly like a vanilla world seed does
— but no terrain is computed until something asks for it.

### What happens on every subsequent chunk load, forever
1. Minecraft's chunk-loading system requests chunk `(x, z)` — from player movement, teleportation, spawn
   chunk keep-alive, whatever the normal trigger is.
2. `ImagineChunkGenerator.fillFromNoise` dispatches the request asynchronously (existing `CompletableFuture`/
   `Executor` pattern — this doesn't change).
3. The containing macro-region `(x >> 5, z >> 5)` is looked up in `MacroCache`; if absent, `macro.onnx` runs
   once for it and the result is cached (§ above — this is the only place a "new" region-scale inference ever
   happens, and it happens lazily, on first contact).
4. The chunk-local slice of that macro result, plus whether this chunk overlaps a candidate structure
   footprint, feed into a `model.onnx` call for chunk `(x, z)`, cached in `chunkCache`.
5. If a structure is triggered and not already in `StructureRegistry`, it's resolved and cached now (§ above).
6. `detail_passes` decoration layers on top.
7. Blocks are written into the chunk exactly as `fillFromNoise` does today.

This is, deliberately, structurally identical to how vanilla Minecraft's own noise-based generator works —
Mc-Imagine only changes *what function computes the terrain*, not *when* that function runs.

### An acknowledged alternative: bulk pre-generation as an opt-in tool, not the default
Server operators sometimes want a bounded area fully generated ahead of time for performance reasons (the
same motivation behind third-party tools like Chunky for vanilla worlds). Nothing above prevents building an
equivalent tool for Mc-Imagine later — walk a bounded region and force `fillFromNoise` for every chunk in it
— but it would be an **opt-in utility layered on top of the lazy pipeline**, not a replacement for it, and
it's explicitly out of scope for the core generation design: the lazy path above must work correctly and
efficiently on its own, since most players will never run a pre-generation tool.

---

## High Intensity Detail & Intricate Structures — Full Mechanics

This section is the detailed answer to "how does `intensity` and `structure_support` actually work inside the
model," walking both worked prompts — the dark-fantasy valley and the castle — through the full pipeline.

### Intensity: multi-pass decoration, not a bigger monolithic model
The hard constraint is that every tier has to run in real time, per chunk, on a player's own machine — so
"high intensity" can't mean "swap in a categorically bigger neural network," or the accessible `"low"` tier
and the demanding `"high"` tier would need incompatible hardware budgets, undermining the whole point of a
shared, tiered contract.

Instead, `capabilities.detail_passes` (`0`/`1`/`2` for `"low"`/`"medium"`/`"high"`) counts a fixed number of
**additive decoration passes** that run after the base `model.onnx` call:
- Each pass scatters micro-features — loose rock, roots, moss, cobwebs, rubble, hanging vines — onto blocks
  already placed by the base pass (or, inside a structure's footprint, by `StructureCompiler`). Passes never
  touch the base heightmap, block-volume shape, or room layout; they only dress what's there.
- A pass can be backed by a small optional third graph, `detail.onnx` (an ML-driven decorator), or by the
  mod's built-in procedural scatter routine seeded off chunk coordinates and driven by lightweight tags the
  base pass already output. A model author gets to choose; `detail_passes > 0` doesn't require shipping
  `detail.onnx`.
- Cost scales **linearly** with tier (each pass is cheap and independent) rather than requiring an
  exponentially larger base model — this is the concrete mechanism, not a hand-wave, behind "attention to
  detail" as a performance-bounded, shippable feature.

**Worked example (dark-fantasy valley)**: the base pass (conditioned on macro-region flavor-zone data, see
below) places the Blighted Moor's coarse dirt and sparse spruce, the Shadowbark Forest's dense dark oak
canopy, and so on. At `intensity: "high"`, the two detail passes then add per-zone ambient dressing — cobwebs
and dead undergrowth scattered through the Blighted Moor, glow lichen clusters in the Moonlit Fen — layered
on top without altering the terrain shape or zone boundaries the base pass and macro context already fixed.

### Structure_support: from "where" to "what" to "how it's dressed"
Three separate questions, resolved by three separate, ordered mechanisms:

**Where does a structure go?** Not learned by the model at all — determined by pure seeded math, the same
way vanilla places villages and strongholds. `macro.onnx`'s `structure_candidate` output is driven by
`hash(seed, region_x, region_z)` checked against `capabilities.structure_spacing_regions` (default `4`: on
average one candidate site every 4 macro-regions, ≈ every 128×128 chunks — deliberately rare). A second hash
jitters the candidate's exact position within the region. Because this is pure function-of-seed math with no
dependency on generation order, **any chunk, visited in any order, can independently compute whether it
overlaps a candidate footprint** — a player approaching a castle from its back wall, long before its
gatehouse chunk has ever been loaded, still correctly triggers footprint reservation and carver/vanilla
structure skipping.

**What gets built there?** The first chunk touching a candidate site sets `structure_request: 1` on its
`model.onnx` call. `StructureGraphHead` produces *preference weights* over the template library — which
rooms, roughly how many, what order of difficulty and payoff, given the prompt ("abandoned," "hidden,"
"escape-room," "worth the effort"). Those preferences drive a **Wave Function Collapse** constraint-
propagation pass over the template library's declared connector sockets, which resolves the actual room
graph (`structure_graph_nodes`/`_edges`/`_origin`) with a hard guarantee: every connection between rooms is
socket-valid by construction. This is the key correction from a naive design — a free-generation sequence
model *could* emit a room graph with a hidden door connecting to a socket that doesn't exist on the other
room, and there'd be no way to guarantee otherwise without WFC (or something equivalent) in the loop. The
neural net supplies *taste*; WFC supplies *correctness*.

For the castle prompt, this produces exactly the graph documented in `docs/model-spec.md`'s worked example:
gatehouse → hidden-door hall → trapped corridor → puzzle chamber → treasure vault, each edge a WFC-validated
connector, each `redstone_template_id` a tested, pre-built contraption the model selected but never designed.

**How is it dressed?** `StructureCompiler` stamps each room's NBT template at its graph-assigned offset, then
runs the same `detail_passes` mechanism used on open terrain *inside* the structure's volume — cobwebs and
dust in the gatehouse, scorch marks by the dart-volley trap, disturbed rubble in the vault. This is the entire
difference between a `"high"` intensity castle and a `"medium"` one: the room graph, template library, and
WFC validity guarantee are identical; only the decoration density changes.

**Caching, once, forever**: the compiled result (footprint + placed rooms + applied decoration plan) is
cached in `StructureRegistry` keyed by candidate-site id. `StructureGraphHead` and the WFC pass run exactly
once per structure, no matter how many chunks — or play sessions — later touch its footprint.

### Terrain flavor zones: the mechanism behind "multiple biomes, different flavors of one world"
This is what makes *"vast valleys with multiple biomes that contain different flavors of a beautiful, dark
fantasy type world"* achievable, and it's a macro-tier concern, not a per-chunk one:
- The valley macro-shape comes from `macro.onnx`'s `region_heightfield` — the prompt embedding biases the
  landform toward broad, continuous valley systems at low frequency, evaluated lazily per macro-region.
- "Multiple biomes, different flavors of one aesthetic" is `capabilities.flavor_zones`: a small, model-
  declared table of named style regions sharing one embedding-space aesthetic (Blighted Moor, Shadowbark
  Forest, Ashen Wastes, Moonlit Fen — see the worked example in `docs/model-spec.md`), each with its own
  `block_bias`. `macro.onnx` assigns every region a blend across up to 4 nearby zones
  (`flavor_zone_ids`/`flavor_zone_weights`), so zones form coherent, smoothly-blended patches across the
  valley — a player crossing from Shadowbark Forest into Ashen Wastes sees a gradient, not a hard cut, and
  never sees the same patch re-interpret "dark fantasy" differently 16 blocks later.
- `model.onnx` receives that blend as a chunk-local slice (`flavor_zone_ids`/`flavor_zone_weights` passthrough)
  and uses it to bias which blocks from `block_palette` get used where — the actual mechanism connecting "the
  macro pass decided this region leans 70% Shadowbark / 30% Blighted Moor" to "this specific chunk's
  `block_volume` is mostly dark oak with some coarse dirt at the edges."

---

## Model Training Pipeline (`model/` Package)

Everything above describes how a `.mcim` model gets *loaded and run*. This section covers how one actually
gets *made* — `model/src/mc_imagine_model/` already scaffolds the full pipeline as stub modules (data →
architecture → training → export); none of it is implemented yet, but the file layout itself encodes the
intended design and should be read as such. `docs/model-spec.md` is the contract every stage below has to
converge on — a training run is only useful if its output tensors match that spec exactly, across all three
possible graphs.

```
data/        chunk_extractor.py, dataset.py, labeler.py, world_generator.py
model/       text_encoder.py, positional.py, heads.py, imagine_net.py, macro_net.py (planned)
training/    losses.py, train.py, config.cuda.yaml, config.mps.yaml
export/      export_onnx.py, package_mcim.py
```

### Foundation Models & Architecture Decisions
Rather than designing every component from scratch, each piece is deliberately built on an established
foundation — see `docs/model-spec.md` §"Foundation Architecture" for the full rationale; summarized here
against the actual stub files:

- **`text_encoder.py` (`PromptEncoder`)**: replace the from-scratch `vocab_size: 50000` transformer stub with
  a frozen, pretrained MiniLM-class sentence encoder (the architecture family distributed as
  `sentence-transformers/all-MiniLM-L6-v2` — 6 layers, 384-dim, ~22M params). This is the single highest-
  leverage change in the whole pipeline: it means training data only has to teach the mapping from "meaning"
  to terrain/structure output, not language itself, directly reducing the data volume needed at every tier.
  Ruled out: anything LLM-scale — incompatible with the real-time-per-chunk constraint.
- **`heads.py` (`TerrainHead`/`BiomeHead`) and `imagine_net.py`/`macro_net.py`**: adopt a convolutional,
  single-forward-pass generator architecture in the lineage of **TOAD-GAN → World-GAN** (GAN research trained
  directly on Minecraft block-token grids), rather than inventing a decoder architecture from nothing.
  Diffusion-style architectures are explicitly ruled out — multi-step sampling is the wrong computational
  profile for a per-chunk, real-time workload; a single conditional forward pass is required.
- **`heads.py` (`StructureGraphHead`)**: outputs *preference weights* over the template library, not a raw
  graph — the actual graph is resolved by a Wave Function Collapse pass (not a stub file yet; belongs
  alongside the mod-side `StructureCompiler` in Java, or as a small Python reference implementation used
  during training-time validation) that guarantees connector-socket validity. This removes "never emit an
  invalid graph" from the list of things the neural net has to learn perfectly.

### `macro_net.py` — new stub, mirrors `imagine_net.py`
A new module for `MacroFieldNet`, the model backing `macro.onnx`: same `PromptEncoder` foundation as
`ImagineNet`, conditioned on `(region_x, region_z)` instead of `(chunk_x, chunk_z)`, with output heads for
`region_heightfield`, `flavor_zone_ids`/`flavor_zone_weights`, and `structure_candidate`. Only relevant for
models declaring `requires_macro_field: true` — the low-intensity/no-structures reference model never trains
or exports this graph at all.

### Data: sourcing `(prompt, region)` pairs, by capability tier
A training example is a text prompt paired with ground-truth output tensors for a whole macro-region (not a
single 16×16 chunk — training on isolated chunks is why cross-chunk seams would never get learned away, and
now that macro-region is a defined 32×32-chunk unit rather than an arbitrary size, it's the natural training
unit too). Four sourcing strategies map onto the existing stub files:

1. **Procedural bootstrap** — `world_generator.py` (headless MC server, varied seeds) + `chunk_extractor.py`
   (parses `.mca` region files into tensors — note this already aligns 1:1 with the macro-region unit, since
   both are `.mca`-sized) generate terrain with known parameters; `labeler.py` then auto-captions from those
   parameters ("high-frequency noise, biome=plains" → templated sentence). Cheap, unlimited volume, low
   linguistic diversity — the right way to bootstrap the `low_intensity` terrain tier.
2. **Mined existing builds** — the same `chunk_extractor.py` path, pointed at community schematics/curated
   build collections instead of freshly generated worlds, captioned by a human or an auxiliary
   vision-language model against rendered screenshots. More creative variance than pure procedural.
3. **Human-authored paired data** — a builder constructs a region to match a written prompt. Most expensive
   per example, and the one the `intricate_structures` tier can't skip: someone has to actually design
   escape-room pacing (room order, difficulty curve, payoff placement), not just place blocks. This is
   *labeling*, not extraction, so it doesn't map to an existing stub — it's a manual authoring workflow that
   feeds into `McImagineDataset` in the same tensor format as the other three sources.
4. **Caption paraphrasing** — an LLM-generated paraphrase pass over captions from (1)/(2), configured via
   the training config's `data.augmentation` flag, so the model learns to generalize prompt language instead of
   pattern-matching literal keywords from templated captions.

`McImagineDataset.__getitem__` (`data/dataset.py`) is the convergence point — its returned dict already
lists `structure_markers`/`structure_graph_nodes`/`structure_graph_edges`/`structure_origin`/`capability_tier`
alongside the terrain tensors, and should grow `region_heightfield`/`flavor_zone_ids`/`flavor_zone_weights`/
`structure_candidate` fields to match `docs/model-spec.md`'s `macro.onnx` contract once `requires_macro_field`
models start training.

### Architecture: `ImagineNet` and its heads
`model/imagine_net.py`'s TODOs already name the right components — `PromptEncoder` (text →
conditioning vector, now a MiniLM-class foundation per above), `CoordinateEncoder`/`SeedEncoder` (chunk
position + stochastic variation), and per-tensor output heads (`model/heads.py`: `TerrainHead`, `BiomeHead`,
`StructureHead`, `StructureGraphHead`). Two things worth calling out:
- `SeedEncoder` has to genuinely map `seed` to a noise/style vector the decoder conditions on — right now
  even the *dummy* model ignores its seed input entirely, and a trained model inheriting that would break
  the "same seed ⇒ same result, different seed ⇒ variation" guarantee.
- Model size is bounded by "must run per-chunk, in real time, on a player's own machine" — that's the reason
  `manifest.json`'s `requirements.min_ram_mb`/`recommended_vram_mb` fields exist. Budget that constraint before
  picking layer widths, not after training something too large to ship in a `.mcim`. The multi-pass detail
  system exists specifically so `"high"` intensity doesn't blow this budget the way a bigger monolithic model
  would.

### Training & Export
`training/losses.py` already has the right shape — `TerrainLoss`, `BiomeLoss`, `StructureLoss`,
`StructureGraphLoss` combined by weight in `CombinedLoss` — and needs a `MacroFieldLoss` (regression over
`region_heightfield` + classification over `flavor_zone_ids`/weights) once `macro_net.py` exists.
`training/train.py` and `training/config.{cuda,mps}.yaml` wire dataset → model → loss → optimizer; `export/export_onnx.py`
mirrors `model/generate_dummy_model.py`'s already-working `torch.onnx.export` call (same `input_names`/
`output_names` convention), just against a real checkpoint instead of a stub `nn.Linear` — and now needs to
export up to three separate graphs (`model.onnx`, `macro.onnx`, `detail.onnx`) per training run instead of one.

`export/package_mcim.py` builds a manifest conforming to `docs/model-spec.md` and writes a working ZIP
archive — this was fixed as part of this pass and needs a further update to handle the now-nested
`io.chunk`/`io.macro`/`io.detail` schema and the new capability fields (`requires_macro_field`,
`detail_passes`, `structure_spacing_regions`, `flavor_zones`), and to optionally bundle a second/third ONNX
file when the caller provides them.

### The Two Benchmarks
Both worked examples in `docs/model-spec.md` are concrete, named targets — not abstract aspirations — and
both belong to the same flagship model, `imaginator-high_intensity-intricate_structures`:

- **The Dark Fantasy Valley benchmark**: a prompt like *"vast valleys with multiple biomes that contain
  different flavors of a beautiful, dark fantasy type world"* produces a continuous valley landform (via
  `macro.onnx`'s `region_heightfield`) hosting spatially coherent, smoothly-blended flavor zones (via
  `flavor_zone_ids`/`flavor_zone_weights`) — done when a player can walk for hundreds of blocks through a
  single valley system and see Blighted Moor blend into Shadowbark Forest blend into Ashen Wastes without a
  single jarring chunk-boundary discontinuity. Prerequisites: M13 (base architecture), M18 (flavor-zone
  training).
- **The Castle benchmark**: a prompt like *"castles in deep forests, abandoned, with hidden rooms, redstone
  escape-room puzzles, and loot worth the effort"* produces a room graph that `StructureCompiler` lays out
  using the M14 template library, WFC-validated, in-game, end to end, discoverable correctly regardless of
  which direction the player approaches from. Prerequisites: M13, M14, M15, M16.

Together they exercise every mechanism this document describes: macro-region landform and flavor blending,
lazy per-chunk streaming, seeded structure placement, WFC-constrained composition, and multi-pass detail
dressing.
