# Mc-Imagine PoC Plan — `imaginator-low_intensity-no_structures`

**Goal:** a player types a prompt in the Mc-Imagine world-creation UI, presses Create, and explores a
world whose terrain and biomes were produced by a *trained neural network* responding to that prompt.
Same prompt + seed + model ⇒ same world. No structures tier.

**Scope target (from `docs/model-spec.md`):** the `imaginator-low_intensity-no_structures` reference
model exactly — `intensity: "low"`, `structure_support: "none"`, `requires_macro_field: false`,
`detail_passes: 0`. One graph (`model.onnx`). No `macro.onnx`, no `detail.onnx`, no
`StructureCompiler`, no WFC, no template library.

**Milestone mapping (`PROJECT.md`):** completes M5, M6, M7, M8, M12 (procedural half), M13.
Explicitly out of scope: M9, M10, M11 (partial), M14–M19.

---

## 0. Where the code actually is today

Verified by reading every source file. The pipeline "works" in the M1–M4 sense but nothing about it is
prompt-driven yet:

| Claim in `PROJECT.md` | Reality |
|---|---|
| Prompt drives generation | **The prompt reaches nothing.** `McImagineCustomizeScreen` writes it to the static `McImagineCreateWorldMode.currentPrompt` and it is never read again. `ImagineChunkGenerator.promptTokens` is an all-zeros `int[128]` that nothing ever sets. |
| Model is selected by the user | `ImagineChunkGenerator.loadFromDirectory` picks `dummy-test-v1`, or `models.get(0)`, ignoring `selectedModelPath` entirely. |
| Generation is seeded | `ImagineChunkGenerator.seed` is `0L` and nothing sets it. The dummy ONNX graph ignores its `seed` input anyway. |
| Settings persist in the world | `CODEC` serialises **only** `biome_source`. On world reload the generator is rebuilt with no prompt, no model, no seed. |
| `ModelInfo` carries capabilities | `ModelInfo` is 7 strings. No `capabilities` block, no palette, no tier. |
| Tokenizer exists | `PromptTokenizer` is an empty class with a comment. |
| `.mcim` conforms to spec | `generate_dummy_model.py` writes `format_version: "0.1.0"`, flat `io.input_names` (spec wants nested `io.chunk`), `structures: false` (spec wants `structure_support`), and emits only `heightmap` as `float32` — no `block_volume`, no `biome_grid`. |
| Training pipeline | Every module in `model/src/mc_imagine_model/` is a `# TODO` stub returning `torch.zeros(0)`. Nothing trains, nothing exports. |

Plus four latent correctness bugs that will bite the moment a real multi-output model is loaded:

1. `ModelSession.generateChunk` — `if ("heightmap".equalsIgnoreCase(name) || !foundHeightmap)` parses the
   **first output of any name** as the heightmap. With a 3-output model, `block_volume` gets read as the
   heightmap. Must key strictly on tensor name.
2. `ModelSession.extractIntArray` handles only flat 1-D `int[]`/`float[]`. ONNX Runtime returns
   `int[16][384][16]` (nested) for `block_volume` — this silently yields `new int[0]`, and the mod falls
   back to `buildBlockVolumeFromHeightmap`. The AI's block volume would never actually be used.
3. `ModelSession.close()` calls `env.close()`. `OrtEnvironment.getEnvironment()` is a process-wide
   singleton; closing it breaks every other session in the JVM.
4. `fillFromNoise` never updates the `ProtoChunk` heightmaps (`WORLD_SURFACE_WG`, `OCEAN_FLOOR_WG`), never
   places water, and never writes air. Vanilla decoration and structure placement read those heightmaps.

There are also 25 zero-byte `.mcim` files in `mod/fabric/run/mcimagine/models/` left over from UI testing.

---

## 1. Two architectural decisions that make this tractable

These are the load-bearing calls. Everything downstream follows from them.

### 1a. `ImagineChunkGenerator` must extend `NoiseBasedChunkGenerator`, not `ChunkGenerator`

You chose the **full §3 `none` row** — vanilla carvers, vanilla surface rules, and vanilla structures all
running on top of AI terrain. Implementing `applyCarvers` from scratch is effectively blocked: 1.20.1's
`CarvingContext` constructor requires a `NoiseBasedChunkGenerator` *instance*, and `SurfaceSystem.buildSurface`
requires a `NoiseChunk`. Reimplementing both is days of fighting MC internals.

Extending `NoiseBasedChunkGenerator(BiomeSource, Holder<NoiseGeneratorSettings>)` and overriding **only**
`fillFromNoise`, `getBaseHeight`, `getBaseColumn`, and `codec()` gives us the entire §3 `none` row for free:

- `applyCarvers` — inherited. Vanilla caves/ravines carve the AI stone volume. ✔
- `buildSurface` — inherited. `SurfaceSystem` walks each column top-down off `chunk.getBlockState`, so it
  dresses **whatever blocks the AI wrote**, biome-correctly (snow layers, sand shores, podzol). ✔
- `spawnOriginalMobs` — inherited (delete the current no-op override). ✔
- `createStructures` / `applyBiomeDecoration` — already inherited from `ChunkGenerator`; villages, ores,
  trees and grass place onto AI terrain once the heightmaps are correct (fix #4 above). ✔

**Consequence for the model:** the model should emit a *raw* volume — stone / deepslate / water / bedrock —
and let vanilla surface rules paint the top per biome. `SurfaceSystem` only rewrites columns whose block
equals the settings' `defaultBlock` (stone), so a model that deliberately emits e.g. `red_sandstone` keeps
its override, while stone columns get biome-appropriate dressing. This is strictly better-looking than the
current hardcoded grass/dirt and costs the model nothing.

The world preset JSON gains `"settings": "minecraft:overworld"`.

### 1b. Cross-chunk seamlessness without `macro.onnx`, via global-coordinate conditioning + halo

`model-spec.md` says the low tier gets continuity from "the terrain being intentionally conservative
(small heightmap deltas) rather than active coherence machinery." Taken literally that produces visible
seams, because a conv decoder emitting an independent 16×16 tile per chunk has no reason to agree at borders.

The fix costs nothing at runtime and needs no second graph: make the network an **implicit field over global
block coordinates**. Inside `model.onnx`:

1. Build a `(16+2R)×(16+2R)` grid of **global** block coordinates `(X, Z)` from `chunk_x`/`chunk_z` (`R = 4`).
2. Encode each cell as multi-octave Fourier features — deterministic, no learned params.
3. Run a stack of **valid-padded** 3×3 convs (no zero padding), shrinking `24×24 → 16×16`.

Because every input value is a pure function of the global coordinate and no zero-padding is ever introduced,
the output for a border column is bit-identical regardless of which chunk requested it. Seams are impossible
by construction, and the decoder is still the convolutional single-forward-pass generator
`docs/model-spec.md` §"Foundation Architecture" asks for (TOAD-GAN/World-GAN lineage, explicitly not diffusion).

---

## 2. Phases

Each phase has an explicit verification gate. Do not start the next phase until the gate passes.

### Phase 0 — Baseline bring-up (M5)

Nothing here is new code; it establishes that the existing pipeline runs at all.

- `rm mod/fabric/run/mcimagine/models/*.mcim` except `dummy-test-v1.mcim` (25 zero-byte files).
- `cd mod && ./build.sh` — confirm Gradle resolves, Java 17 toolchain works, `onnxruntime:1.19.2` downloads.
- `./launch.sh` — create a world via the Mc-Imagine preset, confirm the dummy sloped terrain appears.
- Fix bug #3 (never close `OrtEnvironment`).
- `mod/fabric/build.gradle`: add `include(implementation("com.microsoft.onnxruntime:onnxruntime:1.19.2"))`
  so a built jar carries the runtime, not just the dev classpath.

**Gate:** game launches, world generates from `dummy-test-v1.mcim`, F3 shows `Model: Dummy Test Model v1.0.0`.

### Phase 1 — Spec amendments first (`docs/model-spec.md` → 0.5.0)

`model-spec.md` states: *"If a training or loading change requires a shape/field change, update this file
first and bump `format_version`."* Three things are genuinely underspecified and both sides need them pinned
before either is written:

1. **Axis order.** `heightmap: int32[16,16]` doesn't say `[z][x]` or `[x][z]`. The mod reads
   `heightmap[z*16 + x]`, so pin it as **`[z][x]`**. Pin `block_volume` as **`[x][y][z]`** with
   `y` indexed from `minY = -64`. Pin `biome_grid` as **`[x][y][z]`** at quart resolution. A transposed
   world is otherwise the most likely silent failure in the whole project.
2. **`capabilities.biome_palette`** (new, optional): `List[String]` of namespaced biome ids that
   `biome_grid` indexes into — the exact mirror of the existing `block_palette`. Without it there is no way
   to consume `biome_grid`, which §5 of `PROJECT.md` flags as currently-extracted-never-used.
3. **Tokenizer.** State that `tokenizer.json` for this reference model is BERT-family WordPiece
   (uncased, `[PAD]=0`, `[CLS]=101`, `[SEP]=102`), matching the MiniLM foundation §"Foundation
   Architecture" already mandates.

Bump `format_version` to `"0.5.0"`, update both worked examples, and add a `## Versioning` note that 0.4.0
manifests remain loadable (the loader defaults every new field).

**Gate:** `docs/model-spec.md` is self-consistent and `PROJECT.md`'s references to it still hold.

### Phase 2 — Mod: capabilities, palettes, tokenizer, prompt plumbing (M6, M7)

This is the phase that makes the prompt *matter*. Files:

**`api/ModelInfo.java`** — add a `ModelCapabilities` record mirroring the manifest field-for-field
(`intensity`, `structureSupport`, `promptTags`, `blockPalette`, `biomePalette`, `maxPromptTokens`,
`requiresMacroField`, `detailPasses`), a `capabilities()` accessor, and a `formatVersion()`. Defaults for
legacy manifests: `intensity="low"`, `structureSupport="none"`, `requiresMacroField=false`, `detailPasses=0`.

**`model/ModelLoader.java`** — parse the nested `capabilities` block with the existing defensive `getString`
pattern extended to arrays and nested objects. Also parse `io.chunk.input_names`/`output_names` and validate
them against the actual `OrtSession` names at load time. Per §7 "Load time": run **one trial inference with
dummy inputs** when a model is discovered, and mark it unavailable in the UI rather than deferring the
failure to the first chunk.

**`prompt/PromptTokenizer.java`** — implement WordPiece in pure Java (~200 lines), reading `vocab` out of
`tokenizer.json` inside the `.mcim`. Lowercase → strip accents → split on whitespace and punctuation →
greedy longest-match-first with `##` continuations → `[CLS] … [SEP]` → pad to `max_prompt_tokens` with `0`.
No native dependency, fully deterministic. **Validated by a golden-vector test** (Phase 6): Python writes
~50 `(prompt, token_ids)` pairs into `mod/common/src/test/resources/`, the Java test asserts exact equality.
A tokenizer mismatch here is invisible at runtime and would quietly destroy prompt fidelity.

**`prompt/PromptParser.java` / `ParsedPrompt.java`** — the M7 tagger. Regex/keyword rules extracting
`terrain`, `biome_blend`, `structures`, `loot`, `redstone`. Tags not in `capabilities.prompt_tags` are
**stripped and warned about, never a hard failure** (§2). For this tier that means "castles with hidden
rooms" is dropped with a UI warning and the terrain clause still generates.

**`model/BlockPalette.java` (new)** — resolves `block_volume` ids through
`capabilities.block_palette` via `BuiltInRegistries.BLOCK`, replacing the hardcoded 8-case `switch` in
`getBlockStateFromId`. Index `0` is always air; unknown ids fall back to stone with a one-time warn.
Same shape for `BiomePalette`.

**`model/ModelRegistry.java` (new)** — process-wide `Map<String, ModelSession>` keyed by model id, so the
`ModelSession` is loaded once and shared rather than reconstructed per generator. Closed on server stop.

**UI plumbing — the critical path:**
- `McImaginePresetEditor` currently drops the `WorldCreationContext` argument. Pass it into
  `McImagineCustomizeScreen`.
- On **Done**, build a new `ImagineChunkGenerator` carrying the prompt string and model id, and install it
  into the overworld `LevelStem` via `parent.getUiState().updateDimensions(...)` /
  `WorldCreationContext.withSettings(...)`.
- Add `prompt` (String) and `model_id` (String) to `ImagineChunkGenerator.CODEC` alongside `biome_source`
  and `settings`. **This is what makes the prompt persist into `level.dat` and survive a world reload** —
  and it is exactly what `project-outline.md` §Phase 2 means by "save prompt + model + seed in world save
  data".
- Show the M7 warnings in the customize screen before Done becomes active.
- Read the seed from `RandomState.legacyLevelSeed()` inside `fillFromNoise` rather than plumbing it
  separately; it is already correct and available there.

> The exact 1.20.1 signatures for `WorldCreationUiState.updateDimensions` and `WorldCreationContext.withSettings`
> need confirming against the Mojang mappings at implementation time — this is the one place in the plan
> where the API surface is asserted from memory rather than read out of this repo.

**Gate:** with the dummy model still loaded, a `System.out` in `ModelSession.generateChunk` prints the
tokenized user prompt; the prompt survives quit-and-reload of the world.

### Phase 3 — Mod: generator rearchitecture (M8)

**`generation/ImagineChunkGenerator.java`** — extend `NoiseBasedChunkGenerator` per §1a. Delete the
`applyCarvers`/`buildSurface`/`spawnOriginalMobs` no-op overrides so vanilla runs. Rewrite `fillFromNoise` to:

- decode `block_volume` with correct `[x][y][z]` → section indexing (bug #2), reading via
  `OnnxTensor.getByteBuffer()` rather than materialising a nested `int[16][384][16]` per chunk;
- write through `LevelChunkSection` with `acquire()/release()` instead of 98,304 individual
  `chunk.setBlockState` calls;
- fill air above the surface and water up to sea level;
- **update `WORLD_SURFACE_WG` and `OCEAN_FLOOR_WG` heightmaps** (bug #4) — without this, vanilla decoration
  and structure placement land in the wrong place or not at all;
- clamp every height to `[minY, maxY]` and reject mismatched tensor lengths before use (§5, §7
  "Tensor validation"), rather than trusting raw model output.

**`model/ModelSession.java`** — strict name-keyed output dispatch (bug #1); shared fallback utility so
`ModelSession` and `ImagineChunkGenerator` stop duplicating the sine-wave terrain (§7).

**`generation/ImagineBiomeSource.java` (new)** — a `BiomeSource` reading `biome_grid` out of the same chunk
cache and mapping ids through `capabilities.biome_palette`. This is what makes a "scorching desert" prompt
actually *read* as desert — grass tint, foliage, mob spawns, and the vanilla surface rules from §1a all key
off biome. Codec-serialisable so it round-trips through `level.dat`. Note the ordering: MC resolves biomes
at the BIOMES stage before NOISE, so the first inference for a chunk happens there and `fillFromNoise`
reads the cache — one inference per chunk either way.

**`generation/ChunkCache.java`** — replace the unbounded `ConcurrentHashMap` with a bounded LRU (§8 / M11's
cheap half). A few thousand entries is plenty; the current map is an unbounded leak over a long session.

**Gate:** dummy model still generates a world, now with vanilla caves, biome-correct surfaces, trees, ores
and villages layered on top. Fly 500 blocks in a straight line — no crash, no missing chunks.

### Phase 4 — Python: procedural data pipeline (M12, procedural half)

Per your answer: procedural first, `.mca` second (Phase 8).

**`data/world_generator.py`** — `ProceduralWorldSource`, implementing the documented `WorldGenerator`
interface but synthesising terrain directly in numpy instead of driving a headless server. Samples a
parameter vector per macro-region — `base_height`, `relief_amplitude`, `relief_frequency`,
`ridge_sharpness` (valley ↔ mountain), `plateau_quantization`, `water_level`, `erosion`,
`surface_profile_id`, `biome_class` — and renders a 512×512 heightfield + per-column profile + biome map
(fBm / ridged-multifractal / domain-warped noise). Emits `.npz` shards, one per macro-region, aligned 1:1
with the `.mca` region unit `PROJECT.md` calls the natural training unit.

**`data/labeler.py`** — `ChunkLabeler.label_region` turning the *known* parameter vector into templated
captions: high amplitude + ridged + high water → *"vast deep valleys, beautiful mountains overlooking
water-filled craters."* Build ~40 templates × synonym slots ⇒ thousands of surface forms. Wire
`config.yaml`'s existing `data.augmentation` flag to an optional paraphrase pass (strategy 4).

**`data/dataset.py`** — implement `McImagineDataset.__getitem__` for real: load shard, slice a 16×16 chunk
(plus the 4-cell halo), tokenize the caption with the *same* HF tokenizer the mod's Java WordPiece mirrors,
return the dict already declared in the stub. Leave `McImagineMacroDataset` stubbed (not this tier).

Target ~200k chunk samples across ~2k parameter draws.

**Gate:** `python -c "from mc_imagine_model.data.dataset import McImagineDataset; d = McImagineDataset('data/'); print(len(d), d[0]['heightmap'].shape)"` prints a real length and `(16, 16)`. Render 20 sampled
heightmaps to PNG and eyeball that captions match what you see.

### Phase 5 — Python: ImagineNet + training (M13)

Sized for Apple Silicon MPS: frozen MiniLM (~22M) + a ~8M-param decoder, 1–3 hours to train.

**`model/text_encoder.py`** — `PromptEncoder` wrapping frozen `sentence-transformers/all-MiniLM-L6-v2`
(6 layers, 384-dim), mean-pooled over the attention mask, L2-normalised. Replaces the `vocab_size: 50000`
from-scratch stub. Add `transformers` to `requirements.txt`; vendor the checkpoint under
`model/checkpoints/` so training and export are offline-reproducible.

**`model/positional.py`** —
- `CoordinateEncoder`: multi-octave Fourier features of **global block** `(X, Z)` at λ ∈ {2048…8} → 36
  deterministic channels. This is the mechanism from §1b.
- `SeedEncoder`: `seed → sin(seed·aᵢ + bᵢ)` over fixed constants → `Linear(32→64)`. Must genuinely vary
  with seed — `PROJECT.md` explicitly calls out that even the dummy ignores its seed and that a trained
  model inheriting that would break the "same seed ⇒ same result, different seed ⇒ variation" guarantee.

**`model/imagine_net.py`** — fuse `concat(prompt 384, seed 64) → MLP → 128`, broadcast across a 24×24
coordinate-feature grid, 4× valid-padded 3×3 conv + GELU at 128 channels, crop to 16×16.

**`model/heads.py`** — `TerrainHead` → heightmap (1ch, `63 + 96·tanh`) + column-profile logits (8 classes)
+ per-column water level. `BiomeHead` → biome logits over the 8–12 palette classes at 4×4. Leave
`StructureHead`/`StructureGraphHead` stubbed — `structure_support: "none"`.

**`training/losses.py`** — `TerrainLoss` = MSE(heightmap) + CE(profile); `BiomeLoss` = CE(biome);
`CombinedLoss` with `structure_weight = structure_graph_weight = 0`. Exactly the shape the stubs declare.

**`training/train.py` / `config.yaml`** — AdamW 3e-4, batch 64, cosine schedule, `device: "mps"`,
checkpointing, a held-out validation split, and a per-epoch qualitative dump (render 8 fixed prompts to PNG).

**Gate — this is the gate that decides whether the PoC is real:** hold out 6 prompts spanning the parameter
space ("vast deep valleys…", "endless flat desert dunes", "towering snow-capped peaks", "shallow tropical
archipelago", "eroded red mesa plateaus", "gentle rolling grassland"). Render each at fixed seed. They must
be *visibly, unmistakably different from each other*, and each must match its caption. If they all look the
same, the model has collapsed to the caption prior and no amount of mod work will save the PoC — go back to
data diversity before proceeding.

### Phase 6 — Python: ONNX export + `.mcim` packaging

**`export/export_onnx.py`** (`--graph chunk` path only). The exported graph takes exactly the four spec'd
inputs and emits exactly the three spec'd outputs, doing all of the following *inside* the graph:

- run frozen MiniLM on `prompt_tokens` (attention mask derived as `tokens != 0`, valid because `[PAD] == 0`);
- build the global-coordinate halo grid from `chunk_x`/`chunk_z`;
- run the decoder, producing heightmap `[16,16]`, profile ids, water level, biome grid;
- **expand to `block_volume` `int32[16,384,16]` with `Where` ops** over a broadcast y-grid: bedrock at
  `minY`, stone/deepslate below `h−4`, profile subsurface in `(h−4, h)`, profile surface at `h`, water in
  `(h, water_level]`, air above. Keeping expansion in-graph means the tensor contract stays exactly
  spec-conformant while the *network* only ever predicts ~350 numbers per chunk instead of 98,304.

Then `onnx.checker` + an ORT round-trip asserting shapes, dtypes, **and** determinism (same inputs twice ⇒
identical bytes), plus a border-continuity assertion: chunk `(0,0)`'s last column must equal chunk `(1,0)`'s
first column, exactly. Also emit the golden tokenizer vectors for the Java test here.

**`export/package_mcim.py`** — already ~90% correct. Update it for the 0.5.0 schema: emit `biome_palette`,
and correct `format_version`. Invoke it to produce
`imaginator-low_intensity-no_structures-1.0.0.mcim` containing `model.onnx`, `tokenizer.json`, `manifest.json`
with the exact `capabilities` block from `model-spec.md`'s worked example.

**Gate:** `ort.InferenceSession` loads the `.mcim`'s `model.onnx`, all three outputs have spec shapes/dtypes,
the determinism and border-continuity assertions pass. Expected size ~120 MB (MiniLM fp32 dominates);
int8-quantising the encoder is a documented later optimisation, not PoC work.

### Phase 7 — End-to-end integration and verification

Drop the `.mcim` into `mod/fabric/run/mcimagine/models/`, `./launch.sh`, and walk the success criteria from
`project-outline.md` §9 that are in scope for this tier:

1. Select model → type *"vast deep valleys, beautiful mountains overlooking water-filled craters"* → set
   seed `12345` → Create. Terrain matches the prompt.
2. Same prompt + same seed twice ⇒ identical worlds (compare block reads at a fixed coordinate set).
3. Different prompt, same seed ⇒ visibly different world. Run all 6 held-out prompts.
4. Different seed, same prompt ⇒ same character, different layout (proves `SeedEncoder` works end to end).
5. Fly 1000 blocks — **no seams at chunk borders** (the §1b guarantee, verified in-game rather than only
   in the export assertion).
6. Vanilla caves, biome surfaces, trees, ores and villages all present on AI terrain (the §3 `none` row).
7. Quit to title, reload the world — prompt and model persisted, terrain identical.
8. **Measure inference latency** and log it to F3. Budget is `<100 ms/chunk` (`project-outline.md`
   §"Model Size & Performance Targets"). Expect ~30 ms — MiniLM at seq-128 on CPU is ~15–25 ms of that.

If MiniLM turns out to dominate: the correct fix is a future `format_version` adding an optional
`prompt_embedding` input so the mod encodes once per world instead of once per chunk. That is a spec change
and therefore explicitly *not* PoC work — record it, don't do it.

**Then update the docs to match reality:** flip M5–M8, M12, M13 to DONE in `PROJECT.md`'s milestone table,
and correct `README.md` (it currently says models go in `.minecraft/mcimagine_models`, but both the UI and
the loader use `mcimagine/models`, and it describes a `pipeline/` directory that does not exist — the
Python package is `model/`).

### Phase 8 — Follow-on: real `.mca` as a second data source (M12 proper)

Once the PoC is demonstrably working, implement `data/world_generator.py`'s headless-server path and
`data/chunk_extractor.py`'s `.mca` → tensor path, feeding `McImagineDataset` through the same interface.
Retrain on real vanilla terrain distributions and compare against the procedural baseline on the same 6
held-out prompts. This is a strict improvement to an already-working system rather than a dependency of it.

---

## Deliverables

| Path | Status |
|---|---|
| `docs/model-spec.md` | amended → `format_version: "0.5.0"` |
| `mod/.../api/ModelInfo.java` + `ModelCapabilities` | rewritten |
| `mod/.../model/{ModelLoader,ModelSession}.java` | rewritten |
| `mod/.../model/{ModelRegistry,BlockPalette,BiomePalette}.java` | new |
| `mod/.../prompt/{PromptTokenizer,PromptParser,ParsedPrompt}.java` | implemented |
| `mod/.../generation/{ImagineChunkGenerator,ChunkCache}.java` | rewritten |
| `mod/.../generation/ImagineBiomeSource.java` | new |
| `mod/.../ui/{McImaginePresetEditor,McImagineCustomizeScreen}.java` | prompt/model plumbing |
| `model/src/mc_imagine_model/data/{world_generator,labeler,dataset}.py` | implemented |
| `model/src/mc_imagine_model/model/{text_encoder,positional,imagine_net,heads}.py` | implemented |
| `model/src/mc_imagine_model/training/{losses,train,config.yaml}` | implemented |
| `model/src/mc_imagine_model/export/{export_onnx,package_mcim}.py` | implemented |
| `imaginator-low_intensity-no_structures-1.0.0.mcim` | **the deliverable** |

## Explicitly not in this PoC

`macro.onnx`, `detail.onnx`, `MacroCache`, `StructureRegistry`, `StructureCompiler`, WFC, the structure
template library, `structure_graph_*` / `structure_markers` tensors, flavor zones, detail passes, batched
inference, speculative pre-generation, GPU execution providers, and the `.mca` extractor (Phase 8).
Both `docs/model-spec.md` benchmarks (Dark Fantasy Valley, Castle) belong to
`imaginator-high_intensity-intricate_structures` and are out of scope by definition.
