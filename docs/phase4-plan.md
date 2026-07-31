# Mc-Imagine Phase 4 Plan — Volumetric Terrain, and the v1.1.2 Release

**Goal:** break the heightfield ceiling. Every release so far has predicted *one height per column*
and expanded it into blocks with a fixed rule, which makes overhangs, arches, undercut cliffs and
walk-in cave mouths not merely unimplemented but **unrepresentable**. This release changes the
representation so the model authors actual 3D surface geometry, and then trains it to.

**Definition of done:** a player spawns into a `.mcim` stamped `1.1.2`, walks to a cliff, and
**stands underneath it** — sky above, rock overhead, ground below — in terrain no heightmap can
produce. Captured in `Imgs/` alongside the existing baselines, with `capabilities.caves: true` in
the manifest for the first time.

**On the phase number:** the `docs/*-plan.md` series is the numbering the team actually uses
(`phase2-plan` → `phase3.1-plan` → this). It does *not* line up with `docs/project-outline.md` §4,
whose "Phase 4" is the entire model-research effort. `phase3.1-plan.md` §0 already flagged that
drift; this document does not try to fix it, it just declines to add to the confusion.

---

## 0. Where this sits

### 0.1 What v1.1.1 owes this release

`docs/phase3.1-plan.md` is in flight. Its Gate 1 (per-term loss logging, rehearsal config fix,
`--version` enforcement, sweep script) landed on the dev machine — see `docs/CHANGELOG.md`'s
`[Unreleased]` block — but Gates 2–4 (diagnosis, the `relief_weight` sweep, the full run, the
release) need the CUDA box and have not happened.

**This matters concretely, not just administratively.** The occupancy band designed in §1.2 is
*anchored to the predicted heightmap*. If the heightmap is terraced, the overhangs carved relative
to it are terraced too, and the headline screenshot is a staircase with a hole in it. Phase 3.1
§2.1's terracing diagnosis and whatever smoothness term it produces are load-bearing inputs here.

**Sequencing consequence, and it's a favourable one:** the GPU is the bottleneck and §7's Gate 0 and
Gate 1 need no GPU at all. Build the 3D ground truth and prove the plumbing *while* the 1.1.1 sweep
and full run occupy the CUDA box. Do not start Gate 2 (the first volumetric training run) until
1.1.1's terrain quality is settled — training a volumetric head on top of an unresolved terracing
bug wastes the run and confounds the diagnosis of both.

### 0.2 What "we can't do overhangs" actually meant

`phase3.1-plan.md` §3 listed "caves and overhangs (architecturally impossible while `block_volume`
expands from one height per column)" under *explicitly not in scope*. That was accurate. The
mechanism is `export/export_onnx.py:141-146`:

```python
block = torch.full(shape, AIR_ID)
block = torch.where(y_b <= h_b, rock_b, block)          # everything below h is solid
block = torch.where((y_b >= h_b - 4) & (y_b <= h_b - 1), subsurface_b, block)
block = torch.where(y_b == h_b, surface_b, block)
block = torch.where((y_b > h_b) & (y_b <= water_b), WATER_ID, block)
```

`y <= h → solid` is a total order on each column. There is exactly one air/solid transition per
column, by construction, in the graph, forever. No amount of training changes that. The network
predicts ~350 scalars per chunk (`data/dataset.py` docstring) and the other 98,000 cells are a
consequence of them.

This plan's entire contribution is deleting that assumption.

### 0.3 One honest correction before anyone writes a changelog entry

**Caves already exist in Mc-Imagine worlds.** `ImagineChunkGenerator.applyCarvers` delegates to a
composed `NoiseBasedChunkGenerator`, and vanilla's carvers happily cut caves and ravines through
whatever solid mass the AI wrote — that is called out in the class javadoc as a deliberate win.

So the claim for v1.1.2 is **not** "caves for the first time." It is:

> The model now authors surface geometry that is not a function of a heightmap — overhangs,
> undercuts, arches and open cave mouths that it *chose*, at the places its prompt implies, rather
> than wherever vanilla's carver noise happened to intersect the surface.

Say that. This project has already shipped one overstated headline number (`phase3.1-plan.md` §1.2,
the "83–100%" that was a ceiling and not a result); the fix for that is not to be quieter, it is to
claim the true thing, which is still a big thing.

---

## 1. The architectural change

### 1.1 What has to become true

Three properties, and every design below is judged against them:

1. **Expressive enough** to represent an undercut cliff, a natural bridge, and a cave mouth you can
   walk into — arbitrary solid/air alternation near the surface.
2. **ONNX-expressible and cheap.** Still one forward pass per chunk, still `<100ms` on a mid-range
   box (`project-outline.md` §2.2 "Model Size & Performance Targets"), still no multi-step sampler.
3. **Seam-preserving.** `poc-plan.md` §1b's guarantee — every output cell is a pure function of
   (global block coordinate, seed) — survives, and stays *provable* by
   `export_onnx.verify_coordinate_purity`, not merely believed.

### 1.2 Chosen representation: a surface-anchored occupancy band

Keep the heightmap head. Add a second head that predicts **binary occupancy for a 64-block band
anchored to the predicted height**, and expand blocks from the band instead of from a comparison.

```
        y                     current                     v1.1.2
        │
  h+2   │  air                                     air
  h     │  ████ surface                            ████ surface  ← band top (h + 2)
  h-6   │  ████                                    ░░░░ AIR ── you can stand here
  h-12  │  ████   one transition,                  ████
  h-30  │  ████   guaranteed                       ░░░░ cave mouth
  h-62  │  ████                                    ████          ← band bottom (h - 61)
        │  ████                                    ████ solid, deterministic below the band
```

- **Inside the band** (64 cells per column, y from `h+2` down to `h-61`): the occupancy head decides.
- **Below the band:** solid rock, deterministically, exactly as today. Vanilla carvers handle deep
  caves down there and always have.
- **Above the band:** air, or water per §1.5.

**Why 64.** It has to cover the tallest natural void you want a player to walk into and the full
horizontal reach of an undercut, and it is the term that sets the head's output size. Measured
against the ground truth §2 produces, the deepest cavity that intersects the surface should sit
comfortably inside it; 64 is the starting value and **Gate 0 measures whether it is right** — if the
3D generator routinely produces voids deeper than ~48 blocks, either the band grows or the generator
is over-carving. Do not tune this after training starts; it changes the tensor shape.

**Why anchored to `h` and not absolute.** An absolute band would have to span the whole world height
to work at every elevation, which is the 98,304-cell problem again. Anchoring means the head only
ever models "the interesting 64 blocks", wherever they are, and the model does not spend capacity
learning that y=-30 is usually stone.

**Why this is the right shape of change for *this* codebase, specifically:** it is *additive on a
working system*. The heightmap head, `ReliefLoss`, the pinned `relief_frequency`, the coordinate
encoder, the whole Phase 2/3 body of work — all of it keeps its job, unchanged. If the occupancy
head learns nothing, it degenerates to "solid below h, air above" and you get v1.1.1 terrain back.
There is no version of this plan where a failed volumetric head produces *worse* terrain than the
last release, and given this project's history (`phase3.1-plan.md` §1.5: "if terrain ever goes flat
again"), that fallback is worth a lot.

**Rejected: full `[16,384,16]` occupancy.** 98,304 logits per chunk. A 1×1 conv from 192 channels to
384 y-channels is only ~74k params so it is not *impossible* — it is just mostly wasted. The class
imbalance is brutal (the overwhelming majority of cells are trivially air or trivially deepslate),
and the trivial cells dominate the gradient, which is precisely how you get a model that scores well
and produces nothing.

**Rejected: multi-layer heightmaps.** Predict N (top, bottom) pairs per column — 4 layers is 2048
values, exports trivially, and genuinely does overhangs. Rejected because supervising it is a
discrete assignment problem (which predicted layer corresponds to which ground-truth layer, in a
column where the count differs?), and because it cannot express a cave mouth that is open on one
side. The assignment ambiguity is the kind of thing that trains to a plausible-looking degenerate
solution, and this project does not need another one of those.

**Rejected: SDF / implicit surface + meshing.** More elegant, worse fit. Minecraft geometry is
natively voxel-quantized; predicting a continuous field and re-quantizing it adds a step whose
failure mode is exactly the terracing §0.1 is already fighting.

### 1.3 Architecture, and keeping the seam proof intact

Current stack: `CoordinateEncoder` → 28×28 grid (halo 6) → 6 valid 3×3 convs at 192ch → 16×16 →
heads. Every conv is `padding=0` on purpose; that is *the* mechanism behind the seam guarantee.

Proposed:

```
coord grid (16 + 2*halo)^2, halo = 9          →  34 × 34
  6 × valid 3×3 conv2d, 192ch                 →  22 × 22        (unchanged block)
  project 1×1 conv2d: 192 → 32 × 16           →  22 × 22 × (32ch × 16 y-levels)
  reshape to [B, 32, 16, 22, 22]                                (C, y, z, x)
  3 × conv3d, kernel 3, valid in x/z, pad 1 in y, 32ch  →  16 × 16, 16 y-levels
  transposed conv in y (stride 4): 16 → 64 y-levels
  1×1×1 conv3d → 1                            →  occupancy logits [B, 64, 16, 16]
```

Three things to note:

- **Halo goes 6 → 9.** Each 3D conv consumes one ring of x/z halo just as a 2D one does, so the
  seam guarantee is preserved *by the same argument as before* — no zero padding anywhere in x/z.
  `imagine_net._check_conv_arithmetic` already exists to catch exactly this class of off-by-one and
  its error message already says "halo == num_conv_layers"; **extend it to
  `halo == num_conv_layers + num_conv3d_layers`** and let it keep doing its job. Padding in *y* is
  fine and necessary: y is not a chunk axis, so it cannot produce a seam.
- **Cost of the wider halo:** the 2D backbone now runs on 34² instead of 28² — 1.47× the backbone
  conv work. Real, and the cheapest part of this plan's compute story.
- **Cost of the new head:** ~98k (projection) + ~83k (three 32-channel 3D convs) + the y-upsample ≈
  **~200k params on top of 2.57M trainable.** The extra inference work is small relative to the
  frozen 22M-param MiniLM encoder that already dominates the graph. Per-chunk latency is not the
  risk in this plan; §4 is.

`verify_coordinate_purity` needs no change — it tests `CoordinateEncoder` itself, which is upstream
of all of this. But its *sufficiency* argument in the docstring ("the valid-padding-only conv stack
being a fixed deterministic function") now has to cover the 3D stack too. Update that prose; it is
the written record of why the guarantee holds.

### 1.4 `heightmap` changes meaning — say so in the spec

Today `heightmap` is the thing everything is derived from. After this change it is **the topmost
solid cell of the emitted volume**, which is what the mod already wants it for: `getBaseHeight`,
`WORLD_SURFACE_WG` priming, vanilla decoration and structure placement.

Those two definitions must not drift, or decoration floats and structures bury themselves. Enforce
it in both places:

- **At export:** emit `heightmap` by *reading the final volume* (topmost solid, band-relative, with
  the head's `h` as the floor if a column somehow ends up empty), not by rounding the head's output.
  Then it is true by construction.
- **In training:** a consistency term (§3.3), so the heightmap head — which still anchors the band
  and still carries all of `ReliefLoss` — does not wander away from what the occupancy head does.

### 1.5 Water and surface dressing under an overhang

Two rules that today are one-liners and stop being one-liners the moment a column has more than one
solid run. Both must be ONNX-expressible in the exported graph, because the mod takes `block_volume`
literally for trained models (`ImagineChunkGenerator.resolveBlockState` javadoc is explicit about
why: applying mod-side water fill to a trained model's volume "floods every valley, cave mouth and
canyon floor the model deliberately left dry").

**Water.** Today: `y > h && y <= water_level → water`. Under overhangs, that floods a dry cavern
whose ceiling is 40 blocks below sea level, which is wrong and looks terrible. Rule:

> A cell is water iff it is air, `y <= water_level`, **and no solid cell exists above it within the
> band** — i.e. it is open to the sky.

That last clause is a cumulative maximum of the solid mask along y, top-down — `ReduceMax` on a
flipped axis, or a `CumSum` on the 0/1 mask compared against zero. Standard opset-17 ops, one extra
tensor. It is deliberately *not* a connected-component flood fill: a real flood fill is not
ONNX-expressible per-chunk and would break chunk independence anyway (water connectivity crosses
chunk boundaries). The consequence is that a cavern with a sideways opening to the ocean stays dry
where vanilla's aquifers would flood it. Accept it, document it, and note that vanilla's own carvers
and aquifers run afterward on this terrain regardless.

**Surface dressing.** Today: profile surface block at exactly `y == h`, subsurface for the 4 below.
Under overhangs there are multiple solid runs. Rule:

> Dress only the **topmost solid run** of each column — surface block at its top cell, subsurface
> for the 4 cells below that. Every other solid cell is rock (stone above y=0, deepslate below).

So the underside of an overhang is stone and the floor of a cave mouth is stone. That is what rock
undercuts actually look like, and vanilla's `buildSurface` still runs afterward and may dress
further. Same cumulative-max mask as the water rule, so it costs nothing extra.

`BlockPalette` has produced exactly this class of bug before — `phase3.1-plan.md` §2.2 notes the
Phase 2 off-by-one that rendered red sand as water. Extend `BlockPaletteTest`; do not read-through.

---

## 2. Ground truth: the 3D generator

**The model cannot learn overhangs from a dataset that has none, and today's dataset is a 2D
heightfield.** This is the largest single piece of work in the plan, and it is where it can most
easily go quietly wrong.

### 2.1 What gets built

`data/world_generator.py`'s `ProceduralWorldSource.render_region` keeps everything it does today —
archetypes, captions, the canonical noise field, `VALLEY_MASK_SCALE`, `SLOPE_NORM_SCALE`, the whole
Phase 2/3 body of measured constants — and gains a second stage that produces, per region, an
**occupancy band raster** `[CANVAS, CANVAS, 64]` alongside the heightfield.

The band is carved by a **3D noise field**, sampled at `(global x, y, global z + seed offset)`.

Four feature generators, each gated by a new per-archetype parameter, each aimed at a specific thing
a player can see:

| Feature | Mechanism | Where it shows up | Archetypes |
|---|---|---|---|
| **Cliff undercut** | 3D noise gated by a slope mask (reuse `SLOPE_NORM_SCALE`'s gradient), carving *into* the cliff face below the lip | The headline: overhanging cliffs you shelter under | `fjord_rift`, `deep_canyon`, `snow_peaks` |
| **Natural arch / bridge** | Horizontal tunnel carved through steep narrow ridges | Sea arches, canyon bridges | `fjord_rift`, `deep_canyon`, `mesa_plateaus` |
| **Cave mouth** | Cavity whose top surface intersects the terrain, opening outward | Walk-in cave entrances at grade | most land archetypes, weighted low |
| **Hoodoo / pillar** | Thin vertical rock columns left standing where the carve removes their surroundings | Badlands spires | `mesa_plateaus`, `deep_canyon` |

`desert_dunes`, `rolling_grassland`, `swamp` and `savanna_plains` get near-zero carve strength.
Flat terrain with overhangs in it is not a feature; the caption "endless flat desert dunes" should
produce a solid dune field and the model needs to learn that the *absence* of 3D structure is also
prompt-determined.

### 2.2 Two constraints that are not negotiable

**(a) The 3D field must be `WORLD_PERIOD`-periodic in x and z.** Identical argument to
`spec_constants.WORLD_PERIOD`, which is written out at length there and is the single most
load-bearing comment in the repo. `CoordinateEncoder`'s features are exactly 2048-periodic; a 3D
field that is not gives the network *identical inputs with different targets*, whose MSE/BCE optimum
is their average — a uniform grey mush that thresholds back to a plain heightfield. Every octave's
lattice must wrap in x/z exactly as `_perlin2d` already does. y needs no periodicity (it is bounded
and directly available).

**(b) The 3D feature frequencies must be PINNED per archetype**, degenerate range, excluded from the
25% archetype blend — exactly as `relief_frequency` is, for exactly the reason
`sample_params`'s docstring spends a page on. The caption cannot describe a carve frequency and the
seed cannot reveal it, so a randomized one is annihilated by conditional averaging. Measured for the
2D case: frequency randomization dropped achievable relief to 20–37% of target. There is no reason
to expect the 3D case is kinder, and every reason to expect it is worse (a carve is a *local*
feature; averaging over its phase erases it faster than averaging over a smooth field's).

`tests/test_model.py` already asserts every `relief_frequency` range is degenerate. **Extend that
same test to the new 3D frequency parameters.** That assertion is the closest thing this repo has to
an institutional memory of its own worst bug; make it cover the new surface too.

### 2.3 The dataset stops being free

`data/dataset.py`'s docstring currently makes a point of what it does *not* materialize:
`block_volume`, because it is derived. That changes — samples now carry a `[64,16,16]` occupancy
target, and every shard grows.

Budget it before generating 8000 regions. A `[520,520,64]` uint8 band is ~17 MB raw per region;
occupancy is a *bit* mask and compresses hard, and `np.savez_compressed` on a mostly-solid /
mostly-air volume should do well — but **measure it on 10 regions before committing to 8000**, and
if it is bad, pack to `np.packbits` along y (8× immediately). Running out of disk 6 hours into a
data generation run is an avoidable way to lose a day.

Also: **`data/dataset.py`'s shard contract assertions must gain a band check.** The existing ones
(biome id range, height band) exist precisely so a stale directory fails loudly instead of training
a confused model. A v1.1.x shard directory has no band array at all; that must be an explicit,
readable error, not a `KeyError` at step 400.

### 2.4 Look at it before you train on it

Add `viz.render_cross_section` — a vertical slice through a region, drawn as an image, so overhangs
and cavities are visible as shapes rather than inferred from statistics. `viz.py` already renders
shaded-relief heightfields for the qualitative dumps; this is the volumetric sibling and it serves
the same purpose (`TRAINING.md`: "**Look at those PNGs** — they are the real signal, more than the
loss number").

Add `scripts/diagnose_overhangs.py`, alongside the existing `diagnose_terracing.py` /
`diagnose_speckle.py`, computing the metrics §9 tracks:

- **overhang cells**: air cells with at least one solid cell above them in the band, as a % of band
  cells
- **multi-run columns**: % of columns with more than one solid↔air transition — the direct measure
  of "not a heightfield"
- **max cavity height** and its distribution (this is what validates the band size, §1.2)
- **walkable-void count**: cavities ≥3 blocks tall with ≥2 blocks of floor — the ones a player can
  actually enter, which is the only number that corresponds to the screenshot

Crucially, run it on **ground truth and on model output with the same code**, so "how much of the
data's 3D structure survived training" is one ratio and not two incomparable numbers. That framing
is what made `ReliefLoss`'s retention metric useful; reuse it.

---

## 3. Losses

### 3.1 Occupancy BCE, and the trap it walks into

Per-cell binary cross-entropy on the band is the obvious term and it is necessary. It is also, on
its own, **the fourth iteration of the failure this project keeps shipping.**

The argument is `ReliefLoss`'s docstring almost verbatim, one dimension up:

> The network sees only `(caption, chunk coords, world seed)`. Anything not determined by those
> inputs is noise, and the BCE-optimal answer for an unpredictable binary quantity is its **marginal
> probability**. A band that is 70% solid on average is best answered — pointwise — with 0.7
> everywhere. Threshold that at 0.5 and you get solid everywhere: a heightfield. Threshold a noisier
> version and you get uniform swiss cheese. Both score well. Both are worthless.

There is a real reason to think it can work anyway — the 3D field *is* a deterministic function of
global coordinates and the seed offset, so the phase is recoverable in principle, which is what
§2.2(a) and §2.2(b) exist to guarantee. But "recoverable in principle" was also true of the 2D
heightfield, and the measured retention was 21%.

So: BCE is necessary and not sufficient. Add §3.2.

Two details on the BCE itself:
- **Weight it by band position or by a hard-example mask.** Most band cells are trivially solid
  (deep) or trivially air (high); those dominate an unweighted mean. Weight toward cells near a
  solid/air transition in the *target*.
- **Log it separately** from everything else. `CombinedLoss.forward` now returns a components dict
  (Phase 3.1 §1.4, already landed) — this is exactly what that was for. Add `occupancy` and
  `overhang` keys.

### 3.2 `OverhangLoss` — match the *amount*, not the location

Direct descendant of `ReliefLoss`, same reasoning, and the plan lives or dies on it.

> Per-chunk **overhang-cell count** (or the count of multi-run columns) is a *magnitude*. Its
> optimum under averaging is not zero, and it is a statistic the caption genuinely determines — an
> archetype's carve strength is exactly the kind of parameter that survives conditional averaging,
> the same way amplitude and erosion did (measured 83–100%) while frequency did not (20–37%).

So: match predicted per-chunk overhang magnitude to the target's, MSE on the (normalized) counts,
computed differentiably from the occupancy *probabilities* rather than from a thresholded mask.

A model that puts correctly-scaled 3D structure in slightly the wrong place is penalized far less
here than a model that puts none anywhere. That is the trade that produced mountains in v1.1.0
(`Imgs/SecondModelVariableTerrain.png`) and it is the trade that will produce overhangs.

Inherits `ReliefLoss`'s known stationary point at a bit-exactly uniform prediction
(`phase3.1-plan.md` §1.5). Same analysis, same verdict: measure-zero, a local maximum, unreachable
by a real conv stack. Pin it in `test_model.py` the way `ReliefLoss`'s is pinned, and if the band
ever comes out uniform, **check this first before re-deriving it.**

### 3.3 Heightmap/volume consistency

The heightmap head and the occupancy head can disagree about where the surface is. §1.4 makes
export authoritative, but during training they need to be pulled together or the band drifts off its
anchor and half of it lands underground. A soft term: penalize the gap between the head's `h` and a
soft-argmax of the topmost solid cell in the predicted band.

Keep it modest. Its job is to stop drift, not to dominate — a heavy consistency weight is a
flatness term wearing a hat (`phase3.1-plan.md` §2.1's warning, which applies here word for word).

### 3.4 New weights, and how to set them

`occupancy_weight`, `overhang_weight`, `consistency_weight`. All three follow the established
pattern: **default `0.0`, declared explicitly in every shipped config, asserted by the config test**
(Phase 3.1 §1.1 added `test_configs_declare_relief_weight` for precisely this — extend it rather
than writing a second one). The `0.0` default is for backward compatibility with an un-pulled
checkout and must never be reached by a config we ship.

Sweep at rehearsal scale (400 regions / 400 steps) per §7 Gate 2. Pick the knee, not the max, and
watch that relief retention and biome accuracy do not regress — a new term that buys overhangs by
giving back the mountains is a bad trade, and §9 has a column for it.

---

## 4. Capacity — the risk that will actually bite

State this plainly because it is the most likely reason this release underdelivers:

**The current model has 2,566,742 trainable parameters and achieves ~21% relief retention on a
2D heightfield.** This plan asks that same backbone — plus ~200k params — to additionally learn a
3D occupancy field. There is a live possibility that the honest answer is "not enough capacity,"
and that the occupancy head produces mush no matter how good the loss design is.

`project-outline.md` §2.2's "Lite" tier is ~10M params with a `<50ms/chunk` GPU target. We are at a
quarter of that, and the frozen MiniLM encoder (22M, not trainable) means the *inference* budget is
already being spent — the trainable part is small enough that growing it is close to free at
runtime.

**Do:** treat a capacity increase as part of this phase, not a follow-up. Sweep
`conv_channels ∈ {192, 256, 320}` and `num_conv_layers` (remembering that layer count and halo are
locked together by `_check_conv_arithmetic`, so this is not a free knob — each added 2D layer needs
another ring of halo and grows the input grid). Measure both relief retention and the §2.4 overhang
metrics at each point, and record per-chunk inference latency, because that is the constraint that
eventually stops this.

**Gate this.** §7's Gate 2 has an explicit go/no-go: if the rehearsal model cannot produce
multi-run columns at a rate meaningfully above the marginal-probability baseline, do not start the
full run. Diagnose first. A 9-hour run that produces nothing while printing normal-looking progress
has happened here before (`config.cuda.yaml`'s mixed-precision comment) and the lesson was written
down; this is the place to apply it.

---

## 5. Mod-side work

Less than you would expect, which is a credit to the Phase 1/2 design. `ImagineChunkGenerator`
already reads `block_volume` cell-by-cell over the full y range, already primes
`WORLD_SURFACE_WG`/`OCEAN_FLOOR_WG` as it writes (the Bug #4 fix), and already trusts a trained
model's volume literally. **Overhangs arrive through the existing code path with no change.**

What does need doing:

1. **`FallbackTerrain` stays 2D and that is correct** — it is the no-model path. But
   `validateBlockVolume`'s fallback (`buildBlockVolumeFromHeightmap`) is also what a *malformed*
   trained volume falls back to, which silently converts a broken volumetric model into a
   heightfield model. Log that distinctly and loudly; a silent downgrade to "looks like the last
   release" is the hardest possible bug to notice.
2. **`BlockPaletteTest` extension** for §1.5's dressing rules — the topmost-run rule and the
   openness-gated water rule, on hand-built volumes with known overhangs. Test the rules, not the
   model.
3. **`ImagineChunkGeneratorTest`**: a column with two solid runs must round-trip through
   `fillFromNoise` and `getBaseColumn` with both runs intact, and `getBaseHeight` must return the
   *topmost* solid — the assertion that catches a §1.4 drift from the mod's side.
4. **Format-version handling** (§6): a 0.5.0 `.mcim` must keep loading and keep working. The mod
   reads the volume literally either way, so this is mostly a `ModelLoader` version-check and a log
   line, but write the test.
5. **Not in this release:** the F3 progress readout, per-tick budget, and profiling from
   `project-outline.md` Phase 3. Still deferred, still deliberately — `phase3.1-plan.md` §0's
   judgement that "it is not what players are complaining about" has not changed. §4's latency
   measurements will tell us if it starts to.

---

## 6. Spec and format changes — `model-spec.md` 0.5.0 → 0.6.0

`docs/model-spec.md` is the canonical contract and its own rule is "if a training or loading change
requires a shape/field change, update this file first and bump `format_version`." This change
qualifies, and it is a *semantic* break even where shapes are unchanged:

- **`heightmap`** is redefined as the topmost solid cell of `block_volume` (§1.4). Same shape, new
  guarantee, and downstream code depends on it.
- **`block_volume`** is no longer derivable from `(heightmap, profile, water_level)`. The
  "Output Tensors" section should say so explicitly, since that derivability is currently assumed
  in three separate docstrings.
- **`capabilities.caves`** becomes meaningful and true. The field already exists in the schema and
  `package_mcim.py` already has a `--caves` flag that nothing has ever passed; this is what they
  were for.
- **Backward compatibility:** 0.5.0 manifests remain loadable and 0.5.0 models remain correct — a
  heightfield volume is a valid volume. Follow the precedent 0.5.0 set for 0.4.0 in the Versioning
  section, and write the compatibility note there.

Release mechanics, carrying forward `phase3.1-plan.md` §2.4's unfinished items:

- Model version **1.1.2**. `--version` is already required on `build_mcim`/`package_mcim`.
- Tag `v1.1.2`. Confirm `v1.1.0` and `v1.1.1` tags exist first — as of this writing `v1.1.0` still
  does not, which means no artifact in the repo records that it happened.
- `mod/gradle.properties` `mod_version` is still `0.1.0-alpha`, untouched since Phase 0. §2.4 asked
  for a decision on whether the mod and model version together; this release forces it, because the
  mod gains volumetric block rules that a 1.1.x mod does not have. **Decide, and write the decision
  in `README.md`** — right now "v1.1.0" refers to the model only and nothing in the repo says so.
- `docs/CHANGELOG.md` entry, with §0.3's honest framing.
- New `Imgs/` capture, and update `docs/TRAINING.md`'s baseline table with the §9 numbers.

---

## 7. Sequencing

**Gate 0 — the data exists and you have looked at it.** No GPU. §2's 3D generator, §2.2's two
constraints and their tests, §2.4's cross-section viz and `diagnose_overhangs.py`, and the shard-size
measurement on 10 regions. **Exit criterion:** a rendered cross-section that visibly shows an
overhang you would want to stand under, plus a measured cavity-height distribution that confirms or
corrects the 64-block band. Runs in parallel with v1.1.1's GPU time.

**Gate 1 — the plumbing works, with no model involved.** Export a graph whose occupancy head is
replaced by a hand-written deterministic 3D carve, package it, load it in Minecraft, walk under the
overhang. This proves §1.5's water and dressing rules, the ONNX cumulative-max ops, the
`heightmap`-from-volume derivation, and every mod-side item in §5 — **before** a single GPU-hour is
spent. Also no GPU. If the screenshot at the end of Gate 1 is right, the rest of the plan is a
training problem rather than an engineering one.

**Gate 2 — learnability and capacity, at rehearsal scale.** 400 regions / 400 steps. Sweep §3.4's
three new weights and §4's capacity axis.

**Explicit go/no-go — a conjunction. All five conditions must hold, or the gate is a FAIL:**

1. **multi-run column rate ≥ marginal-probability baseline + 5.0 points**, and
2. **walkable-void *count* retention ≥ 25% of ground truth**, and
3. **walkable-void *mean floor area* retention ≥ 25% of ground truth**, and
4. **p90 cavity height ≥ 3 blocks**, and
5. **relief retention not regressed from v1.1.1's number.**

If any condition fails, stop and diagnose — do not start the full run.

**Why this is a conjunction, and not the single multi-run criterion this section originally
specified.** Condition 1 alone is *maximized* by exactly the failure §3.1 exists to warn about.
Measured on a **synthetic** band of pure 1-block speckle — 18% pocket density, random, no structure
of any kind — built while writing `scripts/diagnose_overhangs.py` and reproduced independently by a
second construction:

    multi-run columns 100.00%   vs. baseline 0.00%   =>  +100.00 points, a maximal PASS
    walkable voids 0            cavity p50 = p90 = max = 1

Multi-run rate counts *transitions*, and 1-block speckle maximizes transitions, so "uniform swiss
cheese" does not merely sneak past the criterion written to detect it — it posts the best possible
score on that criterion while containing nothing a player can enter.

`overhang_cells` does not rescue condition 1 either: a band with zero enterable structure can post a
large overhang percentage, because the metric counts sheltered *cells* and speckle shelters a great
many of them. Note what is deliberately **not** claimed here: that speckle out-scores real terrain on
`overhang_cells`. That ordering is not a stable property — it tracks pocket density, and on a
12-deep fixture with a 4-tall room the structured band scored double the speckle band (33.3% vs
16.7%). The claim that survives measurement is only that the metric cannot substitute for conditions
2 and 3. (These figures come from hand-built fixtures, not from the Phase 4 generator, which does not
exist yet — §2's generator is task A4. Replace them with real numbers once Gate 0 has run, and do not
let a fixture number inherit the authority of a measurement; that is how the 2D `_fbm` docstring
acquired its wrong "sub-percent" claim.)

A warning instead of a FAIL was considered and rejected: it is defeated by adding a single 4×4 room
to 0.39% of columns, which flips `voids == 0` false and suppresses the warning while the band is
still 99.6% speckle.

**Why condition 3 exists: counting voids does not measure having voids.** Walkable-void *count*
retention is a density ratio, so a model that shatters ground truth's structure into many tiny voids
scores *above* 100%. Measured, a partially-trained band posted **5564% count retention** — 59 voids
over 4096 columns against ground truth's 0.259 per 1000 — while its largest void was 5 floor cells
against ground truth's 4708.

Condition 4 caught that particular band, but only **incidentally**, because its cavities happened to
be 1 block tall. An adversarial band built specifically to defeat the three-condition version — 512
voids each exactly 3 tall with exactly 2 floor cells, against a ground truth of two large caverns —
**passes conditions 1, 2 and 4 together** (count retention 25600%, p90 = 3, margin met) while its
largest walkable void is 2 floor cells against ground truth's 1280, i.e. 0.2%. A p90 of 3 is a low
bar for a partially-trained occupancy head, so shattering at ≥3 tall is not a contrived scenario.

**A total-floor-area term does not close this** — that was the obvious fix and it was measured and
rejected: the adversarial band retains 49.2% of ground truth's total floor area, comfortably passing
a 25% threshold. It has plenty of floor; it is merely fragmented. What discriminates is **mean floor
area per void** (`floor_cells / void_count`), which scores the adversarial band at 0.2% while passing
a legitimately-weaker-but-honest model at 27.7% and rejecting speckle at 0%.

Conditions 2 and 3 are complementary and neither alone suffices: **count** catches a model producing
too few voids (speckle reads 0% there), **mean size** catches a model producing too many bad ones.
Mean size is preferred over a largest-void term for three reasons — both of its inputs are already in
the metrics dict so nothing new has to be defined or measured; both pool additively across regions,
whereas largest-void pools with `max()` and would let one lucky region carry the whole gate; and a
single order statistic is noisy on a 400-region rehearsal. Do **not** additionally cap condition 2 at
a ceiling: with condition 3 present the cap is redundant, and it would penalize a model that is
legitimately more cavernous than the ground truth.

Condition 2 is the one that corresponds to the screenshot — it is §9's "the number that matches the
screenshot" — and condition 3 rejects speckle directly, since a 1-block cavity is not a place a
player stands. Both are already computed by `diagnose_overhangs.py`; the original criterion's defect
was that the verdict function could not see them, not that the numbers were missing.

**A prerequisite on condition 2, recorded because it is easy to reintroduce:** walkable voids must be
measured in **world y**, not in band index. Band index is per-column anchored, so ±1 index is ±1
world block only where the anchor is flat. The quantity that actually drives the misalignment is the
*anchor* delta `|Δ rint(h)|`, not `|Δh|`. Measured over the 600 v1.1.1 shards in `model/data/`, the
fraction of adjacent column pairs with `|Δ rint(h)| > 1`: `snow_peaks` 28.5%,
`valley_mountains_craters` 22.0%, `fjord_rift` 21.1%, `deep_canyon` 8.4% (p99 9, max 46),
`mesa_plateaus` 3.4% — precisely §2.1's cliff-undercut and arch archetypes.

A band-index measurement severs one continuous shelter into many components and therefore
**overcounts voids**. On a 2:1 cliff carrying a single continuous 3-tall 8×24 shelter that is flat in
world y, band index reports **24 voids where the truth is 1**, and no tolerance setting fixes it:
tolerance 0 and 1 both give 24, tolerance 2 gives 1 — because in band-index space the tolerance that
works *is the local grade*, so there is no single correct value. In world y, tolerance 0 is already
right. An independent reproduction on a differently-shaped fixture — an L-shaped shelter sloping on
*both* axes — gives 15 voids against a truth of 1, and additionally collapses floor cells 72 → 30.

On a gently-rolling **synthetic** region the error is 173 → 70. Do not read that as "2.5×" in
general: it is tolerance-dependent, measured at mean |Δh| = 0.340 as 8.97× at tolerance 0, 1.03× at
tolerance 1 and 0.90× at tolerance 2 — i.e. **above the local grade band index over-merges and
*under*counts**, so "band index is an inflated upper bound" holds only at tolerance ≲ grade.

(This paragraph originally called that a "real" region. It was not, and could not have been: the 600
shards in `model/data/` predate this release and carry no band array at all, so a real shard yields
the degenerate fallback band with zero voids. The error is recorded rather than quietly deleted
because it is the second time in this document's short life that a fixture number was handed the
authority of a measurement — the thing the paragraph three screens up explicitly warns against.)

Condition 2 is a retention ratio, so an inflated count on both sides partially cancels — which is
exactly why the error is easy to miss and worth pinning here. **If exactly one side has a heightfield
available**, both sides must be re-measured in band-index space rather than mixed — in *either*
direction. A world-y ground truth paired against a band-index model output produces a retention
number that is mostly an artefact
of a missing file.

**Gate 3 — the full run.** 8000 regions on the chosen config. Per-term logging (already landed) is
what makes this readable; watch `occupancy` and `overhang` separately from `terrain` and `biome`.

**Gate 4 — export, play, release.** The four benchmark prompts from `docs/TRAINING.md` plus at least
one new overhang-specific prompt, the `Imgs/` capture, §6's spec bump and release mechanics.

**Explicitly not in 1.1.2:** `macro.onnx` and cross-chunk coherence (the natural v1.2.0 — coherent
mountain *ranges* rather than per-chunk mountains, and the other genuinely big visible leap
available); structures of any tier; detail passes; async/performance work; multiplayer. Volumetric
terrain is the entire release.

---

## 8. Risks

| Risk | Why it is plausible here | Mitigation |
|---|---|---|
| Occupancy collapses to the marginal probability → mush or a heightfield | This exact failure has shipped three times in 2D (`ReliefLoss` docstring, `sample_params` docstring, `phase3.1-plan.md` §1.2) | §3.2's `OverhangLoss`; §2.2's periodicity and pinned frequencies; Gate 2's go/no-go |
| Not enough capacity at 2.57M params | Relief retention is 21% on a *strictly easier* 2D task | §4 treats capacity as in-scope, with measured latency as the counterweight |
| Terracing from v1.1.1 propagates into the band | The band is anchored to `h`; a terraced `h` gives terraced overhangs | §0.1 — do not start Gate 2 until 1.1.1's terrain is settled |
| Enclosed cavities flood, or overhang undersides grow grass | Both rules are one-liners today and stop being one-liners | §1.5's cumulative-max rules, proven in Gate 1 before training |
| Shard size blows up the dataset | The band is ~64× more data per column than a height | §2.3 — measure on 10 regions, `packbits` if needed |
| Silent downgrade to heightfield terrain on malformed output | `validateBlockVolume` already falls back to a heightmap-derived volume | §5.1 — log the trained-model fallback distinctly and loudly |

---

## 9. Measurements log

Fill in as the gates produce numbers. Empty rows are the work. Every metric below comes from
`scripts/diagnose_overhangs.py` (§2.4) run identically on ground truth and on model output, so the
retention column is one honest ratio.

| Date | Gate | Measurement | Ground truth | Model | Retention | Notes |
|---|---|---|---|---|---|---|
| | 0 | Overhang cells (% of band) | | — | — | Data only |
| | 0 | Multi-run columns (%) | | — | — | Data only |
| | 0 | Max cavity height distribution | | — | — | Validates the 64-block band |
| | 0 | Shard size, 10 regions | | — | — | Extrapolate to 8000 |
| | 1 | Walk-under screenshot, hand-carved volume | — | — | — | Pass/fail; plumbing only |
| | 2 | Multi-run columns, rehearsal | | | | Go/no-go condition 1: ≥ baseline + 5.0 pts |
| | 2 | Walkable-void *count* retention, rehearsal | | | | Go/no-go condition 2: ≥ 25%. **World y, not band index** |
| | 2 | Walkable-void *mean floor area* retention | | | | Go/no-go condition 3: ≥ 25% — rejects shattering |
| | 2 | p90 cavity height, rehearsal | | | | Go/no-go condition 4: ≥ 3 blocks — rejects speckle |
| | 2 | Relief retention (must not regress from v1.1.1) | | | | v1.1.1 baseline: TBD (v1.1.0 was ~21%) |
| | 2 | Biome accuracy (must not regress) | | | | |
| | 2 | `occupancy_weight` / `overhang_weight` sweep | | | | Pick the knee |
| | 2 | Capacity sweep: `conv_channels ∈ {192, 256, 320}` | | | | Record per-chunk latency at each |
| | 3 | Full-run overhang metrics, 8000 regions | | | | |
| | 4 | Per-chunk inference latency, mid-range GPU | — | | — | Target `<100ms` (`project-outline.md` §2.2) |
| | 4 | Walkable voids per 100 chunks, benchmark prompts | | | | The number that matches the screenshot |
