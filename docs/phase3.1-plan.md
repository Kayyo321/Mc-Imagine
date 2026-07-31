# Mc-Imagine Phase 3.1 Plan — Terrain Quality & the v1.1.1 Release

**Goal:** close the issues the v1.1.0 merge left open, then convert the relief breakthrough into
*visible* in-game quality. v1.1.0 proved the model can make mountains. v1.1.1 makes those mountains
look like terrain instead of a staircase, and makes the surface material coherent instead of
speckled.

**Definition of done:** a clean CUDA box runs the five commands in `docs/phase2-plan.md`, produces
a `.mcim` stamped `1.1.1`, and the four benchmark prompts in `docs/TRAINING.md` render terrain that
is visibly better than the `Imgs/SecondModelVariableTerrain.png` baseline on both:
smooth slopes (no 4+ block cliff bands on gentle grades) and coherent surface material (no
dirt/snow speckle).

---

## 0. Where the project actually is (2026-07-30)

`docs/project-outline.md` §4 is the original pre-implementation roadmap and its phase numbering no
longer matches reality. The outline predicted mod work would run Phases 0–3 and model research
would be a parallel "Phase 4." What actually happened is that the mod raced ahead and the model
became the critical path. Current true state:

| Outline phase | Outline status | Actual status |
|---|---|---|
| Phase 0 — Foundation | planned | **Done.** Architectury multi-module builds Fabric + Forge, 1.20.1. |
| Phase 1 — Model loader & ONNX | planned | **Done.** `ModelLoader`, `ModelSession`, `.mcim` discovery, EP selection, fallback terrain. |
| Phase 2 — Prompt system & UX | planned | **Done.** `PromptParser`, `PromptTokenizer` (+ golden-vector test), world-creation tab, preset editor, seed field. |
| Phase 3 — Async generation & perf | planned | **Partial.** `ChunkGenerationQueue` and `ChunkCache` exist; no profiling, no per-tick budget, no F3 progress readout. Deferred deliberately — it is not what players are complaining about. |
| Phase 4 — The real AI model | "Weeks 15–30+" | **In progress, and it is the whole project right now.** Day 1 → Phase 2 → Phase 3 all landed here. |
| Phase 5 — Polish & release | planned | Not started. |

**Milestones achieved that the outline does not record — fold these in when it is next edited:**

- Day-1 PoC shipped (`Imgs/Day1ProofOfConcept.png`): prompt → playable world, end to end.
- Phase 2 (2026-07-28) fixed four bugs, took the biome palette 8 → 12, params 800 k → 2.57 M, and
  made the repo reproducible on a second machine (`docs/TO_TRAIN.md`).
- **Phase 3 / v1.1.0 (2026-07-30, PR #1)** fixed terrain flatness at its root: `ReliefLoss` matches
  per-chunk height *magnitude* instead of pointwise value, and `relief_frequency` is pinned per
  archetype so the caption fully determines feature scale. Result is visible in
  `Imgs/SecondModelVariableTerrain.png` — real mountains, for the first time.

**Corrections the outline needs:** §4's phase numbering should be relabelled to match the
`docs/*-plan.md` series that the team actually uses; §9's success criteria should note that
`block_volume` is expanded from one height per column, so "caves/overhangs authored by the AI" is
out of scope for 1.x and comes from vanilla carvers.

---

## 1. Known issues — do these first

These are ordered. §1.1 is a live trap that will waste a GPU day if anyone hits it.

### 1.1 `config.rehearsal.yaml` is a stale copy of the pre-Phase-3 CUDA config — **blocking**

Added in PR #1 and never mentioned in the commit message. Two independent defects:

- **No `relief_weight` key.** `train.py:363` reads `train_cfg.get("relief_weight", 0.0)` and
  `CombinedLoss` defaults it to `0.0`, so a rehearsal silently trains the *pre-Phase-3 objective* —
  the exact flat-terrain failure v1.1.0 exists to fix. The rehearsal would validate a pipeline that
  does not match the real run, which defeats its entire purpose (`docs/TO_TRAIN.md` §"Why the
  prompt is shaped this way").
- **`mixed_precision: true`** (line 64). `config.cuda.yaml` just set this to `false` with a comment
  describing the 2026-07-28 run where GradScaler collapsed 2048 → 0.0, every optimizer step from
  epoch 12 was skipped, 26/127 tensors went NaN, and nine hours produced nothing while the log
  printed normal progress.

**Do:**
1. Add `relief_weight: 10.0` to the `training:` block, kept identical to the other two configs.
2. Set `mixed_precision: false`.
3. Rewrite the header — it currently reads "Training configuration — NVIDIA / CUDA. This is the
   config the real runs use" and its resume example points at `config.cuda.yaml`/`cuda_run1`. It
   should say what it is: a fast smoke test, `num_epochs: 1`, paired with `--regions 400
   --max-steps 50`.
4. Add a test in `model/tests/test_model.py` that loads every `config.*.yaml` and asserts each one
   declares `relief_weight` explicitly. The `0.0` default exists for backward compatibility with a
   collaborator's un-pulled checkout; it must never be reached by a config we ship.

### 1.2 The 83–100% figure is a ceiling, not a result — fix the record

The commit message advertises "83-100% retention" and "dramatically improved." That number is the
*theoretical* relief surviving conditional averaging. What training actually achieved is in
`config.cuda.yaml:41-43`: **~21% of target, holding steady, versus a peak of 19% at step 750
followed by collapse to 6%.** Preventing the collapse is a real and important win. It is not 83%.

**Do:** correct the claim in `docs/TRAINING.md`'s baseline table, and record 21% as the v1.1.0
number that v1.1.1 has to beat. Nobody should tune against the ceiling by mistake.

### 1.3 `relief_weight: 10.0` was never swept

The value appears fully formed in both shipped configs with no sweep recorded. It is an additive
term on a different scale from the others, and §1.2 says it is achieving 21% of target — which is
exactly the signature of a weight that is too low.

**Do:** sweep `relief_weight ∈ {5, 10, 25, 50, 100}` at `--regions 400 --max-steps 400`. Record
final relief-retention % and terrain/biome loss for each in §4. Pick the knee of the curve, not the
max — relief retention traded against biome accuracy is a bad deal.

### 1.4 No per-term loss logging

`CombinedLoss.forward` (`losses.py:275-287`) returns a bare scalar and `train.py` only accumulates
the total. You cannot tell whether the relief term is falling, or what it is trading against. This
is untenable for a term whose stated justification is that the headline metric gets *worse*.

**Do:** have `CombinedLoss.forward` return `(total, components_dict)` with keys
`terrain / biome / relief`, log each at `log_every`, and keep the scalar-only path working for the
existing call sites (or update them — there are few). §1.3's sweep is not readable without this.

### 1.5 Known stationary point at exactly-flat predictions — accept, don't fix

`ReliefLoss` is a function of `|deviation|`, so autograd yields exactly zero gradient at a
bit-exactly constant patch. This is inherent to every magnitude-matching loss. The docstring's
argument that it is benign is sound (measure-zero, and it is a local *maximum*; measured
within-chunk std is 0.006 at init versus float32 ULP ~1.15e-5 at terrain magnitudes), and
`test_model.py` pins the behaviour.

**Do:** nothing — but if terrain ever goes flat again, check this first before re-deriving it.

### 1.6 Documentation and version drift

- `world_generator.py:171` says "all 13 archetypes"; there are **11**. The commit message repeats 13.
- `sample_params`'s docstring still says "memorize 9 discrete clusters" — stale from an earlier phase.
- `config.cuda.yaml:15` resume example says `cuda_run1`; `checkpoint_dir` is now `cuda_run3`. Same
  stale path in `docs/TRAINING.md:72` and `:97`.
- **No `v1.1.0` git tag exists**, and `build_mcim.py:57` / `package_mcim.py:189` still default
  `--version` to `1.0.0` — so no artifact in the repo actually records that 1.1.0 happened. Tag the
  merge commit retroactively, and make `--version` a **required** argument so an unstamped `.mcim`
  cannot be built.

---

## 2. Next steps — earning the "visible progress" in v1.1.1

`Imgs/SecondModelVariableTerrain.png` is the v1.1.0 baseline and it is the honest brief for this
release: the mountains are *there*, and they are *wrong* in three specific, fixable ways.

> **Diagnose before fixing.** Each item below starts with a measurement. This project's two worst
> bugs both looked plausible and raised no errors, and the Phase 2 water-pool theory was confidently
> wrong until someone audited the id-mapping code (`docs/phase2-plan.md` §0). Do not ship a fix for
> a cause you have not measured.

### 2.1 Terracing — the staircase problem *(highest visual impact)*

**Symptom:** slopes render as wide flat shelves separated by sharp 3–5 block cliffs, over the whole
frame, including on gentle grades where vanilla would give a smooth ramp.

**Diagnose:**
1. Dump a raw predicted heightmap for a mountain prompt and histogram the per-column height
   *deltas*. Terracing appears as spikes at integer multiples with near-empty gaps between them.
2. Determine which stage introduces it. Three candidates, distinguishable by where the histogram
   first goes multi-modal:
   - **Ground truth.** `plateau_quantization` is a real archetype parameter; if the *targets* are
     already terraced, the model is faithfully reproducing the training data and the fix is in
     `world_generator.py`, not the model.
   - **The model.** `ReliefLoss` matches per-chunk *std*, and std is indifferent to how the
     deviation is distributed — a chunk that is half-high and half-low scores identically to a
     smooth ramp of the same magnitude. **This is a plausible side effect of the v1.1.0 fix and is
     the first thing to check on the model side.**
   - **The mod.** Rounding float height → int block y in `ImagineChunkGenerator`, or per-chunk
     quantization at the 16×16 patch boundary.
3. Cross-check the seam: if shelves align to 16-block boundaries it is the mod or the patch
   structure; if they cut across chunks it is the data or the loss.

**Fix, once located.** If it is the loss, the candidate is a smoothness/total-variation term that
penalizes *second* differences — high curvature — while leaving first differences (the relief
`ReliefLoss` is protecting) alone. Add it as `smoothness_weight`, defaulting to `0.0`, same
backward-compatible pattern `relief_weight` used. Sweep it jointly with §1.3, and watch that relief
retention does not regress: a smoothness term is a flatness term wearing a hat, and this project has
shipped flat terrain three times.

### 2.2 Surface material speckle

**Symptom:** snow, dirt and stone interleave as per-block noise across a single continuous slope
instead of forming coherent layers. Dirt patches appear at peak altitude where only snow belongs.

**Diagnose:**
1. Is the *biome grid* noisy, or is the surface-block rule? Dump the predicted `biome_grid` for the
   frame and check spatial coherence at its 4×4 resolution.
2. If biomes are coherent, the bug is in the mod: audit surface/filler depth selection in
   `BlockPalette` and `ImagineChunkGenerator`. Note `BlockPalette` has already produced exactly this
   class of bug once — the Phase 2 off-by-one that rendered red sand as water. `BlockPaletteTest`
   exists; extend it rather than trusting a read-through.
3. If biomes are genuinely noisy, it is a `BiomeLoss` problem: per-cell cross-entropy has no spatial
   coherence pressure, so it is free to speckle.

**Fix:** whichever the measurement indicates. If it is biome noise, the cheapest correct fix is
altitude-conditioned surface selection in the mod (snow above a threshold) rather than a new loss
term — it is deterministic, testable, and does not risk the relief work.

### 2.3 Ocean floor has perlin-noise-like dips *(new, low priority)*

**Correction:** `Imgs/EndlessSeaWithUnconnectedIslands.png` is **not** a bug capture. It's a manual
test where the prompt used was the literal string "endless sea with unconnected islands," and the
model reproduced that brief well — an archipelago in open ocean is the correct output for that
prompt, not evidence of an ocean-dominance defect. There is no ocean-dominant-worlds bug to fix here;
this section is retitled from the original "Ocean-dominant worlds" diagnosis.

**What the same image does show:** the ocean floor has visible dips/undulation that read as
perlin-noise artifacts rather than a flat or gently sloping seabed — visible as the mottled darker
patches under the water surface in the screenshot.

**Decision: do not open a separate workstream for this.** Reasoning:
- It's underwater and has near-zero visual impact relative to §2.1's terracing and §2.2's speckle,
  which affect the visible land surface players actually stand on.
- The mechanism is almost certainly the same one already under investigation in §2.1: `ReliefLoss`
  matching per-chunk magnitude rather than shape has no preference between a smooth gradient and an
  uneven one, and there's no reason the seafloor would be exempt from whatever §2.1 finds on land.
- If §2.1's fix (e.g. a curvature/smoothness term) lands, re-check this image against the new
  checkpoint before doing any dedicated seafloor work — it may resolve for free.

**Do:** note it in §4 when §2.1 is diagnosed and fixed; only spin up separate work if the seafloor
still looks wrong after that lands.

### 2.4 Release mechanics for 1.1.1

- Bump `mod_version` in `mod/gradle.properties` (currently `0.1.0-alpha` and untouched since
  Phase 0) — decide whether the mod and model version together or separately, and **write the
  decision down**; right now "v1.1.0" refers to the model only and nothing in the repo says so.
- Make `--version` required on `build_mcim` (§1.6) and stamp `1.1.1`.
- Tag `v1.1.0` retroactively on `e4de386`, then tag `v1.1.1` on release.
- Add a `docs/CHANGELOG.md`. Three releases have now shipped with the narrative living only in
  commit messages, which is why §1.2's overstated claim went unnoticed.
- Update `docs/TRAINING.md`'s baseline table with v1.1.0 numbers so v1.1.1 has something to beat.

---

## 3. Sequencing

**Gate 1 — trustworthy measurement.** §1.1, §1.4, §1.6. Nothing below is readable without per-term
logging and a rehearsal config that trains the real objective. Cheap; do it in one sitting.

**Gate 2 — diagnosis.** §2.1, §2.2, §2.3 diagnosis steps only, all three on the *existing* v1.1.0
checkpoint. No retraining. Record every measurement in §4 as you go.

**Gate 3 — fix and sweep.** §1.3's `relief_weight` sweep jointly with whatever §2.1 turns up, at
rehearsal scale (400 regions / 400 steps). Pick weights from measured curves.

**Gate 4 — real run and release.** Full 8000-region run on the chosen config, export, load in
Minecraft, check the four benchmark prompts, then §2.4.

**Explicitly not in 1.1.1:** async/performance work (outline Phase 3), structure placement, caves and
overhangs (architecturally impossible while `block_volume` expands from one height per column),
multiplayer. Terrain *quality* is the entire release.

---

## 4. Measurements log

Fill this in as Gate 2 and Gate 3 produce numbers. Empty rows are the work.

| Date | Measurement | Result | Conclusion |
|---|---|---|---|
| 2026-07-30 | v1.1.0 relief retention (from `config.cuda.yaml`) | ~21% of target, stable | Baseline for 1.1.1 |
| | Height-delta histogram, mountain prompt | | §2.1 terracing source |
| | Biome-grid spatial coherence | | §2.2 speckle source |
| | Re-check seafloor dips (`Imgs/EndlessSeaWithUnconnectedIslands.png`) against post-§2.1 checkpoint | | §2.3 — resolved for free, or needs its own work |
| | `relief_weight` sweep {5,10,25,50,100} | | §1.3 chosen weight |
