# Mc-Imagine Phase 2 Plan — Root-Cause Fixes + Cross-Machine Training Handoff

**Goal:** fix the two quality bugs the Day-1 PoC exposed at their actual root, make real
architectural progress, and leave the repo in a state where it can be cloned onto a different
machine (NVIDIA GPU), trained from scratch, exported, and tested in-game — with no undocumented
manual steps.

**Definition of done (the only thing that matters):** on a clean clone of this repo, on a CUDA box,
these five commands work and produce a better world than Day 1:

```bash
pip install -e model/                                  # REQUIRED — the package must be importable
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match your CUDA version
python -m mc_imagine_model.scripts.bootstrap           # vendors the MiniLM checkpoint
python -m mc_imagine_model.scripts.generate_data --regions 8000 --out model/data
python -m mc_imagine_model.training.train --config model/src/mc_imagine_model/training/config.cuda.yaml
python -m mc_imagine_model.scripts.build_mcim --checkpoint <ckpt> --out <name>.mcim
```

`pip install -e model/` is not optional and is easy to miss: without it every `python -m
mc_imagine_model.…` invocation fails on a clean clone. Console-script aliases
(`mc-imagine-bootstrap`, `mc-imagine-generate-data`, `mc-imagine-train`, `mc-imagine-build-mcim`)
are registered as part of that install.

---

## Status: Phases 1, 2, 4, 5 landed (2026-07-28). Phase 3 deferred.

Measured outcomes, all from the regenerated dataset and the current code:

| | Day 1 | Now |
|---|---|---|
| Deepest terrain generated | y = 27.7 | **y = −7.9** (57% of regions dip below sea level) |
| Region relief std (mean / max) | 5.65 / 23.09 | **7.58 / 22.77** |
| Columns pinned at the height ceiling | 14% of regions exceeded it | **0** (0/162,240,000 columns) |
| Cross-region continuity | independent field per region | **halo overlap bit-identical**, seam step 0.10 vs interior 0.09 |
| Canonical-field coverage by training data | ~3% of a 65536² domain | **37.5× oversampling** of one 2048² domain (500× at 8000 regions) |
| Slope-loss contribution to the gradient | n/a (term didn't exist) | 1.053 — was **2.7e-5** under the first, wrong normalization |
| terrain : biome loss balance | — | **2.24 : 1** (was 0.021 : 1) |
| Trainable params | ~800 k | **2.57 M** |
| Biome palette | 8 | **12** (badlands family present at 14% of columns) |
| Java tests | 21 | **34** (30 pass, 4 auto-skip on the stale Day-1 model) |

**Four bugs were found that the plan did not predict**, each of which would have silently degraded
or blocked the retrain. They are documented inline below rather than quietly fixed:
1. `BlockPalette` off-by-one — the *actual* cause of the Day-1 screenshot (§0).
2. The valley carve was a measured no-op (~3 blocks, not ~55) because the ridged noise is skewed.
3. The coordinate encoder's 2048-block period made the canonical field unlearnable anyway (§1a-bis).
4. `pyproject.toml` was invalid, so `pip install -e model/` failed on any clean clone.

## 0. Verified diagnosis — what is actually wrong (measured, not assumed)

Every claim below was checked against the shipped model and the real training shards.

| Symptom | Measured evidence | Actual root cause |
|---|---|---|
| Peaks are flat-topped, no relief | `"towering snow-capped peaks"` → height min **158.84**, max **158.98**, std **0.03**. `TerrainHead` is `63 + 96·tanh(x)` ⇒ hard ceiling **159**. Ground-truth data reaches **235.6**; **14%** of regions exceed 159. | **tanh saturation.** The head physically cannot represent the target, so it pins at the ceiling where the gradient ≈ 0 and all local variation dies. |
| Terrain is smooth everywhere; no deep valleys | Model output std **0.02–3.73** per prompt vs. ground-truth per-chunk std up to **8.75** and per-region std up to **23.09**. | **The training target is unlearnable, so MSE collapses to the conditional mean.** `world_generator.render_region` seeds Perlin with `params.seed*7919 + region_x*104729 + region_z*15485863` — *every macro-region is an independent pseudorandom field*. The network only sees (prompt, seed, global coords) and cannot invert a hash, so the MSE-optimal output is the *average* terrain: flat, at the right mean height. This is the dominant cause, not model capacity. |
| Valleys never go "underground" | Ground-truth global min over 200 regions = **27.7**. The `np.clip(…, 4.0, 250.0)` floor is **never reached** (0% of regions go below 4). | **The data has no deep valleys to learn from.** Sea level is 63; terrain dips below it in 40% of regions but only modestly. Nothing approaches bedrock. |
| Red-mesa prompt spawns random water pools **and** a "bedrock floor" | Model's own `block_volume` for the mesa prompt contains **zero water** and places bedrock only at `y == minY`. But `BlockPalette` prepended `minecraft:air` to a palette that **already declared air at index 0**, shifting every id by one. | **An off-by-one in `BlockPalette` (Java), not a model or biome problem.** Reproduced exactly: model `stone`(2) rendered as **bedrock**, model `red_sand`(8) — the mesa's entire surface layer — rendered as **water**, model `bedrock`(1) rendered as air. That is precisely "a bedrock floor with random pools of water". Fixed by detecting the explicit-air convention instead of assuming the implicit one. |

> **Correction (recorded deliberately).** An earlier draft of this table blamed the water pools on
> the missing badlands biome — the theory being that mesa→savanna mislabeling let *vanilla* savanna
> lake decoration place them. That was **wrong**, and it was wrong in the most dangerous way: it was
> plausible, it fit the visible symptom, and it was supported by real evidence (the biome grid *is*
> 100% savanna). The palette off-by-one is the actual cause, found only because the Java audit
> checked the id-mapping code rather than trusting the narrative. The biome-palette work below is
> still worth doing — correct biome labels drive grass colour, foliage, mob spawns and vanilla
> decoration — but it is a **quality fix, not the bug fix**.

**A structural limitation worth stating plainly:** `block_volume` is expanded in-graph from a *single
height per column* (`torch.where` over a y-grid). Overhangs, arches and caves are therefore
**impossible for the model to express** by construction — today they can only come from vanilla
carvers. "Valleys you can walk down into" is achievable (it's just a large negative height
excursion); "tunnels and overhangs authored by the AI" requires a true 3D volume head and is
deliberately **out of scope** here.

**Handoff blockers found (repo is currently not reproducible elsewhere):**
- `model/data/` (1.6 GB) and `model/checkpoints/` are gitignored — correct, but there is **no script
  to regenerate either**. `generate_shards` is a method with no CLI; MiniLM was fetched ad-hoc.
- `requirements.txt` uses `>=` everywhere — no reproducible pin, no CUDA guidance.
- `config.yaml` hardcodes `device: "mps"`, `num_workers: 0`, no AMP, no resume-from-checkpoint.

---

## 1. Two decisions that drive everything downstream

### 1a. Replace per-region random noise with one canonical field + seed-as-offset

This is the load-bearing fix. Today each region gets its own pseudorandom Perlin field, so the
target is a different unlearnable function in every region. Instead:

- Use **one fixed gradient table** for the whole project (no per-region reseeding).
- Derive a per-world **offset** from the world seed alone: `(off_x, off_z) = g(world_seed)`.
- Sample all noise at `(global_x + off_x, global_z + off_z)`.

Consequences, all of them wanted:
- The target becomes a **deterministic, continuous, learnable function of global coordinates** — the
  exact thing `CoordinateEncoder`'s Fourier features are built to represent. Relief becomes
  learnable instead of averaging away.
- Landforms become **continuous across region boundaries**, so a valley can genuinely span hundreds
  of blocks. (This delivers most of what a macro field was going to buy us for this tier.)
- `same seed ⇒ same world` and `different seed ⇒ different layout, same character` both still hold —
  a different seed samples a different translation of the field, which is essentially how vanilla
  Minecraft seeds already behave.
- The §1b seamlessness guarantee is untouched: output stays a pure function of global coordinate.

The seed offset must be fed into the model (added to coordinates *before* Fourier encoding) so the
network can actually apply it, rather than having to infer it.

### 1b. Fix the head range and the data range together, in one pass

Widen `TerrainHead` to cover the full useful band (target ≈ `[-64, 256]` rather than `[-33, 159]`),
**and** re-range the generator so ground truth lives inside what the head can represent. Add a
startup assertion that fails loudly if any training target falls outside the head's range — this
class of silent saturation should never survive a run again.

---

## 2. Phases

Each phase has a gate. Phases 1–2 change the data contract, so they land as one
regenerate-and-retrain cycle; the Day-1 `.mcim` is invalidated by design.

### Phase 1 — Root-cause bug fixes

1. **Head range** (`model/heads.py`): widen the height/water squash; assert targets are representable.
2. **Canonical noise + seed offset** (`data/world_generator.py`, `model/positional.py`): per §1a.
3. **Deep valleys in the data** (`data/world_generator.py`): add canyon/deep-valley archetypes,
   widen `relief_amplitude` and `base_height`, strengthen the existing `valley_mask` carving, drop
   the clip floor so terrain can reach well below sea level.
4. **Biome palette** (`spec_constants.py`, `data/labeler.py`, `model/heads.py`): add `badlands`
   (+ `eroded_badlands`, `stony_peaks`, `beach` as cheap wins), map mesa profile → badlands, bump
   `NUM_BIOMES`. Manifest-only change — `biome_grid` keeps its `[4,96,4]` shape, so **no
   `format_version` bump is needed**.
5. **Java water override** (`ImagineChunkGenerator.resolveBlockState`): the
   `air && y <= seaLevel ⇒ water` rule currently overrides a spec-conformant model's authoritative
   volume. Gate it so it only applies to `FallbackTerrain`-derived volumes.

**Gate:** a 50-step smoke train + render of the 6 held-out prompts shows height std **> 8** for
valley/peak prompts (vs. 0.03–3.73 today), no prompt pinned at a head bound, and mesa predicting a
badlands-family biome.

### Phase 2 — Capacity and fidelity (this is what the GPU is for)

- Deeper/wider decoder (more conv layers and channels than the current 4×128) — currently only
  ~800k trainable params, which is tiny.
- More and more-diverse data: ~8000 regions, more chunks sampled per region.
- Add a **slope/gradient term** to `TerrainLoss` so relief is supervised directly rather than only
  absolute height (guards against a smooth-but-correct-mean solution returning).
- Optionally unfreeze the top MiniLM layers once the rest is stable.

**Gate:** held-out prompts are visually distinct *and* show real local relief; val loss tracked
against a Day-1 baseline on identical prompts.

### Phase 3 — Next architectural milestone: the macro field (M10) — **DEFERRED to a later session**

> Scope decision (2026-07-28): deferred so Phases 1/2/4/5 land cleanly and the repo ends fully
> trainable on another machine. Note that §1a's canonical-noise fix already delivers most of what
> the macro field was going to buy *for this tier* — landforms continuous across region boundaries —
> so deferring it costs less than it looks. It remains required for prompt-driven regional variation
> (flavor zones) and structure-candidate placement at higher tiers. Design below is unchanged and
> ready to pick up.

Implement `macro.onnx` end to end: `MacroFieldNet` + `MacroFieldLoss` + `McImagineMacroDataset`
(Python), `MacroCache` and `macro_local_height` / `flavor_zone_*` slicing (Java), per the contract
already written in `docs/model-spec.md`.

**Critical constraint:** everything stays behind `capabilities.requires_macro_field`. A model that
declares `false` — including everything Phases 1–2 produce — must keep working untouched. This is
what keeps the "end state is always trainable" requirement true even if macro training is never run
on the other machine.

**Gate:** the low-tier `.mcim` still loads and generates identically with the macro code present but
unused; a macro-enabled model round-trips through export and ORT.

### Phase 4 — GPU and cross-machine portability (the actual handoff requirement)

- `scripts/bootstrap.py` — downloads and vendors the MiniLM checkpoint via `huggingface_hub`.
- `scripts/generate_data.py` — real CLI over `ProceduralWorldSource.generate_shards`
  (`--regions`, `--out`, `--seed`, `--workers`).
- `config.cuda.yaml` alongside `config.mps.yaml`: `device: cuda`, AMP on, `num_workers: 8`,
  `pin_memory`, larger batch.
- Resume-from-checkpoint, and device-independent seeding so runs are comparable across machines.
- Pin `requirements.txt`; document the CUDA-matched torch install line separately (it must not be
  hard-pinned in the shared file, since the CUDA wheel index differs per machine).

**Gate:** `--max-steps 50` smoke train passes on **both** this Mac (MPS) and CPU, proving no
device-specific code paths, before the friend ever sees it.

### Phase 5 — Verification and handoff

- `docs/TRAINING.md`: exact commands, expected wall-clock on a modern NVIDIA card, expected loss
  curve, and what "good" looks like at each gate.
- Extend `render_gate_prompts.py` into a proper before/after comparison against the Day-1 numbers
  recorded in §0, so improvement is *measured*, not eyeballed.
- Keep the Java test suite green (21 tests today), including the tokenizer golden-vector test — the
  biome-palette change must not silently break the tokenizer/manifest contract.
- Acceptance checklist the friend can run top-to-bottom.

---

## Explicitly out of scope

True 3D volume prediction (overhangs/AI caves), structure tiers (M9/M14–M16), detail passes (M17),
`.mca` extraction from real worlds (M12 proper), int8 quantization of the MiniLM encoder, and
batched/speculative chunk inference. None of these block the handoff.

## Risk register

| Risk | Mitigation |
|---|---|
| Canonical noise makes all worlds feel same-shaped | Seed offset + per-region params still vary character; if too uniform, add seed-driven rotation and low-dim frequency modulation. |
| Bigger model blows the `<100 ms/chunk` budget | Current latency is **8.9 ms** — roughly 10× headroom. Re-measure at the Phase 2 gate. |
| Phase 3 half-finished leaves repo untrainable | Macro field is gated behind `requires_macro_field: false`; the chunk-only path is never allowed to depend on it. |
| Friend's box can't reproduce data | Phase 4 bootstrap + data CLI are the gate; data generation is fully deterministic from `--seed`. |
