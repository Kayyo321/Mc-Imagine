# Changelog

All notable changes to the Mc-Imagine model, training pipeline, export scripts, and documentation are recorded here.

## [Unreleased] - Phase 4 / v1.1.2 (2026-07-31)

### Added & Changed
- **Gate 0 — the 3D ground-truth generator** (`docs/phase4-plan.md` §2, no GPU required):
  - **Occupancy band raster**: `ProceduralWorldSource.render_region` (`data/world_generator.py`) gains
    a second rendering stage, `_render_occupancy_band`, that carves a `[CANVAS, CANVAS, 64]` band per
    region and returns it as `"band"` in `render_region`'s output dict, alongside the existing
    heightfield/profile/biome maps. `heightfield` returned from `render_region` is now the
    post-hoodoo, band-consistent surface (§1.4) — everything upstream of the band (profile map, carve
    gates) still reads the pre-hoodoo heightfield, per the contract documented at the call site.
  - **Four carve features**, each gated by a new per-archetype parameter and pinned to a degenerate
    frequency range excluded from the 25% archetype blend (§2.2b, same reasoning as
    `relief_frequency`): cliff undercut (slope-gated, `fjord_rift`/`deep_canyon`/`snow_peaks`),
    natural arch/bridge (`fjord_rift`/`deep_canyon`/`mesa_plateaus`), cave mouth (most land
    archetypes, weighted low), and hoodoo/pillar (`mesa_plateaus`/`deep_canyon`).
    `desert_dunes`/`rolling_grassland`/`swamp`/`savanna_plains` carry near-zero carve strength by
    design.
  - **`WORLD_PERIOD`-periodic 3D noise field**: the carve kernels sample `(global x, y, global z +
    seed offset)`, exactly `WORLD_PERIOD`-periodic in x/z (never in y, which is bounded and handed to
    the network directly) — the same periodicity argument `spec_constants.WORLD_PERIOD` makes for the
    2D field, extended to a third dimension (§2.2a).
  - **`tests/test_model.py`**: `test_relief_frequency_is_pinned_per_archetype` extended to cover the
    new 3D carve/hoodoo frequency parameters; `test_fbm3d_is_bit_exactly_world_period_periodic`,
    `test_band_top_solid_cell_is_the_returned_heightfield`,
    `test_band_is_bit_exactly_world_period_periodic_and_deterministic`, and
    `test_flat_archetypes_have_no_3d_structure` added to pin the band's determinism, periodicity, and
    the post-hoodoo heightfield contract.
  - **`data/dataset.py`**: samples now carry the `[64,16,16]` occupancy target, and the shard-contract
    assertions (biome id range, height band) gain an explicit band-presence check — a v1.1.x shard
    directory with no band array now fails loudly and readably instead of a `KeyError` deep into
    training (§2.3).
  - **`viz.render_cross_section`**: renders a vertical slice through a region's band so overhangs and
    cavities are visible as shapes, the volumetric sibling of the existing shaded-relief heightfield
    dumps (§2.4).
  - **`scripts/diagnose_overhangs.py`**: computes overhang cells (% of band), multi-run columns (%),
    max cavity-height distribution (validates the 64-block band), and walkable-void count/mean-floor-
    area — measured in **world y**, not band index, per §7's prerequisite (band-index measurement on
    a sloped column overcounts voids by up to 24x, measured on a hand-built fixture). Runs identically
    on ground truth and on model output so retention is one honest ratio (§2.4).
  - **Shard-size measurement (§2.3)**: measured on the 10 regions in `model/data_test_audit/`
    (`model/overhang_report.json` is the accompanying diagnostic run) — raw `bool[520,520,64]` band is
    ~16.5MB/region; `np.savez_compressed` brings that down to a measured 839KB-1031KB/region (mean
    ~915KB), a ~16-20x compression ratio, projecting to ~7.0-7.9GB for 8000 regions. **Decision:
    `np.savez_compressed` is sufficient; `np.packbits` is not needed** — comfortably within disk
    budget, so §2.3's fallback plan is not exercised for this release.
- **3D Occupancy Head (`Occupancy3DHead`)**: Added 3D volumetric head to `ImagineNet` (`model/heads.py`), converting 22x22 2D feature maps via 1x1 Conv2d, reshaping to 3D, applying 3x Conv3d layers (valid in x/z, padded in y), 4x ConvTranspose3d upsampling along y, and 1x1x1 Conv3d to output 1-channel occupancy logits `[B, 64, 16, 16]`.
- **Spatial Grid & Halo Expansion**: Increased spatial halo to `halo: 9` (34x34 grid input -> 22x22 2D features -> 16x16 3D features) to support valid 3D convolutions. Updated `_check_conv_arithmetic` to check `halo == num_conv_layers + num_conv3d_layers`.
- **Volumetric Loss Functions**:
  - `OccupancyLoss`: Weighted Binary Cross-Entropy loss focusing weight on solid/air transition boundaries in the target 3D band.
  - `OverhangLoss`: Differentiable MSE matching predicted per-chunk overhang cell count against target using differentiable `cummax` along y.
  - `ConsistencyLoss`: Penalizes discrepancy between `TerrainHead` height prediction and soft-argmax topmost solid cell in predicted occupancy band.
  - Updated `CombinedLoss` to support `occupancy_weight`, `overhang_weight`, `consistency_weight` and return `'occupancy'`, `'overhang'`, `'consistency'` in `components_dict`.
- **YAML Configs & Test Suite**: Updated `config.cuda.yaml`, `config.mps.yaml`, and `config.rehearsal.yaml` with `halo: 9`, `num_conv3d_layers: 3`, and default 3D loss weights (`0.0`). Updated `test_model.py` with expanded weight declarations, halo 9 arithmetic tests, and forward/backward dry-run test.
- **Specification & Documentation**: Updated `docs/model-spec.md` to version `0.6.0` documenting redefined `heightmap`, `block_volume` non-derivability, and `capabilities.caves: true`.

- **Gate 2 — learnability and capacity sweep, rehearsal scale (`docs/phase4-plan.md` §7, §9)**: 400
  regions (368 train / 32 val), one-factor-at-a-time sweep of `occupancy_weight`, `overhang_weight`,
  `consistency_weight` around the shipped defaults, plus a `conv_channels ∈ {192, 256, 320}` capacity
  sweep. 14 short training runs total, each evaluated with `scripts/diagnose_overhangs.py
  --checkpoint` against the full 400-region ground truth, plus two ad hoc measurements
  (`relief_retention` and `biome_accuracy`) that no existing tool computes. **Verdict: Gate 2 FAILS,
  uniformly, across every configuration tested — do not start Gate 3 on this backbone/loss
  combination without further diagnosis.**

  **Two real bugs found and fixed/worked around before the sweep could run at all:**
  - `train.py`'s `build_targets()` never forwarded `batch["occupancy_band"]` into the loss's targets
    dict, so `occupancy_weight`/`overhang_weight` were silent no-ops (always a zero loss regardless
    of weight). Fixed — `build_targets` now forwards the band when present, and step/val logging
    prints all seven `CombinedLoss` components instead of the original three.
  - **`train.py`'s checkpoint-saving code only runs after a full epoch completes**, but `--max-steps`
    (used for every prior rehearsal/smoke run in this repo, including Phase 3.1's `run_sweep.sh`)
    `return`s from *inside* the per-epoch loop the instant `global_step >= max_steps` — before the
    post-loop code that writes `epoch_NNN.pt`/`best.pt`/`last.pt` is ever reached. At the shipped
    `config.mps.yaml` batch size (64), one epoch over the 368-region train split is 575 steps, so
    every `--max-steps 400` run in this sweep would have written **zero checkpoints** — training
    would log normally for 400 steps and leave nothing on disk to evaluate. Not a hypothetical: this
    is exactly what `--max-steps 400` against the shipped config does. Worked around, without
    touching `train.py` (out of scope for this pass — see task constraints), by setting
    `batch_size: 92` (`36800 / 92 = 400.0` exactly) and `num_epochs: 1` in every sweep config and
    omitting `--max-steps` entirely, so one full epoch *is* 400 steps and the loop exits into the
    checkpoint code naturally. All 14 runs share this identical batch size, so the OFAT comparison
    is still apples-to-apples; it does mean these runs are not bit-for-bit "the shipped
    `config.mps.yaml` at batch 64," and `train.py`'s checkpoint gating is a real defect worth fixing
    before anyone else reaches for `--max-steps` on a config whose epoch is longer than the requested
    step count.

  **The headline result: 0.0000% multi-run columns on every one of the 14 trained checkpoints**,
  regardless of `overhang_weight` (swept 0.0003–1.0, log-spaced per the scale finding below),
  `occupancy_weight` (0.25–2.0), `consistency_weight` (0.05–0.5), or `conv_channels` (192/256/320).
  Every checkpoint's occupancy band thresholds back to a pure single-transition heightfield —
  zero overhang cells, zero cavities of any height, zero walkable voids. This is not a marginal miss
  on the go/no-go margin; it is the literal floor on every metric `diagnose_overhangs.py` computes.
  Full per-run table (`multi_run_column_pct` is 0.0000% on all 14; `relief_retention`/`biome_accuracy`
  are the two ad hoc metrics, methodology below):

  | Run | `occupancy_weight` | `overhang_weight` | `consistency_weight` | `conv_channels` | Relief retention | Biome accuracy |
  |---|---|---|---|---|---|---|
  | baseline (shipped defaults) | 1.0 | 1.0 | 0.25 | 192 | 22.00% | 62.79% |
  | overhang_0.0003 | 1.0 | 0.0003 | 0.25 | 192 | 20.91% | 75.72% |
  | overhang_0.001 | 1.0 | 0.001 | 0.25 | 192 | 16.43% | 75.07% |
  | overhang_0.003 | 1.0 | 0.003 | 0.25 | 192 | 20.74% | 77.57% |
  | overhang_0.01 | 1.0 | 0.01 | 0.25 | 192 | 19.82% | 79.35% |
  | overhang_0.03 | 1.0 | 0.03 | 0.25 | 192 | 18.63% | 76.43% |
  | occupancy_0.25 | 0.25 | 1.0 | 0.25 | 192 | 18.71% | 51.05% |
  | occupancy_0.5 | 0.5 | 1.0 | 0.25 | 192 | 15.77% | 51.38% |
  | occupancy_2.0 | 2.0 | 1.0 | 0.25 | 192 | 18.52% | 56.00% |
  | consistency_0.05 | 1.0 | 1.0 | 0.05 | 192 | 19.64% | 81.95% |
  | consistency_0.1 | 1.0 | 1.0 | 0.1 | 192 | 17.12% | 55.28% |
  | consistency_0.5 | 1.0 | 1.0 | 0.5 | 192 | 16.16% | 47.10% |
  | capacity_256 | 1.0 | 1.0 | 0.25 | 256 | 18.64% | 60.82% |
  | capacity_320 | 1.0 | 1.0 | 0.25 | 320 | 18.16% | 45.61% |

  **The scale finding is confirmed, concretely, in the training logs.** With shipped defaults, the
  baseline run's loss components at step 150 show a live instability: `terrain` jumped 3.8→46.1,
  `biome` 2.1→10.7, while `overhang` crashed 239→9.1 in the same step. `overhang_weight=1.0` on an
  unnormalized per-chunk-count MSE (range ~0–64) so dominates the post-`clip_grad_norm_` gradient
  direction that the optimizer sacrifices terrain/biome quality chasing the overhang term down — the
  exact mechanism the task briefing predicted from the dry-run numbers (`terrain=5.3, ...,
  overhang=243.0` at step 0). The low-`overhang_weight` runs (0.0003–0.03) show no such spike and
  post consistently higher biome accuracy (75.1–79.4%) than the shipped-default baseline (62.8%),
  confirming the direction of the fix even though it does not (yet) buy any overhang structure.
  **The shipped `overhang_weight: 1.0` default in all three configs remains wrong** and should not
  ship as-is; `overhang_weight ∈ [0.003, 0.01]` is the best-supported starting point from this data
  for a follow-up sweep (relief 19.8–20.7%, comparable to baseline's 22.0%; biome 77.6–79.4%, clearly
  better than baseline's 62.8%).

  **Why "pick the knee" (§3.4) could not be done as instructed, and what was done instead.** A knee
  is a point on a tradeoff curve between overhang productivity and relief/biome retention. With
  overhang productivity pinned at exactly 0.0000% across the entire tested grid, there is no curve —
  every point is the same point on the axis that matters. The `occupancy_weight`/`consistency_weight`
  sweeps are additionally confounded: per the plan's own one-factor-at-a-time design, both were run
  holding `overhang_weight` at its shipped (broken) default of 1.0, so both inherit the same
  gradient-clip-domination instability the `overhang_weight` sweep exists to diagnose. That the
  `occupancy_weight=1.0`/`consistency_weight=0.25` point (i.e. the shared baseline) scores best
  within each of those two sweeps is more likely an artifact of that shared confound than evidence
  that 1.0/0.25 are themselves good values — a clean read on `occupancy_weight` and
  `consistency_weight` needs a re-sweep at a corrected `overhang_weight`, not this data.
  `consistency_weight=0.05` is worth flagging as a lead (81.95%, the best biome accuracy of all 14
  runs) but is equally confounded and unconfirmed.

  **Capacity (`conv_channels ∈ {192, 256, 320}`, shipped-default weights): no learnability benefit
  at this scale**, and if anything relief retention was highest at 192ch (22.00%) vs 256ch (18.64%)
  and 320ch (18.16%) — plausibly single-run noise rather than a real capacity effect (one run per
  point, no seed averaging), but there is no positive signal for growing capacity yet either.
  Per-chunk inference latency (this Mac's CPU/MPS, **not** the target mid-range GPU
  `project-outline.md` §2.2 specifies), median of 20 reps at batch=1: 192ch 17.0ms (MPS) / 35.1ms
  (CPU); 256ch 11.5ms / 23.9ms; 320ch 12.7ms / 26.2ms. All three are comfortably under the 100ms
  target (in fact under 40ms even on CPU); the non-monotonic ordering across widths is within
  measurement noise on this hardware, not a reliable capacity-vs-latency curve. **Recommendation:
  stay at 192ch** — no measured benefit to the extra params yet, and no latency pressure forcing a
  decision either way; revisit if a longer/Gate-3-scale run shows an occupancy-head capacity ceiling.

  **A calibration note on go/no-go condition 1, separate from the model's result.** Measured against
  the full 400-region ground truth (not a 32-region sample): ground truth's own multi-run column rate
  is 2.9833%, against a marginal baseline of 0.0000% — a +2.98 point margin, which is itself **below**
  the default 5.0-point bar `--baseline-margin` sets. A model that reproduced this rehearsal dataset's
  ground truth exactly would not clear condition 1 at its default margin. This does not change the
  Gate 2 verdict here (the trained checkpoints scored +0.00 points, not +2.98), but it is worth
  recording before anyone tunes toward condition 1 in isolation on this dataset — the 5.0-point
  default may be calibrated against a different (denser) archetype mix than this rehearsal set's.

  **Two conditions computed by hand, not by `diagnose_overhangs.py`'s own verdict.** The tool's
  `evaluate_baseline_gate()` implements only 3 of the plan's 5 go/no-go conditions (multi-run margin,
  walkable-void *count* retention, p90 cavity height) — by its own docstring, it explicitly leaves out
  condition 3 (mean floor area retention, added later in §7 specifically to defeat the
  void-shattering adversarial fixture) and condition 5 (relief retention, "measured by the terrain
  tooling, not this one"). Both were computed manually for every run in this sweep: condition 3 from
  `floor_cells / void_count` on both sides of the JSON output (note `aggregate_overhang_metrics`,
  used whenever ground truth is a multi-region directory, omits the `mean_floor_cells_per_void` key
  the single-region path provides — it is not recomputed from the pooled totals, so it must be
  derived rather than read); condition 5 via the ad hoc `relief_retention` script below. Both were
  moot here since conditions 1/2/4 already fail outright on every run (0 voids of any kind), but a
  future near-miss sweep should not trust the tool's printed `VERDICT` line as the full 5-condition
  answer.

  **Two metrics with no existing tool, measured ad hoc for this sweep** (methodology recorded here
  since no script owns it): **relief retention** — mean per-chunk height-std of the prediction over
  the 3,200-sample held-out validation split, divided by the same over the ground-truth target,
  mirroring what `ReliefLoss` optimizes and how the v1.1.0 ~21% figure in `docs/TRAINING.md` was
  originally produced. **Biome accuracy** — per-cell `argmax(biome_logits) == biome_grid` over the
  same validation split; `diagnose_speckle.py` measures a prediction's own spatial coherence, not
  agreement with a target, so this is a new (simple, honest) measurement, not a replacement for that
  tool. Neither v1.1.0 nor v1.1.1 has a recorded biome-accuracy baseline to regress against.

  **Disk**: the 14 sweep checkpoint directories total ~5.1 GB under `model/training_runs/gate2_*`
  (already gitignored); the full `model/training_runs/` tree (including pre-existing `run1`,
  `mps_run1`, `rehearsal`) is 6.7 GB.

  **Recommendation:** do not start Gate 3. The rehearsal-scale occupancy head has not demonstrated
  it can produce a second solid/air transition in *any* column under *any* tested weight or capacity
  configuration — the loss curves (occupancy loss falling from ~0.72 to ~0.03–0.07 within 250–300
  steps, well below what a purely marginal-collapsed predictor could achieve) suggest the head is
  learning *something* real (very plausibly reconstructing the single dominant transition already
  implied by the heightmap head, which is a much easier and much more common target than the rare
  second transition), not that learning has plateaued — but 400 steps, half of them inside
  `warmup_steps: 200`, was not enough to observe whether genuine multi-run structure ever emerges.
  Before spending Gate 3's 8000-region/9-hour CUDA budget: run one longer confirmatory sweep point
  (order of 2,000–4,000 steps, `overhang_weight` in the recommended 0.003–0.01 range) to see whether
  multi-run columns leave zero given more optimizer steps, or whether the collapse is structural
  (loss design, initialization, or the `OverhangLoss` stationary point discussed in §3.2 turning out
  to be reachable in practice despite the docstring's argument that it shouldn't be) and needs a
  design change rather than more steps.

- **Phase 4.1 §2.2 — the confirmatory run (2026-08-05)**: one longer run at the corrected
  `overhang_weight: 0.005`, 400 regions / 3,000 steps (`config.mps.yaml`, MPS, `consistency_weight`
  also now 0.0 per `docs/remote-training-readiness-plan.md` §2.H1/§5 item 0.2 — see below),
  evaluated with `scripts/diagnose_overhangs.py --checkpoint training_runs/mps_run1/last.pt
  --ground-truth model/data`.

  **Result: multi-run columns are still exactly 0.0000% at 3,000 steps** (7.5x the original 400-step
  Gate 2 sweep length, and 2,500 steps past `warmup_steps: 200`). Zero overhang cells, zero
  cavities, zero walkable voids — an identical floor to every one of the 14 original sweep
  checkpoints. Full 5-condition gate: `1. multi-run margin +0.0000 pts (need >=5.00) NOT MET`,
  `2. walkable-void retention 0.00% NOT MET`, `3. mean floor area retention 0.00% NOT MET`,
  `4. p90 cavity height 0 blocks (need >=3) NOT MET`. VERDICT: FAIL.

  **The occupancy loss curve is a plateau, not a slow descent.** It falls 0.7215 (step 0) -> ~0.025
  by step ~450 (still inside epoch 0), then holds in the 0.016-0.05 band for the remaining ~2,550
  steps with no further improvement — sampled at steps 450, 925, 1,675, 2,325, 2,675, 3,150: 0.0246,
  0.0159, 0.0235, 0.0258, 0.0320, 0.0242. This settles the question `docs/phase4.1-plan.md` §2.2
  posed as still-open after the original 400-step run ("some evidence for undertrained, not yet
  distinguishable from the data on hand"): 3,000 steps is long enough that a genuinely-still-learning
  loss would show *some* further descent, and this one does not. Per §2.2's own decision point —
  *"if multi-run columns are still exactly 0.0000% at this length, treat that as a real negative
  result, not a 'needs more steps' excuse to keep extending"* — **this is Step C's zero-result
  branch**: proceed to §2.3's structural diagnosis list (occupancy head init bias; gradient flow
  through the y-upsample; cell-level transition-weighting magnitude; capacity, only after the first
  three) before any further training spend. Not executed here — §2.3 is a separate diagnosis task
  with its own scope, and `docs/remote-training-readiness-plan.md` explicitly does not own
  relitigating Phase 4.1 (see that document's header). **Gate 2 has no verdict yet and Gate 3 remains
  correctly not started** — this result does not change that, it just closes off "the confirmatory
  run hasn't been tried" as an open question.

  Checkpoint: `model/training_runs/mps_run1/last.pt` (123.4 MB, matches the measured checkpoint size
  elsewhere in this log). Full diagnostic JSON and training log kept alongside this run for §2.3's
  future diagnosis pass.

## [v1.1.1] - Phase 3.1 prep (2026-07-30)

Gate 1 groundwork for the v1.1.1 release (`docs/phase3.1-plan.md`), done on the dev machine ahead
of the real training run. Training, the `relief_weight` sweep, and export still have to happen on
the CUDA machine before this becomes a tagged `v1.1.1` release.

### Fixed & Improved
- **Rehearsal Training Config**: Added explicit `relief_weight: 10.0` and set `mixed_precision: false` under `hardware:` in `config.rehearsal.yaml`. Updated header comments for fast smoke testing (`num_epochs: 1`) paired with `--regions 400 --max-steps 50`.
- **Config Validation Test**: Added `test_configs_declare_relief_weight()` in `model/tests/test_model.py` to assert that `config.cuda.yaml`, `config.mps.yaml`, and `config.rehearsal.yaml` all explicitly define `relief_weight` under `training`.
- **Loss Component Dict**: Updated `CombinedLoss.forward` in `losses.py` to return `(total_loss, components_dict)` containing keys `'terrain'`, `'biome'`, `'relief'`, and `'slope'`.
- **Training & Validation Loss Logging**: Updated `train.py` training and validation loops to unpack `(loss, loss_dict)`, log individual loss components (`terrain`, `biome`, `relief`) in step logs and epoch validation metrics, and added CLI flag `--relief-weight`.
- **Packaging CLI Enforcements**: Made `--version` a required CLI argument in `build_mcim.py` and `package_mcim.py`.
- **Hyperparameter Sweep Script**: Created executable `run_sweep.sh` to sweep `relief_weight ∈ {5, 10, 25, 50, 100}` with `--regions 400 --max-steps 400`.
- **Documentation & Comments**:
  - Corrected doc comments in `world_generator.py` to refer to 11 archetypes.
  - Updated resume example comments in `config.cuda.yaml` and `docs/TRAINING.md` to point to `cuda_run3`.
  - Updated baseline metrics table in `docs/TRAINING.md` with measured Phase 3 / v1.1.0 relief retention metric (~21% of target, stable).

## [v1.1.0] - 2026-07-28

### Added & Changed
- **Relief Supervision**: Introduced `ReliefLoss` to match within-chunk height standard deviation against ground-truth targets, mitigating terrain flattening under MSE.
- **Hardware & FP32 Stability**: Disabled fp16 mixed precision on CUDA configs due to GradScaler fp16 underflow/overflow causing gradient clipping during extended training runs.
- **Config Hierarchy**: Split hardware/training settings into platform-specific configs (`config.cuda.yaml`, `config.mps.yaml`, `config.rehearsal.yaml`).
- **Relief Baseline**: Measured stable relief retention at ~21% of target across training runs.

## [v1.0.0] - 2026-07-20

### Initial Release
- **Initial Model Architecture**: Implemented `ImagineNet` with frozen MiniLM prompt encoder, 2D convolutional backbone, and heads for heightmap, surface profiles, water levels, and biomes.
- **Procedural Data Sourcing**: Implemented `ProceduralWorldSource` for generating multi-region synthetic training data from Perlin noise archetypes.
- **Export & Mod Packaging**: Implemented `build_mcim.py` and `package_mcim.py` to export PyTorch models to ONNX Runtime-compatible `.mcim` packages for mod integration.
