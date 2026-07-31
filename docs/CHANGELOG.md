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
