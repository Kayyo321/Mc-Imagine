"""
Module for procedural training-data generation ("procedural bootstrap" — PROJECT.md's data
sourcing strategy 1) and (separately, Phase 8, not implemented here) headless Minecraft world
generation for the real-`.mca` data source.

`WorldGenerator` below is the Phase-8 stub (headless-server sourcing) and is intentionally left
untouched — it isn't needed for the procedural bootstrap this phase implements.

`ProceduralWorldSource` is the Phase 4 deliverable: it samples a physically-motivated parameter
vector per macro-region (512x512 blocks, aligned to the `.mca` region unit per docs/model-spec.md
"Runtime Architecture", even though nothing here reads a real `.mca` file), and renders a
heightfield + per-column surface-profile map + per-column biome map for that region using
deterministic fBm / ridged-multifractal / domain-warped noise, implemented directly on top of
numpy (a from-scratch vectorized Perlin-noise kernel — no external noise library), plus a light
`scipy.ndimage.gaussian_filter` smoothing pass on the derived rock-exposure mask.

The noise is **exactly tileable with period `spec_constants.WORLD_PERIOD` (2048 blocks) in both
axes** — every octave's lattice wraps, and every octave's frequency is quantized so its period is
an exact integer number of cells. That number is not arbitrary: it is the largest wavelength in
`model/positional.py`'s `CoordinateEncoder`, whose Fourier features are therefore 2048-periodic
too. Terrain that was *not* 2048-periodic would hand the network identical inputs with different
targets, and MSE answers conflicting targets with their mean — flat terrain. See
`spec_constants.WORLD_PERIOD` for the full argument and the accepted trade-off.

Phase 4 adds `_hash_grad3` / `_perlin3d` / `_fbm3d`, the volumetric siblings of the 2D kernels, and
`_render_occupancy_band`, which uses them to carve the surface-anchored occupancy band
(`spec_constants.BAND_HEIGHT`) that gives the model overhangs, arches and cave mouths. They obey the
same horizontal periodicity contract — and only the horizontal one: y is bounded, directly available
to the network, and deliberately not wrapped. `render_region` now returns that band alongside the
heightfield, and the heightfield it returns is *the topmost solid cell of the band* rather than an
independent quantity (docs/phase4-plan.md §1.4) — true by construction, see
`_render_occupancy_band`.

Each region is rendered on a (512 + 2*HALO_MARGIN) canvas so that every one of the 32x32 chunks in
the region — including edge chunks — can be sliced with its full 4-cell halo (see
docs/model-spec.md's coordinate-conditioning design, `model/positional.py`'s `CoordinateEncoder`)
without any extra padding logic at slice time.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from mc_imagine_model.spec_constants import (
    BAND_BOTTOM_OFFSET,
    BAND_HEIGHT,
    BAND_TOP_OFFSET,
    BIOME_DEFAULT_PROFILE,
    NUM_BIOMES,
    NUM_PROFILES,
    TERRAIN_CLIP_MAX,
    TERRAIN_CLIP_MIN,
    WORLD_PERIOD,
    seed_to_offset,
)

REGION_BLOCKS = 512  # 32 chunks x 16 blocks, matches vanilla .mca region granularity
HALO_MARGIN = 4  # R from docs/poc-plan.md §1b
CANVAS = REGION_BLOCKS + 2 * HALO_MARGIN  # 520

# One fixed gradient-table id for the entire project — see docs/phase2-plan.md §1a. Every octave
# offsets deterministically from this, so the noise field is identical everywhere and the only thing
# a world seed changes is *where in that field* you are standing.
CANONICAL_NOISE_SEED = 20260728

# Negative excursion of `shaped` (see render_region) at which the erosion carve reaches full
# strength. This constant is what makes the Phase-2 valley carving actually *do* something.
#
# Measured over 160 sampled regions: `shaped` is strongly right-skewed, because the ridged
# component `1 - |perlin|` has mean +0.50 once remapped to [-1, 1] (measured +0.5017, stable across
# frequency) while the plain fBm component has mean 0. For the ridge-heavy archetypes that are
# supposed to produce canyons, `shaped` therefore spans roughly [-0.22, +0.86] rather than
# [-1, +1]. An unnormalized `clip(-shaped, 0, 1)**2` mask consequently peaked at 0.22**2 = 0.048,
# so the carve term delivered ~3 blocks instead of the ~55 it was written to deliver, and the
# global minimum height over 160 regions was 34.4 — *no better than the Day-1 baseline of 27.7*
# (docs/phase2-plan.md §0). Dividing by the realistic negative excursion lets the mask reach 1.0 at
# the true lows: global min drops to 5.6, 56% of regions dip below sea level and 4% below y=20,
# while peaks are untouched (the mask is zero for positive `shaped`).
#
# It must stay a FIXED constant, never a per-region statistic such as `-shaped.min()`: a
# region-dependent normalization would reintroduce exactly the region-to-region discontinuity that
# docs/phase2-plan.md §1a exists to remove.
VALLEY_MASK_SCALE = 0.25

# Slope (|grad h|, blocks of height per block of horizontal distance) at which `render_region`'s
# rock-exposure term saturates — i.e. "as steep as this generator's terrain ever gets".
#
# This replaces a per-region `slope / (slope.max() + 1e-6)`. That normalizer was a *per-region
# statistic*, and it fails for the same reason `-shaped.min()` would (see VALLEY_MASK_SCALE above):
# two regions whose terrain meets at a shared boundary can have different `slope.max()`, so the
# identical boundary column gets divided by different numbers, crosses the 0.35 stone threshold on
# one side and not the other, and leaves a visible material seam exactly at the region edge. It also
# inverted the intended physics: the steeper a region was overall, the *smaller* every column's
# normalized slope became, so the most mountainous regions were the least likely to expose rock.
#
# Measured over 200 regions (only those with relief_amplitude > 25, the ones the branch runs on),
# pooled across all columns rather than per region: p50 0.57, p90 2.78, p99 5.40, p99.5 6.28,
# p99.9 9.38, pooled max 31.3. 6.0 sits at roughly the 99.4th percentile, so the term reads ~1.0 on
# genuine cliff faces and stays proportional everywhere below that. Paired with an explicit clip to
# [0, 1] so the rare steeper-than-6 column saturates instead of overshooting the score.
#
# Calibration, measured on the same 200 regions: the old per-region normalizer labelled 0.061% of
# columns `stone`, i.e. the rock-exposure override was very nearly inert (dividing by the region's
# own maximum meant only columns steeper than ~58% of that maximum could clear the 0.35 threshold,
# and the sigma-1.5 blur then erased most of those). At a fixed 6.0 the share is 0.885% — a real
# but still-sparse dusting of exposed rock on cliffs and summits, which is what profile id 4 is for
# and what makes it learnable at all. For reference: 4.0 -> 4.9%, 3.0 -> 10.3%, 2.0 -> 18.4%, which
# would repaint whole mountainsides in stone.
SLOPE_NORM_SCALE = 6.0

# --- 3D gradient table (Phase 4 — docs/phase4-plan.md §2.1) --------------------------------------
# The 12 cube-edge vectors, i.e. Perlin's standard "Improving Noise" table: every permutation of
# (±1, ±1, 0). They are chosen over a set of random unit vectors for two reasons that matter here:
# every component is in {-1, 0, +1}, so the corner dot products `gx*dx + gy*dy + gz*dz` are computed
# with multiplications that are *exact* in float64 (no rounding at all), which is one of the things
# that makes the WORLD_PERIOD-periodicity below bit-exact rather than merely close; and the 12
# directions are evenly spread over the sphere, so the field has no preferred axis, which a naive
# 3D extension of `_hash_angle`'s single-angle scheme cannot give you (a single angle parameterizes
# a circle, and a circle embedded in 3-space is a plane of gradients — the resulting "3D" noise is
# visibly layered along whichever axis got left out).
#
# Split into three separate contiguous 1-D component arrays rather than kept as a (12, 3) table
# because the lookup is `np.take(component, gradient_index)`: taking from a (12, 3) table produces a
# (..., 3) array, which for a [8, 520, 520] slab is a 52 MB allocation per corner (8 corners = 415
# MB of pure temporary) and then needs slicing anyway. Three scalar takes cost the same total bytes
# but let each component be consumed and freed immediately.
_GRAD3 = np.array(
    [
        (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
        (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
        (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
    ],
    dtype=np.float64,
)
_GRAD3_X = np.ascontiguousarray(_GRAD3[:, 0])
_GRAD3_Y = np.ascontiguousarray(_GRAD3[:, 1])
_GRAD3_Z = np.ascontiguousarray(_GRAD3[:, 2])

# Amplitude normalizer for `_perlin3d`, the 3D counterpart of `_perlin2d`'s `* 1.4142`. It is 1.0,
# which is a deliberate choice rather than an oversight — named so nobody "fixes" the missing
# multiply, and so the caveat below travels with it.
#
# `_perlin2d` uses UNIT-length gradients (cos/sin of a hashed angle). 2D Perlin from unit gradients
# is bounded by 1/sqrt(2) *by construction*, so `* 1.4142` maps it onto [-0.99995, 0.99995]: that is
# a proof, and `_perlin2d` genuinely cannot leave [-1, 1]. The cube-edge table above is not
# unit-length — each vector has length sqrt(2) — and that extra factor is roughly what a 3D
# normalization would otherwise have divided back out, which is why 1.0 is the right ballpark.
#
# BUT `_perlin3d` HAS NO SUCH BOUND, AND THIS IS NOT SYMMETRIC WITH THE 2D CASE. The field value is
# `sum_c w_c * (g_c . d_c)` over the 8 corners, the fade weights `w_c` are non-negative and sum to
# 1, and each corner's gradient is chosen independently by the hash — so the supremum over all
# possible gradient assignments is separable and exactly computable as
# `max over (sx,sy,sz) of sum_c w_c * max_g (g . d_c)`. That maximization gives
#
#     sup |_perlin3d| = 1.036354, attained at cell-relative (0.35526, 0.48149, 0.50000)
#
# and it is a property of the gradient table and the fade polynomial, not of the hash — changing
# the hash moves which cells realize it, not the value. Against the hash actually implemented,
# adversarial search (dense grid + Nelder-Mead, 9 seeds x 7 lattice periods) reaches 1.0045.
#
# The measured *typical* scale is much smaller: over 60,000,000 uniformly sampled positions at a
# 19-cell lattice, ZERO exceeded 1.0, sample max 0.9290, and an earlier 4M-sample run gave
# p99.99 = 0.8675, p99.9 = 0.7531, std = 0.2713 (`_perlin2d` on the same sampling: max 0.9037,
# p99.9 0.8136, std 0.3034). An earlier revision of this comment inferred "the field occupies
# [-1, 1]" from exactly that kind of sampling. Uniform sampling cannot establish a supremum; the
# claim was wrong even though every number in it was right.
#
# CHOSEN: document, do not clip. Clipping costs a full extra pass over 17.3M cells per octave and
# would introduce a flat spot precisely where the field is most extreme, destroying the C1
# smoothness that makes cavity walls look carved rather than faceted — in exchange for at most 3.6%
# of headroom on a set that 60M uniform samples never once hit. The consequences a caller must
# actually honour:
#   - `_fbm3d` output can marginally exceed [-1, 1]; do not assume a hard bound when mapping it to
#     a carve strength. Clip at the call site if a downstream step needs one.
#   - `ridged = 1.0 - np.abs(layer)` can therefore emit as low as -0.036354, where ridged noise is
#     conventionally [0, 1]. Any threshold or sqrt() applied to ridged output must tolerate a small
#     negative, or clip explicitly.
#
# Downstream carve thresholds are calibrated against this scale, so if the gradient table or the
# fade polynomial is ever changed, redo the envelope maximization above and update this block.
PERLIN3D_NORM = 1.0

# --- 3D carve parameters (Phase 4 — docs/phase4-plan.md §2.1/§2.2(b)) ----------------------------
# What follows is the *contract* between this file's parameter schema and the band rasterizer that
# consumes it. The rasterizer is deliberately not written yet; everything a caller has to honour is
# stated here so it can be implemented from this block alone.
#
# ONE SHARED 3D FIELD, ONE OPERATOR, THREE GATES. docs/phase4-plan.md §2.1 lists four feature
# generators; naively that is four `_fbm3d` evaluations, and `_fbm3d`'s docstring measures one
# full-canvas 4-octave evaluation at 7.5-8.6 s. Four fields would be ~34 s/region, i.e. ~76 h
# for the 8000-region run before the skip fraction below, which is a schedule risk the plan does not
# budget for. The schema therefore commits to:
#
#   * exactly ONE `_fbm3d` evaluation per region, at `carve_frequency`. The cliff undercut, the
#     natural arch and the cave mouth are the SAME operator on it — the sublevel set
#     `gain * F < CARVE_THRESHOLD` — differing only in gain and in a 2D terrain gate plus a depth
#     window.
#   * exactly ONE extra 2D `_fbm` evaluation for the hoodoo plan field (milliseconds, not seconds).
#
# Sharing is not only the cheap option, it is the geologically right one: an arch *is* an undercut
# that has eaten through a fin, and a cave mouth is the same cavity where the hillside happens to
# open. Because all three are sublevel sets of one field at different gains, they are NESTED where
# their gates overlap — the strongest gain's blob contains the others — so an undercut and a cave
# mouth at the same place are one cavity rather than two coincident ones. That is the intended
# reading and it is why the union in the fill rule below is not double-counting. What sharing
# genuinely costs is that all three features live at one base wavelength per archetype; the 4
# octaves span an 8x range internally, which is the diversity budget available without a second
# field.
#
# **A LEVEL-SET ("sheet") OPERATOR WAS TRIED FOR THE ARCH AND IT PERCOLATES. DO NOT REINTRODUCE IT.**
# The natural way to cut a tunnel is `|F| < half-span`, and the thickness arithmetic works out: at
# median |grad F| = 0.0114/block the half-span needed for an 8-block tunnel is 0.05. Thickness is
# not isolation. MEASURED by connected-component labelling of a 256x256x64 volume at
# `carve_frequency = 5.0`:
#
#     |F| < 0.05*0.60  ->  13.2% of volume, largest component  99.8%,  10 components
#     |F| < 0.05*0.95  ->  21.0% of volume, largest component 100.0%,   4 components
#     |F| < 0.05*1.43  ->  31.3% of volume, largest component 100.0%,   9 components
#
# A level set of a smooth field near its median IS a percolating surface; thickening it gives a
# sponge, not tunnels. That is docs/phase4-plan.md §3.1's uniform-swiss-cheese failure arriving in
# the *ground truth*, where no loss function can rescue it. The sublevel set at the same field and
# the same gains behaves completely differently because it is a sublevel set of the *tail*:
#
#     gain*F < -0.35, gain 0.85 -> 0.669% of volume, largest 97.7%
#     gain*F < -0.35, gain 1.15 -> 3.362% of volume, largest 82.5%,  13 components
#
# and once the 2D gates are applied (which is how it is actually used) the largest component falls
# to 5-35% over hundreds of components — see the per-feature blocks below for the gated numbers.
#
# NO SITING/RARITY FIELD IS NEEDED, and one was considered and rejected on measurement. The concern
# was that gating "where" a feature may occur would need a second cheap 2D field. Measured column
# fractions passing the arch's existing conjunction (slope_norm >= ARCH_SLOPE_MIN AND prominence >=
# ARCH_PROMINENCE_MIN): deep_canyon 0.11%, fjord_rift 2.74%, mesa_plateaus 0.56%. That conjunction
# already IS a rare siting mask; adding a third factor on top drove the arch to exactly zero cells
# in a direct trial. The undercut and cave-mouth gates are looser (10-28% and, at
# CAVE_SLOPE_MIN, roughly 30-60% of columns) and are held in range by the gain instead.
#
# COST. `carve_strength == 0` skips the 3D pass entirely (see CARVE_STRENGTH_EPS), and the four
# flat archetypes carry 5.2 of the 14.6 total archetype weight, so 35.6% of regions by weight
# (35.2% measured over 4000 sampled regions) never touch `_fbm3d` at all. MEASURED on the
# ANCHORED-Y path this schema actually uses — full [520, 520, 64] canvas volume, 4 octaves,
# base_freq 5.0/512, wall clock and peak Python allocation:
#
#     slab  4:  8.6 s,  190 MB      slab  8:  7.8 s,  346 MB      slab 16:  7.5 s,  658 MB
#
# Applying the skip fraction: ~5.6 s/region expected, ~0.9 h for 600 regions and ~12.4 h for 8000,
# single-threaded, on top of the existing 2D pass. Four independent fields would have been ~34 s
# per carving region, ~22 s/region expected, ~49 h for 8000.
#
# WHAT THE BUILT RASTERIZER ACTUALLY COSTS (measured by task A4 on an M-series laptop, 4 regions per
# archetype, full `render_region` wall clock including the 2D pass): 5.10-6.28 s for the seven
# carving archetypes and 0.23 s for the four flat ones. Weighted by ARCHETYPE_WEIGHTS that is
# **3.62 s/region, 0.60 h for 600 regions and 8.05 h for 8000** — a third under the estimate above,
# and the difference is not a faster machine. `_render_occupancy_band` evaluates the field only over
# the union of the three depth windows (depth 3..38, i.e. 36 of the band's 64 indices), because the
# anchor cancels out of the fill rule and therefore every feature's depth window is a fixed set of
# BAND INDICES, known before any noise is sampled. 36/64 = 56% of the volume, and the measured cost
# tracks that almost exactly.
#
# THE RASTERIZER, STATED AS PSEUDOCODE, because prose left four things ambiguous enough that an
# implementer would have had to guess. Order matters: the hoodoo is a heightfield edit and runs
# FIRST, the 2D gates are computed from the PRE-hoodoo landform, and the band is filled from the
# POST-hoodoo anchor.
#
#     h              = the existing 2D heightfield, float32, already clipped to
#                      [TERRAIN_CLIP_MIN, TERRAIN_CLIP_MAX]
#
#     # --- gates, from PRE-hoodoo h.  THIS IS NOT INTERCHANGEABLE — see below. ---
#     dh_dz, dh_dx   = np.gradient(h)
#     slope_norm     = np.clip(hypot(dh_dx, dh_dz) / SLOPE_NORM_SCALE, 0.0, 1.0)
#     prominence     = h - gaussian_filter(h, sigma=ARCH_PROMINENCE_SIGMA)
#
#     # --- hoodoo: a heightfield edit (see the HOODOOS ARE A HEIGHTFIELD EDIT block) ---
#     gain_hoodoo    = min(1.0, carve_strength * hoodoo_strength)
#     thr            = HOODOO_THRESHOLD_OFF + HOODOO_THRESHOLD_SPAN * gain_hoodoo
#     Pl             = _fbm(x, z, seed + HOODOO_SEED_OFFSET,
#                           hoodoo_frequency / REGION_BLOCKS, octaves=HOODOO_OCTAVES, ridged=True)
#     ramp           = np.clip((thr - Pl) / HOODOO_SOFT, 0.0, 1.0)
#     h_carved       = np.clip(h - HOODOO_DEPTH * ramp, TERRAIN_CLIP_MIN, TERRAIN_CLIP_MAX)
#     anchor         = np.rint(h_carved).astype(np.int64)      # int, half-to-even
#
#     # --- band fill, per band index i in [0, BAND_HEIGHT) ---
#     world_y        = anchor + BAND_BOTTOM_OFFSET + i
#     depth          = anchor - world_y                        # 0 AT THE SURFACE CELL
#     solid          = depth >= 0                              # <- THE FILL RULE. Integer compare.
#     air            = undercut | arch | cave_mouth            # <- union, nothing subtracts
#     band_solid     = (solid & ~air) | (world_y < MIN_Y)
#
# Four things that were previously left to guess, now pinned:
#
#   * **The fill rule is `depth >= 0`, i.e. `world_y <= anchor`, an INTEGER comparison against the
#     rounded anchor.** NOT `world_y <= h_carved` against the float. `export_onnx.py` writes
#     `y_b <= h_b` after rounding `h` to an int for exactly this reason, and the two must agree. If
#     the float form is used, a column at h = 80.6 has anchor 81 but topmost solid 80, depth 0 is
#     air, and every depth window is silently off by one in whichever half of the columns round up.
#   * **`slope_norm` and `prominence` are computed from PRE-hoodoo `h`.** The hoodoo shave puts an
#     18-block step at every spire wall, and `np.gradient` reads that as |grad h| ~ 18, i.e.
#     slope_norm pinned at 1.0. MEASURED on three `mesa_plateaus` regions (the archetype that has
#     both hoodoos and arches at full strength), share of columns passing each gate:
#
#         pre-hoodoo  >=0.35:  9.5% / 1.8% / 13.1%      >=0.50:  3.8% / 0.1% /  5.9%
#         post-hoodoo >=0.35: 17.5% / 9.8% / 20.1%      >=0.50: 10.0% / 5.9% / 11.3%
#         post-hoodoo saturated at exactly 1.0: 1.3% / 0.9% / 1.6% of all columns
#
#     So post-hoodoo roughly doubles the eligible area and makes it a picture of the *spire walls*
#     rather than of the landform. The gates describe the terrain the features are carved into, and
#     the hoodoo is a texture applied to that terrain, so pre-hoodoo is the coherent choice.
#   * **The re-clip after the hoodoo shave is required**, even though it is not reachable in
#     practice (measured minimum h over 120 regions: 15.57, against TERRAIN_CLIP_MIN -48, so an
#     18-block shave has 63 blocks of margin). It is there so the band anchor obeys the same
#     invariant the rest of the pipeline assumes rather than depending on that margin holding.
#   * **Depth windows are INCLUSIVE at both ends**: `UNDERCUT_ROOF <= depth <= UNDERCUT_ROOF +
#     UNDERCUT_REACH` and likewise for the others. So UNDERCUT_REACH = 24 means 25 eligible cells.
#   * **The three feature masks combine by UNION and nothing ever re-solidifies a carved cell.**
#     Interaction with the hoodoo: the hoodoo has already lowered `anchor`, and the 3D field is
#     indexed by absolute world y, so a shaved column's carve window slides down 18 blocks with its
#     anchor and samples a *different* part of F than its unshaved neighbour. That is correct and
#     deliberate — the field is a property of world space, the window is a property of the surface.
#
# THE THREE COORDINATE RULES, restated because getting any of them wrong is silent:
#   1. Fold x and z with `np.mod(_, WORLD_PERIOD)` before calling `_fbm3d`, exactly as
#      `render_region` already does for the 2D path. Mathematically a no-op, numerically the thing
#      that makes aliased positions bit-identical.
#   2. Pass **absolute world y**, never a band index and never `y - h`. See `_perlin3d`'s docstring.
#   3. Band index -> world y is `world_y = np.rint(h[z][x]) + BAND_BOTTOM_OFFSET + i`, and the
#      rounding MUST be `np.rint` (half-to-even). `viz.py`'s cross-section renderer already commits
#      to `np.rint` at its `anchors = np.rint(h_line)` line; rounding one way here and the other
#      there shifts every rendered slice one block against its own heightmap, which looks entirely
#      plausible. `round()`/`np.round` are the same half-to-even rule; `int(x + 0.5)` is not.
#
# BROADCASTING NOTE — `_fbm3d`'s docstring suggests passing `x` as `(1, CANVAS, 1)` and `y` as
# `(slab, 1, 1)` so only the corner dot products are full size. **That does not apply here.** The
# band is anchored per column, so world y is a full [CANVAS, CANVAS] array for every band index and
# the y argument is `(CANVAS, CANVAS, slab)`. The slabbing still works, and the anchored path was
# re-measured on its own terms rather than inherited — see CARVE_Y_SLAB for the table. What is lost
# is only the "coordinate arrays stay tiny" saving.

# The ONE carve operator: a cell is eligible for removal where `gain * F < CARVE_THRESHOLD`.
# MEASURED on 1.5-2M uniform samples of a 4-octave `_fbm3d` (persistence 0.5). The distribution is
# frequency-independent, re-checked at every frequency this table actually uses: std 0.1653 /
# 0.1656 / 0.1659 at 10/512, 12/512, 14/512 (and 0.1666 at the 5/512 and 7/512 of the first
# calibration), observed range [-0.700, +0.700], `F < -0.35` at 1.46% / 1.49% / 1.51%. So one set of
# tail numbers describes every archetype:
#
#     F < -0.60  -> <0.005%      F < -0.30  ->  3.5%      F < -0.15  ->  18.9%
#     F < -0.45  ->  0.19%       F < -0.25  ->  6.9%      F < -0.10  ->  27.8%
#     F < -0.35  ->  1.58%       F < -0.20  ->  11.9%
#
# (The -0.60 row is not exactly zero — the observed minimum was -0.664 — just below the resolution
# of the sample. Nothing here relies on it being zero; see the gain paragraph.)
#
# **-0.35 rather than -0.20, and the reason is connected components, not volume.** The operator has
# to land deep enough in the tail that its sublevel set is a scatter of isolated blobs rather than a
# connected filament network. Measured by CCL on a 256x256x64 volume before any 2D gate: gain 1.0 at
# threshold -0.20 gives 11.9% of volume in a handful of components; at -0.35 it gives 1.75%. Once
# the 2D gates are applied — which is how it is actually used — the fully calibrated table gives a
# largest component of 14-34% of carved cells and 0.013-0.37% of the band VOLUME, over 70-180
# components per 256x256 window. Quote both shares: at these volumes "largest = 32% of carved" is
# two ordinary caves, whereas the rejected level-set operator put 21% of the whole VOLUME in one
# component. Volume alone would not have caught that, and neither would the share alone.
#
# The carve gains multiply the FIELD rather than moving the threshold. That form is chosen for one
# property: at `gain == 0` the test is `0 < -0.35`, false for every cell, so "strength 0 carves
# nothing" is a *proof* and not a measurement. This is now the only operator — the arch's
# level-set variant was removed — so the proof covers every 3D feature with no asterisk. (Moving
# the threshold instead would need it pushed below the field's infimum to be provably inert, which
# compresses the whole useful parameter range into the top fifth of [0, 1].) Effective threshold is
# `CARVE_THRESHOLD / gain`, so gain 0.85 -> -0.412 -> ~0.5% of cells before gating, gain 1.0 ->
# 1.58%, gain 1.15 -> -0.304 -> ~3.4%, gain 1.30 -> -0.269 -> ~5.7%.
CARVE_THRESHOLD = -0.35

# Below this the region is treated as having NO 3D carving at all and the `_fbm3d` call is skipped.
# The four flat archetypes are pinned to exactly 0.0, so this is an exact test in practice; the
# epsilon exists so a float blend or a future jittered range cannot produce a 1e-17 "carve" that
# costs ~8.6 s and removes nothing.
CARVE_STRENGTH_EPS = 1e-6

# Minecraft's world floor. Mirrors `export/export_onnx.py`'s MIN_Y and `viz.py`'s `_WORLD_FLOOR_Y`,
# and is deliberately re-declared rather than imported for the reason viz.py states: `export_onnx`
# imports torch at module scope, and the data pipeline that generates 8000 shards has no business
# pulling a training dependency in to learn one integer. The three describe one world, not three
# conventions, and must move together.
#
# It is needed here because the band is a RELATIVE window and can hang below the floor: a column at
# TERRAIN_CLIP_MIN (-48) puts band index 0 at -109. `spec_constants`' band block requires this file
# to render those cells as SOLID, so the target and the volume `export_onnx` actually emits agree
# about them being unreachable rock rather than disagreeing about a region the exporter drops.
WORLD_FLOOR_Y = -64

# Depth windows, in blocks below the band anchor `np.rint(h)`, for each feature's gate. Depth 0 is
# the surface cell itself. Every window starts strictly below 0, which buys an invariant worth
# stating: none of the three 3D features can remove the topmost solid cell of a column, so the
# returned `heightfield` remains *exactly* the topmost solid cell of the band and
# docs/phase4-plan.md §1.4's heightmap redefinition holds by construction rather than by a loss
# term. The hoodoo is the deliberate exception and is handled as a heightfield edit (see below) for
# precisely that reason.
#
# Windows are INCLUSIVE at both ends, so REACH = 24 means 25 eligible cells. All stay inside the
# band's 61 blocks below the anchor (deepest edge 27, 32 and 38 respectively). The tallest cavity
# the generator can produce is bounded by the UNION of the three windows, depth 3..38 = **36
# blocks**, not by the widest single window — inside the ~48-block ceiling docs/phase4-plan.md §1.2
# sets for the 64-block band. Measured max cavity height in the calibration runs below: 20-36.
# These are STARTING VALUES, not measurements; Gate 0's distribution confirms or corrects them.
UNDERCUT_ROOF = 3        # blocks of intact rock kept at the cliff lip, so an overhang has a roof
UNDERCUT_REACH = 24      # how far down the cliff face the undercut may eat, below UNDERCUT_ROOF
ARCH_SPAN_CLEARANCE = 6  # rock guaranteed above an arch tunnel, i.e. the span you walk under
ARCH_REACH = 26
CAVE_ROOF = 4            # a cave mouth needs a roof or it is a trench
CAVE_REACH = 34
HOODOO_DEPTH = 18        # blocks shaved off the top of a "floor" column, leaving spires standing

# Terrain gates. `slope_norm` is exactly `render_region`'s
# `np.clip(|grad h| / SLOPE_NORM_SCALE, 0, 1)` — which today is computed only inside the
# `relief_amplitude > 25` branch and must be hoisted so the carve can always see it.
# `prominence` is `h - gaussian_filter(h, sigma=ARCH_PROMINENCE_SIGMA)`, i.e. how far a column
# stands above its own neighbourhood: positive on a fin or ridge crest, negative in a gully. It is
# what makes an arch an arch rather than a hole in a hillside — a tunnel through a broad slope has
# no span above it to walk under.
#
# Calibration reference for the slope gates, read off SLOPE_NORM_SCALE's own measured percentiles
# (pooled columns, relief_amplitude > 25: p50 0.57, p90 2.78, p99 5.40): slope_norm 0.35 is
# |grad h| = 2.1, which sits between the 50th and 90th percentiles; 0.50 is |grad h| = 3.0, just
# past the 90th. So undercuts are restricted to steep terrain and arches to the steepest ~10%. The
# exact shares are not measured here — Gate 0's diagnostics are what will pin them down.
#
# MEASURED share of columns passing each gate (256x256 windows, one region per archetype):
#
#     archetype        slope>=0.35   slope>=0.50   prom>=4.0   slope>=0.50 AND prom>=4.0
#     deep_canyon         10.2%         0.7%         12.1%              0.11%
#     fjord_rift          27.7%         7.9%         26.8%              2.74%
#     mesa_plateaus        9.4%         4.0%          3.7%              0.56%
#
# That last column is why the arch needs no separate rarity or siting field: the conjunction is
# already a sparse, terrain-correlated siting mask. A trial that added a third 2D mask on top drove
# the arch to exactly zero carved cells in both sampled `deep_canyon` regions.
#
# CAVE_SLOPE_MIN is much looser because a cave mouth is meant to be the common feature — but it is
# NOT zero, and both facts are measured. With no slope gate at all the cave mouth's sublevel set is
# one branching network: CCL over a gated `deep_canyon` volume gave 2-17 components with 54-98% of
# carved cells in the largest. Adding the gate fragments it. Physically it is also the honest gate —
# a cavity under dead-flat ground never daylights, so it is a cave, not a cave *mouth*.
#
# **0.05 and not 0.10, because the archetypes do not share a slope distribution.** Share of columns
# passing, median over 5 regions each:
#
#     threshold   taiga_forest   tropical_archipelago   valley_mountains_craters   rolling_grassland
#      >= 0.05        17.4%             34.6%                    96.7%                    1.5%
#      >= 0.10         0.6%              1.8%                    88.1%                    0.0%
#
# At 0.10 the gentle archetypes are gated out rather than weighted low: measured 3 of 4 sampled
# `taiga_forest` regions and 2 of 4 `tropical_archipelago` regions carved exactly zero cells. At
# 0.05 both are 0 of 6. `rolling_grassland` is unaffected either way because `carve_strength` is
# pinned to 0 there — the gate is not what keeps the flat archetypes solid, and must not be relied
# on for it.
UNDERCUT_SLOPE_MIN = 0.35
ARCH_SLOPE_MIN = 0.50
ARCH_PROMINENCE_MIN = 4.0
ARCH_PROMINENCE_SIGMA = 12.0
CAVE_SLOPE_MIN = 0.05

# Hoodoo plan field: a 3-octave RIDGED 2D `_fbm` at `hoodoo_frequency`, evaluated on the same
# folded global coordinates as everything else. A column is "floor" (shaved by up to HOODOO_DEPTH)
# where the plan field is LOW; the ridged field's thin high crests are what stays standing, which
# is why it is ridged and not plain. MEASURED over 2M uniform samples at both pinned frequencies
# (16/512 and 20/512 give indistinguishable distributions): min 0.243, p10 0.588, p25 0.672,
# p50 0.763, p75 0.838, max 0.9996 — strongly skewed high, as ridged noise is.
#
#     P < 0.40 -> 0.2%    P < 0.50 -> 2.4%    P < 0.55 -> 5.8%    P < 0.75 -> ~47%
#
# Threshold map: `HOODOO_THRESHOLD_OFF + HOODOO_THRESHOLD_SPAN * min(hoodoo_gain, 1.0)`. The OFF
# value is below the infimum of ridged 2D output, so gain 0 removes nothing as a proof rather than
# as an observation. Note the bound to cite here is the **2D** kernel's, not `PERLIN3D_NORM`'s: this
# field comes from `_fbm`, whose `_perlin2d` is bounded by construction (unit gradients, `* 1.4142`
# -> |layer| <= 0.99999041), so `ridged = 1 - |layer| >= 9.59e-6` and any negative OFF value is
# provably inert. -0.05 has margin to spare. (An earlier revision of this block cited
# `1 - sup|_perlin3d|` = -0.0364 — the right shape of argument about the wrong kernel. The
# conclusion held; the citation did not.) The `np.clip(..., 0, 1)` in the depth ramp below makes the
# proof belt-and-braces rather than load-bearing, which is why it stays.
#
# The `min(_, 1.0)` clamp is not cosmetic: past a threshold of ~0.85 the floor swallows the spires
# and the feature inverts into "shave everything", and the gain is a product of two sampled ranges
# so it can exceed 1 without either factor being unreasonable.
#
# HOODOO_SOFT tapers the removal depth over this much of the field's range below the threshold, so
# spires have varied heights instead of standing on one flat cut plane:
#     removal_depth = HOODOO_DEPTH * clip((threshold - P) / HOODOO_SOFT, 0, 1)
HOODOO_THRESHOLD_OFF = -0.05
HOODOO_THRESHOLD_SPAN = 0.80
HOODOO_SOFT = 0.15

# HOODOOS ARE A HEIGHTFIELD EDIT, NOT A BAND CARVE, and that is a considered decision rather than a
# shortcut. A vertical column of removed material with no roof over it is *exactly* representable
# as a lower heightfield value — nothing about a hoodoo needs the band. Implementing it as
# `h_carved = h - HOODOO_DEPTH * ramp` before the anchor is taken keeps three things true at once:
# the band anchor stays the real surface, the returned `heightfield` stays bit-exactly the topmost
# solid cell of the emitted band (§1.4), and the feature costs zero extra 3D evaluations.
#
# Be honest about what that means for the plan's framing: on its own the hoodoo contributes NOTHING
# to the multi-run-column metric docs/phase4-plan.md §2.4 tracks, because it produces no overhangs.
# It earns its place by silhouette, and by handing the undercut and arch features thin fins to eat
# through — a badlands spire that has been undercut IS a multi-run column, but the undercut is what
# made it one.

# Canonical noise-layer ids for the new fields. `render_region` already uses seed + 101 / 202 / 303
# for the two warp components and the ridged relief layer; these continue that sequence. They are
# offsets from CANONICAL_NOISE_SEED and are deliberately NOT a function of the world seed — the
# world seed translates the canonical field (`seed_to_offset`), it does not reseed it.
CARVE_SEED_OFFSET = 505
HOODOO_SEED_OFFSET = 404

# Octave counts, and the y-slab the 3D volume must be evaluated in.
#
# `CARVE_Y_SLAB = 4`, and the deciding axis is MEMORY, not time. `_fbm3d`'s own docstring table
# (measured on the broadcast-coordinate path, which this schema does not use) reads slab 8 as the
# sweet spot. Re-measured here on the ANCHORED-Y path — full [520, 520, 64] canvas, 4 octaves,
# base_freq 5.0/512:
#
#     slab  4:  8.6 s,  190 MB      slab  8:  7.8 s,  346 MB      slab 16:  7.5 s,  658 MB
#
# Wall clock is flat to within ~13% across the range and the ordering did not reproduce between
# machines (an independent run on other hardware measured slab 4 as the *faster* one), so time is
# not a basis for the choice. Peak allocation is not ambiguous: slab 4 halves it, and the
# rasterizer holds the heightfield, the gates and the accumulating band alongside the field.
CARVE_OCTAVES = 4
CARVE_Y_SLAB = 4
# Octaves for the 2D hoodoo plan field. Named rather than left in prose so the distribution
# measured for HOODOO_THRESHOLD_OFF/SPAN below describes the field actually built.
HOODOO_OCTAVES = 3

# --- the frequency grid, and why every pinned carve frequency sits exactly on it -----------------
# Carve frequencies are quoted in CYCLES PER REGION_BLOCKS, matching `relief_frequency`'s
# convention (`render_region` does `base_freq = params.relief_frequency / REGION_BLOCKS`).
#
# `_fbm`/`_fbm3d` quantize each octave to `n_cells / WORLD_PERIOD` with
# `n_cells = max(2, round(WORLD_PERIOD * freq))`, so a requested frequency is shifted by up to
# `0.5 / (WORLD_PERIOD * freq)`. That is NOT negligible at carve scales — measured in `_fbm3d`'s
# docstring, first octave: 4.37/512 -> 2.75%, 3.10/512 -> 3.23%, 1.90/512 -> 5.26%,
# 0.70/512 -> 7.14%. The large features this release exists for (a walk-in cave mouth, an arch
# spanning a canyon) are exactly the low-frequency ones sitting in that 3-7% band.
#
# So every pinned carve frequency below is chosen by picking `n_cells` as an INTEGER FIRST and
# deriving the frequency from it, making the quantizer a genuine no-op and the pinned number the
# number actually sampled. In these units that means every carve frequency is a multiple of
# `CARVE_FREQ_GRID` = REGION_BLOCKS / WORLD_PERIOD = 0.25, which is exact in binary floating point,
# so `carve_lattice_cells` returns the integer with no rounding at all. It also holds for every
# octave: lacunarity is 2, so an integer first-octave `n_cells` stays integral all the way up.
#
# The existing `relief_frequency` values are a mixed bag here — 4.75/2.25/3.0/5.25/4.5/3.75 are
# on-grid, 3.35/3.9/2.65 are not (measured shifts 2.99%, 2.56%, 3.77%). That is committed Phase-2
# behaviour that the trained baselines were produced under and it is deliberately left alone; the
# rule applies to the new parameters only.
#
# --- AND WHY THE CARVE WAVELENGTH MUST BE COMPARABLE TO THE CAVITY, NOT TO THE LANDFORM ----------
# The first calibration of this table picked carve wavelengths of 73-102 blocks, reasoning from
# landform scale ("a canyon arch is a big thing"). That is wrong, and the way it fails is worth
# recording because nothing about it is visible in a single rendered region.
#
# The depth windows above cap any cavity at 36 blocks. A base wavelength of 102 blocks is therefore
# *larger than the largest feature the band can hold*, so the carve is not drawing cavities at its
# own scale — it is sampling a handful of field cells per region and clipping them. A 520-block
# region spans only ~5 wavelengths at 102 blocks, i.e. ~25 independent horizontal cells, and the
# operator is a threshold deep in the field's tail. Whether a given region contains a deep enough
# minimum at all is then close to a coin flip.
#
# MEASURED, pre-gate carved fraction over six `deep_canyon` regions at the same gains, sampled at
# four depths each — this is the spread BETWEEN regions that should have looked alike:
#
#     carve_frequency  5.0  (lambda 102.4 blk):  0.0000 0.0000 0.0021 0.0623 0.0177 0.1010
#     carve_frequency 10.0  (lambda  51.2 blk):  0.0110 0.0103 0.0105 0.0329 0.0012 0.0116
#     carve_frequency 14.0  (lambda  36.6 blk):  0.0091 0.0037 0.0035 0.0122 0.0048 0.0429
#
# Two of the six regions at 102 blocks carve *exactly nothing* while another carves 10% — a spread
# of 1e8, against 26x at 51 blocks and 12x at 37 blocks. A caption cannot predict which side of that
# coin flip a region landed on, so it is precisely the caption-invisible variance this whole file
# exists to eliminate, arriving through the frequency rather than through the gain. It also explains
# an earlier misdiagnosis: "cave mouths are absent from the gentle archetypes" looked like a gain
# problem and was mostly this.
#
# So the carve frequencies below are chosen so the base wavelength lands near the *cavity* scale
# (37-51 blocks against a 36-block cavity cap), not near the landform scale. Measured blob chord at
# the operating threshold, `2 * 0.35 / median|grad F|`: 30.7 blocks at f=10, 25.6 at f=12, 22.0 at
# f=14 — walk-in sized, and inside the band's reach rather than clipped by it.
CARVE_FREQ_GRID = REGION_BLOCKS / WORLD_PERIOD  # 0.25 cycles per REGION_BLOCKS


def carve_lattice_cells(freq_per_region: float) -> int:
    """`n_cells` the noise kernels will actually use for the FIRST octave of a carve frequency
    quoted in cycles per REGION_BLOCKS. Mirrors `_fbm3d`'s quantizer exactly (including its floor
    of 2) so the on-grid assertion below tests the real thing rather than a restatement of it."""
    return max(2, int(round(WORLD_PERIOD * (freq_per_region / REGION_BLOCKS))))


def carve_frequency_is_on_grid(freq_per_region: float) -> bool:
    """True iff the quantizer is a no-op for this frequency, i.e. the pinned value is the value
    actually sampled. See the CARVE_FREQ_GRID block for why that matters at carve scales.

    The `>= 2` is not redundant with `is_integer()`: `_fbm3d` floors `n_cells` at 2, so a requested
    1-cell (or 0-cell) frequency is silently replaced regardless of how integral it is."""
    cells = WORLD_PERIOD * (freq_per_region / REGION_BLOCKS)
    return cells >= 2 and float(cells).is_integer()


class WorldGenerator:
    """
    Handles headless Minecraft server execution to generate world data (Phase 8 — real `.mca`
    sourcing). Not implemented in this pass; `ProceduralWorldSource` below is the Phase 4 stand-in.
    """

    def __init__(self) -> None:
        pass

    def generate_world(self, seed: int, output_dir: str) -> str:
        """
        Generates a Minecraft world for a specific seed.

        Args:
            seed (int): The world generation seed.
            output_dir (str): The directory to save the world data in.

        Returns:
            str: Path to the generated world directory.
        """
        # TODO (Phase 8): Implement headless server launch and generation logic
        pass

    def batch_generate(self, seeds: List[int], output_dir: str) -> List[str]:
        """
        Generates multiple Minecraft worlds in batch.

        Args:
            seeds (List[int]): List of world generation seeds.
            output_dir (str): Base directory to save the worlds in.

        Returns:
            List[str]: List of paths to the generated world directories.
        """
        # TODO (Phase 8): Implement batch generation
        pass


@dataclass
class RegionParams:
    """One macro-region's generation parameter vector — the "known ground truth" that
    `data/labeler.py` turns into a caption and `ProceduralWorldSource.render_region` turns into
    actual heightfield/profile/biome rasters. Field names match docs/poc-plan.md's Phase 4 spec
    verbatim."""

    seed: int
    region_x: int
    region_z: int
    archetype: str
    base_height: float
    relief_amplitude: float
    relief_frequency: float
    ridge_sharpness: float
    plateau_quantization: float
    water_level: float
    erosion: float
    surface_profile_id: int
    biome_class: int

    # --- 3D carve (Phase 4 — docs/phase4-plan.md §2.1) ---------------------------------------
    # Two frequencies and five gains. The split is the whole point and is enforced by
    # `tests/test_model.py::test_relief_frequency_is_pinned_per_archetype`: a FREQUENCY selects
    # *which* pattern is drawn, so averaging over it annihilates the pattern and it must be pinned
    # per archetype; a GAIN rescales a fixed pattern, so the conditional average keeps its
    # structure and it may vary freely. See `sample_params`'s docstring for the measurements.

    # Cycles per REGION_BLOCKS of the single shared 3D `_fbm3d` carve field. PINNED, and always a
    # multiple of CARVE_FREQ_GRID so `_fbm3d`'s quantizer is a no-op. Consumed by the cliff
    # undercut, the arch and the cave mouth (all three read the same field). 0 is not a legal
    # value — an unused field is switched off with `carve_strength`, not with a zero frequency,
    # which would collapse to `n_cells = 2` and a near-constant field.
    carve_frequency: float
    # Cycles per REGION_BLOCKS of the 2D ridged `_fbm` hoodoo plan field. PINNED, same grid rule.
    # Sets the spacing of the spires: wavelength is WORLD_PERIOD / n_cells blocks. Consumed by the
    # hoodoo feature only.
    hoodoo_frequency: float

    # Master gain on the carve field, dimensionless. Multiplies the field before it is compared to
    # CARVE_THRESHOLD, and multiplies every per-feature gain below. 0 means NO 3D structure at all:
    # the band is the plain heightfield expansion, the `_fbm3d` call is skipped entirely, and that
    # is a proof (`0 < CARVE_THRESHOLD` is false everywhere), not a calibration. Amplitude-like.
    carve_strength: float
    # Per-feature gains, dimensionless, each multiplied by `carve_strength` before use. 0 disables
    # exactly that feature and nothing else. All amplitude-like: they may vary.
    # All three 3D features are the SAME sublevel operator on the SAME field; only the gain and the
    # gate differ.
    #   undercut_strength   — cliff undercut: gated to steep faces (slope >= UNDERCUT_SLOPE_MIN),
    #                         depth UNDERCUT_ROOF..UNDERCUT_ROOF+UNDERCUT_REACH.
    #   arch_strength       — natural arch: gated to prominent steep fins (slope >= ARCH_SLOPE_MIN
    #                         AND prominence >= ARCH_PROMINENCE_MIN), depth ARCH_SPAN_CLEARANCE..
    #                         +ARCH_REACH so a span is guaranteed above the hole.
    #   cave_mouth_strength — cave mouth: the loose gate (slope >= CAVE_SLOPE_MIN only), depth
    #                         CAVE_ROOF..+CAVE_REACH, so it daylights wherever a neighbouring
    #                         column is lower.
    #   hoodoo_strength     — hoodoo/pillar: 2D plan-field threshold, applied as a heightfield edit.
    undercut_strength: float
    arch_strength: float
    cave_mouth_strength: float
    hoodoo_strength: float


# --- Archetypes -----------------------------------------------------------------------------
# Each of the 11 archetypes is a named region of parameter space, deliberately constructed so the sampled
# parameter vectors are *discriminative* per docs/poc-plan.md's Phase 5 gate: the 6 held-out
# benchmark prompts each map onto (or very near) one of these, so the labeler's caption vocabulary
# and the world_generator's actual rendered terrain stay in lockstep. See data/labeler.py for the
# caption templates keyed to the same archetype names.
#
# `relief_frequency` is PINNED per archetype (lo == hi), and that is load-bearing — see
# `sample_params` for the full argument. Briefly: it is the one sampled parameter the caption cannot
# describe and the seed cannot reveal, and averaging a fixed-phase noise field over a range of
# frequencies annihilates its structure, so a randomized frequency makes flat terrain the
# MSE-optimal answer. `tests/test_model.py` asserts every range here stays degenerate.
#
# Phase 4 adds `carve_frequency` and `hoodoo_frequency` under the SAME rule, for the same reason one
# dimension up (docs/phase4-plan.md §2.2(b)) — and, additionally, both are chosen on the
# CARVE_FREQ_GRID so `_fbm3d`'s quantizer is a no-op and the pinned number is the number sampled.
# `n=` in the comments below is the integer first-octave `n_cells` the frequency was derived from;
# the feature wavelength is WORLD_PERIOD / n blocks.
#
# The five carve *gains* are amplitude-like and vary freely, with one hard exception that is a
# feature and not an oversight: `desert_dunes`, `rolling_grassland`, `swamp` and `savanna_plains`
# are pinned to `carve_strength=(0.0, 0.0)`. "Endless flat desert dunes" must produce a SOLID dune
# field, because the model has to learn that the absence of 3D structure is prompt-determined too.
# Do not give them a token nonzero value for variety. This is also why the carve gains are excluded
# from the 25% secondary-archetype blend (see `sample_params`): a blended gain would hand a quarter
# of all desert regions someone else's overhangs.
ARCHETYPES: Dict[str, Dict[str, Any]] = {
    "valley_mountains_craters": dict(
        biome_choices=[0, 2], profile=4, base_profile=0,
        base_height=(84, 118), relief_amplitude=(64, 104), relief_frequency=(4.75, 4.75),
        ridge_sharpness=(0.60, 0.95), plateau_quantization=(0.0, 0.2),
        water_level=(55, 74), erosion=(0.65, 0.95),
        # n=48 -> 42.7-block features. Cave mouths only: the plan's undercut/arch lists do not
        # include this archetype, and craters are a heightfield shape. Measured fully gated, 6
        # regions: 0.42% of band cells carved, largest component 25% of carved / 0.10% of volume,
        # 70 components, 34 walkable voids per 256x256 window.
        carve_frequency=(12.0, 12.0), hoodoo_frequency=(16.0, 16.0),
        carve_strength=(0.85, 1.15),
        undercut_strength=(0.0, 0.0), arch_strength=(0.0, 0.0),
        cave_mouth_strength=(0.88, 1.02), hoodoo_strength=(0.0, 0.0),
    ),
    # Phase 2: the archetype that actually answers "deep valleys going underground". High base with
    # very large amplitude and near-max erosion, so the carving term drives floors far below sea
    # level (63) while the ridges stay tall — the biggest vertical range of any archetype.
    "deep_canyon": dict(
        biome_choices=[8, 9, 5], profile=4, base_profile=3,
        base_height=(96, 130), relief_amplitude=(92, 132), relief_frequency=(3.35, 3.35),
        ridge_sharpness=(0.45, 0.85), plateau_quantization=(0.25, 0.65),
        water_level=(18, 44), erosion=(0.75, 1.0),
        # The archetype that gets all four features. n=40 -> 51.2-block carve wavelength, the
        # LARGEST here — but note that is chosen against the 36-block cavity cap, not against the
        # canyon; see the wavelength block above for why landform-scale wavelengths fail. Hoodoos
        # deliberately sparse (gain product 0.30-0.63 -> threshold 0.19-0.46 -> under ~2% of
        # columns shaved): a canyon floor with a few spires, not a badlands. Measured fully gated,
        # 6 regions: 0.63% of band cells carved, largest component 32% of carved / 0.18% of volume,
        # 135 components, 56 walkable voids per 256x256 window, 0 regions carving nothing. All
        # three 3D features fire in 6/6 regions except the arch, which fires in 6/6 here.
        carve_frequency=(10.0, 10.0), hoodoo_frequency=(16.0, 16.0),
        carve_strength=(0.85, 1.15),
        undercut_strength=(1.00, 1.20), arch_strength=(1.10, 1.30),
        cave_mouth_strength=(0.95, 1.10), hoodoo_strength=(0.35, 0.55),
    ),
    # Phase 2: drowned glacial valleys — steep walls plunging straight into deep water.
    "fjord_rift": dict(
        biome_choices=[7, 6, 10], profile=4, base_profile=2,
        base_height=(78, 104), relief_amplitude=(74, 116), relief_frequency=(3.9, 3.9),
        ridge_sharpness=(0.65, 1.0), plateau_quantization=(0.0, 0.15),
        water_level=(56, 68), erosion=(0.7, 1.0),
        # Same n=40 scale as the canyon: sea arches and undercut fjord walls are the same size of
        # thing as their canyon equivalents, and sharing the value keeps one fewer number to
        # justify. No hoodoos — glacial walls do not produce spires. Measured fully gated, 6
        # regions: 0.93% of band cells carved (the most of any archetype), largest component 34% of
        # carved / 0.37% of volume, 90 components, 40 walkable voids. Arch fires in 4/6 regions,
        # undercut in 6/6.
        carve_frequency=(10.0, 10.0), hoodoo_frequency=(16.0, 16.0),
        carve_strength=(0.85, 1.15),
        undercut_strength=(1.00, 1.20), arch_strength=(1.10, 1.30),
        cave_mouth_strength=(0.95, 1.10), hoodoo_strength=(0.0, 0.0),
    ),
    "desert_dunes": dict(
        biome_choices=[1], profile=1, base_profile=1,
        base_height=(58, 74), relief_amplitude=(3, 12), relief_frequency=(3.0, 3.0),
        ridge_sharpness=(0.0, 0.15), plateau_quantization=(0.0, 0.25),
        water_level=(-20, 20), erosion=(0.0, 0.15),
        # ZERO, and it stays zero. The whole "endless flat desert dunes" prompt is the control case
        # that teaches the model the absence of 3D structure is prompt-determined. The frequencies
        # are still real, on-grid values rather than 0.0 so nothing downstream has to special-case
        # them; they are simply never sampled, because `carve_strength == 0` skips the field.
        carve_frequency=(12.0, 12.0), hoodoo_frequency=(16.0, 16.0),
        carve_strength=(0.0, 0.0),
        undercut_strength=(0.0, 0.0), arch_strength=(0.0, 0.0),
        cave_mouth_strength=(0.0, 0.0), hoodoo_strength=(0.0, 0.0),
    ),
    # Ranges are bounded so `max(base_height) + max(relief_amplitude)` = 240 sits clear of
    # TERRAIN_CLIP_MAX (252). They previously summed to 280, and 0.008% of columns across 7/600
    # regions pinned flat against the clip — the same flat-topped-peak failure mode as Day 1's tanh
    # saturation, just arriving via np.clip instead. A ceiling-pinned plateau teaches the model that
    # "towering peaks" means "a table at y=252"; 12 blocks of structural headroom costs ~20 blocks
    # of peak height and buys a summit that actually has relief on it.
    "snow_peaks": dict(
        biome_choices=[3, 10], profile=4, base_profile=2,
        base_height=(100, 140), relief_amplitude=(72, 100), relief_frequency=(5.25, 5.25),
        ridge_sharpness=(0.70, 1.0), plateau_quantization=(0.0, 0.15),
        water_level=(40, 56), erosion=(0.3, 0.55),
        # n=48 -> 42.7 blocks: smaller than the canyon's, because an alpine undercut is a cornice-
        # scale feature. No arches — the plan's arch list is fjord/canyon/mesa, and a snow peak has
        # no thin fins for a tunnel to go through; measured 0/6 regions produce arch cells, which is
        # the gate agreeing with the gain rather than either alone. Measured fully gated, 6 regions:
        # 0.51% carved, largest 21% of carved / 0.10% of volume, 180 components, 52 walkable voids.
        carve_frequency=(12.0, 12.0), hoodoo_frequency=(16.0, 16.0),
        carve_strength=(0.85, 1.15),
        undercut_strength=(0.95, 1.15), arch_strength=(0.0, 0.0),
        cave_mouth_strength=(0.90, 1.05), hoodoo_strength=(0.0, 0.0),
    ),
    "tropical_archipelago": dict(
        biome_choices=[7, 11], profile=1, base_profile=1,
        base_height=(46, 60), relief_amplitude=(20, 38), relief_frequency=(4.5, 4.5),
        ridge_sharpness=(0.0, 0.25), plateau_quantization=(0.0, 0.1),
        water_level=(50, 62), erosion=(0.1, 0.3),
        # "Most land archetypes, weighted low" (docs/phase4-plan.md §2.1) — sea caves at the
        # waterline and nothing else. NOTE THE GAIN IS NOT LOW. Rarity comes from the terrain gate,
        # not from a small number: this archetype is gentle, so few columns clear CAVE_SLOPE_MIN.
        # An earlier revision expressed "weighted low" as a low gain and, because the gain is a
        # product of two uniforms against a steep field tail, that produced ABSENCE rather than
        # rarity — measured 75% of regions carving under 0.01% of cells. The gains here are centred
        # near 1.0 for every carving archetype for exactly that reason. Measured fully gated, 6
        # regions: 0.48% carved, largest 23% of carved / 0.10% of volume, 143 components, 84
        # walkable voids, 0 regions carving nothing.
        carve_frequency=(12.0, 12.0), hoodoo_frequency=(16.0, 16.0),
        carve_strength=(0.85, 1.15),
        undercut_strength=(0.0, 0.0), arch_strength=(0.0, 0.0),
        cave_mouth_strength=(1.00, 1.15), hoodoo_strength=(0.0, 0.0),
    ),
    # biome_choices are badlands-family as of Phase 2 (was [desert, savanna], which is what made the
    # Day-1 red-mesa world sprout vanilla savanna lakes — see docs/phase2-plan.md §0).
    "mesa_plateaus": dict(
        biome_choices=[8, 9], profile=3, base_profile=3,
        base_height=(72, 104), relief_amplitude=(28, 62), relief_frequency=(3.75, 3.75),
        ridge_sharpness=(0.3, 0.6), plateau_quantization=(0.6, 1.0),
        water_level=(28, 46), erosion=(0.5, 0.8),
        # The badlands archetype, and the only one where hoodoos are the point. n=56 -> 36.6-block
        # carve wavelength (the finest of the carving archetypes: mesa arches are short spans).
        # n=80 -> 25.6-block spire spacing. Hoodoo gain product 0.60-1.04 against the min(_, 1.0)
        # clamp -> threshold 0.43-0.75 -> 0.3% to ~47% of columns shaved to the badlands floor.
        # Measured fully gated, 6 regions: 0.11% carved, largest 14% of carved / 0.013% of volume
        # (the most fragmented of any archetype), 174 components, 77 walkable voids. Arch fires in
        # 5/6 regions; undercut in 0/6, which is the gain being 0 as the plan's table specifies.
        carve_frequency=(14.0, 14.0), hoodoo_frequency=(20.0, 20.0),
        carve_strength=(0.85, 1.15),
        undercut_strength=(0.0, 0.0), arch_strength=(1.10, 1.30),
        cave_mouth_strength=(0.90, 1.05), hoodoo_strength=(0.70, 0.90),
    ),
    "rolling_grassland": dict(
        biome_choices=[0], profile=0, base_profile=0,
        base_height=(58, 76), relief_amplitude=(5, 18), relief_frequency=(2.25, 2.25),
        ridge_sharpness=(0.0, 0.2), plateau_quantization=(0.0, 0.1),
        water_level=(44, 60), erosion=(0.1, 0.3),
        carve_frequency=(12.0, 12.0), hoodoo_frequency=(16.0, 16.0),  # inert; see desert_dunes
        carve_strength=(0.0, 0.0),
        undercut_strength=(0.0, 0.0), arch_strength=(0.0, 0.0),
        cave_mouth_strength=(0.0, 0.0), hoodoo_strength=(0.0, 0.0),
    ),
    "swamp": dict(
        biome_choices=[4], profile=5, base_profile=5,
        base_height=(58, 70), relief_amplitude=(3, 9), relief_frequency=(2.25, 2.25),
        ridge_sharpness=(0.0, 0.15), plateau_quantization=(0.0, 0.05),
        water_level=(48, 60), erosion=(0.2, 0.4),
        carve_frequency=(12.0, 12.0), hoodoo_frequency=(16.0, 16.0),  # inert; see desert_dunes
        carve_strength=(0.0, 0.0),
        undercut_strength=(0.0, 0.0), arch_strength=(0.0, 0.0),
        cave_mouth_strength=(0.0, 0.0), hoodoo_strength=(0.0, 0.0),
    ),
    "taiga_forest": dict(
        biome_choices=[6], profile=7, base_profile=7,
        base_height=(64, 86), relief_amplitude=(14, 36), relief_frequency=(3.0, 3.0),
        ridge_sharpness=(0.2, 0.5), plateau_quantization=(0.0, 0.2),
        water_level=(48, 60), erosion=(0.2, 0.45),
        # Land archetype, weighted low — by terrain gate, not by gain; see tropical_archipelago.
        # The occasional hillside cave mouth in the woods. This is the gentlest carving archetype
        # and the only one that still produces the occasional empty region: measured 0.18% carved,
        # 92 components, 52 walkable voids, 1 of 6 regions carving nothing. That last number is the
        # one to watch if the calibration is ever revisited — it was 3 of 4 before CAVE_SLOPE_MIN
        # and the carve wavelength were fixed.
        carve_frequency=(12.0, 12.0), hoodoo_frequency=(16.0, 16.0),
        carve_strength=(0.85, 1.15),
        undercut_strength=(0.0, 0.0), arch_strength=(0.0, 0.0),
        cave_mouth_strength=(1.00, 1.15), hoodoo_strength=(0.0, 0.0),
    ),
    "savanna_plains": dict(
        biome_choices=[5], profile=0, base_profile=0,
        base_height=(60, 80), relief_amplitude=(8, 22), relief_frequency=(2.65, 2.65),
        ridge_sharpness=(0.0, 0.25), plateau_quantization=(0.0, 0.15),
        water_level=(40, 55), erosion=(0.15, 0.35),
        carve_frequency=(12.0, 12.0), hoodoo_frequency=(16.0, 16.0),  # inert; see desert_dunes
        carve_strength=(0.0, 0.0),
        undercut_strength=(0.0, 0.0), arch_strength=(0.0, 0.0),
        cave_mouth_strength=(0.0, 0.0), hoodoo_strength=(0.0, 0.0),
    ),
}
ARCHETYPE_NAMES = list(ARCHETYPES.keys())

# The carve parameters `sample_params` must NOT blend toward a secondary archetype. Two distinct
# reasons, and both are load-bearing:
#   * the two FREQUENCIES, because docs/phase4-plan.md §2.2(b) requires them pinned to the caption's
#     archetype — a blended frequency is caption-invisible for ~25% of regions, which is precisely
#     the hole the pinning exists to close;
#   * the five GAINS, because the blend would otherwise hand `desert_dunes` up to 45% of
#     `deep_canyon`'s carve strength and put overhangs in "endless flat desert dunes". Diversity is
#     not lost: every gain still varies freely inside its primary archetype's own range.
_UNBLENDED_PARAMS = frozenset({
    "relief_frequency",
    "carve_frequency", "hoodoo_frequency",
    "carve_strength", "undercut_strength", "arch_strength",
    "cave_mouth_strength", "hoodoo_strength",
})

# Fail at import, not at Gate 0, if a pinned carve frequency drifts off the quantization grid.
# `tests/test_model.py::test_relief_frequency_is_pinned_per_archetype` asserts the same thing, but
# an import-time check also catches an interactive edit that never runs pytest — and the failure it
# guards against is invisible: the field still renders, just at a wavelength 3-7% away from the one
# written down here, so every calibration number in this file quietly stops describing it.
for _name, _spec in ARCHETYPES.items():
    for _key in ("carve_frequency", "hoodoo_frequency"):
        _lo, _hi = _spec[_key]
        assert _lo == _hi, f"{_name}.{_key} must be pinned (lo == hi), got ({_lo}, {_hi})"
        assert carve_frequency_is_on_grid(_lo), (
            f"{_name}.{_key}={_lo} is off the 1/WORLD_PERIOD quantization grid "
            f"(WORLD_PERIOD * freq / REGION_BLOCKS = {WORLD_PERIOD * _lo / REGION_BLOCKS}, not an "
            f"integer). Pick n_cells as an integer first and use n_cells * CARVE_FREQ_GRID, or the "
            f"quantizer will shift the sampled frequency away from the pinned value."
        )
del _name, _spec, _key, _lo, _hi
# Slightly upweight the 6 archetypes the Phase 5 gate benchmarks against, without starving the rest.
_GATE_ARCHETYPES = {
    "valley_mountains_craters", "desert_dunes", "snow_peaks",
    "tropical_archipelago", "mesa_plateaus", "rolling_grassland",
}
ARCHETYPE_WEIGHTS = np.array(
    [1.6 if name in _GATE_ARCHETYPES else 1.0 for name in ARCHETYPE_NAMES], dtype=np.float64
)
ARCHETYPE_WEIGHTS /= ARCHETYPE_WEIGHTS.sum()


def _region_rng(seed: int, region_x: int, region_z: int, stream: int = 0) -> np.random.Generator:
    """Deterministic per-region RNG — any macro-region can be resampled independently given
    (seed, region_x, region_z), mirroring the "lazily evaluable at any region coordinate" property
    docs/model-spec.md requires of macro.onnx, even though nothing here ships a macro.onnx.

    `stream` selects an independent RNG sub-stream (see `sample_params`, which deliberately draws
    the archetype/parameter choices and the region's *world seed* from two decorrelated streams —
    so `seed` never leaks information about which archetype/caption a region got, and only ever
    drives the specific noise realization. That decorrelation is what gives `SeedEncoder`
    something real to learn: "same caption, different seed -> different layout, same character".

    Streams in use: 0 = archetype + the Phase 2/3 parameters, 1 = world seed, 2 = the Phase 4 carve
    parameters. **New parameters go in a NEW stream, never appended to an existing one.** Appending
    is the tempting option and it silently regenerates the dataset: every draw after the insertion
    point shifts, so the archetype survives but the blend decision, the secondary archetype and
    `biome_class` do not. Measured when the Phase 4 draws were briefly appended to stream 0:
    `base_height` changed for 39.3% of 2000 regions and `biome_class` for 31.1%, with the archetype
    unchanged for all of them — the most confusing possible shape for such a bug."""
    mix = (int(seed) * 1_000_003) ^ (region_x * 20_113) ^ (region_z * 40_213 + 12345) ^ (stream * 99_991 + 7)
    return np.random.default_rng(np.array([mix & 0xFFFFFFFF, (mix >> 32) & 0xFFFFFFFF], dtype=np.uint64))


def _sample_range(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


class ProceduralWorldSource:
    """
    Synthesizes macro-region terrain directly in numpy, standing in for a headless-server +
    `.mca`-extraction pipeline this phase (Phase 4) doesn't implement. See module docstring.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def sample_params(self, region_x: int, region_z: int) -> RegionParams:
        """Samples one macro-region's parameter vector, deterministic given (seed, region_x,
        region_z). Picks an archetype, jitters within its ranges, and — 25% of the time — blends
        toward a second archetype for extra diversity so the model doesn't just memorize 11
        discrete clusters.

        **`relief_frequency` is pinned per archetype and excluded from the blend.** This is the
        third time this project has shipped flat terrain for the same underlying reason, so the
        argument is written out in full:

        The network only ever sees `(caption, chunk coords, world seed)`. The seed enters
        `render_region` solely as `seed_to_offset(seed)` — a *translation* of one canonical noise
        field — so the phase is recoverable. The parameters are not: they come from `stream=0`
        while the world seed comes from `stream=1`, decorrelated on purpose (see `_region_rng`).
        The caption names an archetype and coarse amplitude/erosion buckets, nothing more.

        Under MSE the network's optimal output is therefore `E[height | caption, coords, seed]`,
        averaged over whatever it cannot determine. Measured, per parameter, as the fraction of
        within-chunk relief surviving that average:

            amplitude only        ~100%      erosion only          ~100%
            ridge + plateau     83-100%      frequency only       20-37%   <-- annihilates it

        Amplitude, erosion, ridge and plateau all *rescale or reshape a fixed pattern*, so the
        average keeps its structure. Frequency changes *which* pattern is drawn, so averaging over
        a 2x frequency range decorrelates the fields being averaged and the mean flattens out.
        With frequency randomized the ceiling on achievable relief is ~25% of target; with it
        pinned, ~85-100%.

        Day 1 hit this with a per-region reseeded gradient table (fixed by the canonical field +
        seed offset, `spec_constants.seed_to_offset`). Phase 2 hit it again through a loss-scale
        imbalance that made the slope term a no-op. This is the same failure's third channel.

        Diversity is unaffected in the way that matters: a different seed still translates the
        field to a genuinely different landscape, and amplitude/erosion/ridge/plateau/base_height
        all still vary freely. Only feature *scale* is now fixed per archetype — which is arguably
        correct, since "mesa" and "snow peaks" describe characteristic feature sizes.

        **Phase 4 extends the same rule to `carve_frequency` and `hoodoo_frequency`**
        (docs/phase4-plan.md §2.2(b)). The argument transfers verbatim and there is reason to think
        it bites harder in 3D: a carve is a *local* feature, and averaging over its phase erases it
        faster than averaging over a smooth field's. The 20-37% figure above is the 2D measurement;
        no 3D equivalent has been measured yet, and the pinning is not waiting for one.

        The five carve *gains* (`carve_strength` and the four per-feature strengths) are the
        amplitude-like half of the split and DO vary — they rescale a fixed pattern, so the
        conditional average keeps its structure, exactly as amplitude and erosion do. They are
        nonetheless excluded from the blend for a different reason: see `_UNBLENDED_PARAMS`.
        """
        rng = _region_rng(self.seed, region_x, region_z, stream=0)  # archetype + Phase 2/3 params
        seed_rng = _region_rng(self.seed, region_x, region_z, stream=1)  # world seed only
        # THE PHASE 4 CARVE PARAMETERS GET THEIR OWN STREAM, AND THAT IS NOT TIDINESS. Appending
        # seven draws to `stream=0` leaves the archetype and the primary parameter values intact
        # (they are drawn first) but shifts everything drawn AFTER them: the `rng.uniform() < 0.25`
        # blend test, the secondary archetype, the blend weight and `biome_class`. Measured against
        # the pre-Phase-4 generator over 2000 regions at the same seed: archetype unchanged for all
        # 2000, but `base_height` changed for 786 (39.3%) and `biome_class` for 621 (31.1%). That
        # silently regenerates a third of the dataset and invalidates every trained baseline, for a
        # feature that is supposed to be purely additive. Drawing from `stream=2` leaves `stream=0`
        # bit-identical to what it was, which is the same discipline the file applies two blocks up
        # to the off-grid `relief_frequency` values it declines to "fix".
        carve_rng = _region_rng(self.seed, region_x, region_z, stream=2)
        world_seed = int(seed_rng.integers(1, 2**62 - 1))

        primary = rng.choice(ARCHETYPE_NAMES, p=ARCHETYPE_WEIGHTS)
        spec = ARCHETYPES[primary]

        def draw(spec_: Dict[str, Any]) -> Dict[str, float]:
            return {
                key: _sample_range(rng, *spec_[key])
                for key in (
                    "base_height", "relief_amplitude", "relief_frequency",
                    "ridge_sharpness", "plateau_quantization", "water_level", "erosion",
                )
            }

        values = draw(spec)
        # Drawn once, from the primary archetype only, and never blended — so `carve_rng`'s
        # consumption does not depend on whether this region happened to blend.
        values.update({
            key: _sample_range(carve_rng, *spec[key])
            for key in (
                "carve_frequency", "hoodoo_frequency", "carve_strength",
                "undercut_strength", "arch_strength", "cave_mouth_strength", "hoodoo_strength",
            )
        })
        blend_weight = 0.0
        if rng.uniform() < 0.25:
            secondary = rng.choice(ARCHETYPE_NAMES)
            if secondary != primary:
                other = draw(ARCHETYPES[secondary])
                blend_weight = float(rng.uniform(0.15, 0.45))
                for key in values:
                    # `relief_frequency` and the Phase 4 carve parameters are deliberately NOT
                    # blended — they keep the primary archetype's value. See `_UNBLENDED_PARAMS`
                    # for the two separate arguments (caption-invisible frequency for ~25% of
                    # regions; overhangs appearing in "endless flat desert dunes"), and the note
                    # below on why frequency in particular must stay a function of the caption
                    # alone.
                    if key in _UNBLENDED_PARAMS:
                        continue
                    values[key] = (1 - blend_weight) * values[key] + blend_weight * other[key]

        biome_class = int(rng.choice(spec["biome_choices"]))
        profile_id = int(spec["base_profile"])

        return RegionParams(
            seed=world_seed,
            region_x=region_x,
            region_z=region_z,
            archetype=primary,
            surface_profile_id=profile_id,
            biome_class=biome_class,
            **values,
        )

    # --- noise kernels ------------------------------------------------------------------------
    @staticmethod
    def _hash_angle(ix: np.ndarray, iz: np.ndarray, seed: int) -> np.ndarray:
        seed_mixed = int(seed) & 0xFFFFFF  # fold arbitrary-magnitude seeds into 24 bits first —
        # avoids int64 overflow below (world seeds can be up to 2**62, see world_generator's
        # decorrelated seed-stream in sample_params; 24 bits of mixing entropy is plenty here)
        # while still depending on the full seed value.
        h = (ix.astype(np.int64) * 374761393 + iz.astype(np.int64) * 668265263 + seed_mixed * 2246822519)
        h = h & 0xFFFFFFFF
        h = (h ^ (h >> 13)) * 1274126177
        h = h & 0xFFFFFFFF
        return (h.astype(np.float64) / 4294967296.0) * 2.0 * np.pi

    @classmethod
    def _perlin2d(cls, x: np.ndarray, z: np.ndarray, seed: int, lattice_period: int) -> np.ndarray:
        """Classic gradient (Perlin) noise, fully vectorized, evaluated at arbitrary (possibly
        non-integer, possibly domain-warped) coordinate arrays `x`/`z`.

        **Tileable**: the four integer lattice corners are reduced modulo `lattice_period` *before*
        they are hashed, so the gradient table repeats every `lattice_period` cells and the field
        satisfies `noise(x + lattice_period, z) == noise(x, z)` identically. Without this the field
        is aperiodic, and since `model/positional.py`'s `CoordinateEncoder` produces a feature
        vector with period `WORLD_PERIOD` blocks, every alias of a coordinate would carry *different*
        terrain under an *identical* network input — conflicting regression targets whose MSE
        optimum is their mean, i.e. flat terrain. See `spec_constants.WORLD_PERIOD`.

        `np.mod` (not C-style truncating `%`) is required: lattice indices go negative wherever the
        domain warp pushes a sample left of the origin, and a truncating remainder would map -1 and
        +1 to gradients that are not `lattice_period` apart, tearing the field along x<0 / z<0.
        `np.mod` on int64 uses floored division, so `np.mod(-1, 8) == 7`, which is what tiling means.

        Note the *offset* vectors deliberately keep the unwrapped coordinates (`x - x0` etc.): only
        the gradient lookup wraps, the interpolation geometry does not.
        """
        x0 = np.floor(x).astype(np.int64)
        z0 = np.floor(z).astype(np.int64)
        x1, z1 = x0 + 1, z0 + 1
        sx, sz = x - x0, z - z0

        x0w = np.mod(x0, lattice_period)
        x1w = np.mod(x1, lattice_period)
        z0w = np.mod(z0, lattice_period)
        z1w = np.mod(z1, lattice_period)

        def dot(ix: np.ndarray, iz: np.ndarray, dx: np.ndarray, dz: np.ndarray) -> np.ndarray:
            angle = cls._hash_angle(ix, iz, seed)
            return np.cos(angle) * dx + np.sin(angle) * dz

        n00 = dot(x0w, z0w, x - x0, z - z0)
        n10 = dot(x1w, z0w, x - x1, z - z0)
        n01 = dot(x0w, z1w, x - x0, z - z1)
        n11 = dot(x1w, z1w, x - x1, z - z1)

        def fade(t: np.ndarray) -> np.ndarray:
            return t * t * t * (t * (t * 6 - 15) + 10)

        u, v = fade(sx), fade(sz)
        nx0 = n00 + u * (n10 - n00)
        nx1 = n01 + u * (n11 - n01)
        return (nx0 + v * (nx1 - nx0)) * 1.4142  # renormalize roughly to [-1, 1]

    @classmethod
    def _fbm(
        cls, x: np.ndarray, z: np.ndarray, seed: int, base_freq: float,
        octaves: int = 5, persistence: float = 0.5, lacunarity: float = 2.0, ridged: bool = False,
    ) -> np.ndarray:
        """Fractal-Brownian-motion sum of tileable Perlin octaves, every octave sharing the *same*
        world period of exactly `WORLD_PERIOD` blocks (so their sum does too).

        An octave's world period is `lattice_period / freq` blocks, which is only exactly
        `WORLD_PERIOD` when `WORLD_PERIOD * freq` is an integer. The requested `freq` generally is
        not, so each octave's frequency is **quantized** to `n_cells / WORLD_PERIOD` for the nearest
        integer `n_cells` and the quantized value is what actually gets sampled. Rounding
        `n_cells` but sampling at the *requested* frequency would leave the period off by a
        fraction of a cell — which reads as "close enough" and silently reintroduces exactly the
        aliasing the tiling exists to remove, since the mismatch accumulates over 16-32 aliases.
        The cost is that effective frequencies land on a `1/WORLD_PERIOD` grid; at these octave
        counts that is a sub-percent shift in feature wavelength.

        `n_cells` is floored at 2 because a 1-cell period is a constant field (all four corners
        wrap to the same gradient).
        """
        total = np.zeros_like(x, dtype=np.float64)
        amp, freq, norm = 1.0, base_freq, 0.0
        for o in range(octaves):
            n_cells = max(2, int(round(WORLD_PERIOD * freq)))
            # Exact in binary: WORLD_PERIOD is a power of two, so q_freq is an integer scaled by a
            # negative power of two and `integer_coord * q_freq` is computed without rounding.
            q_freq = n_cells / WORLD_PERIOD
            layer = cls._perlin2d(x * q_freq, z * q_freq, seed + o * 1013, n_cells)
            if ridged:
                layer = 1.0 - np.abs(layer)
            total += amp * layer
            norm += amp
            amp *= persistence
            freq *= lacunarity
        return total / norm

    # --- 3D noise kernels (Phase 4 — docs/phase4-plan.md §2.1/§2.2) ---------------------------
    # These are the volumetric siblings of `_hash_angle`/`_perlin2d`/`_fbm` above and exist to carve
    # the occupancy band (overhangs, arches, cave mouths, hoodoos) that a heightfield cannot
    # represent. `render_region` calls them via `_render_occupancy_band` (its second stage, after
    # the 2D heightfield/profile/biome maps) and returns the resulting band as `"band"` in its
    # output dict. The 2D path itself is untouched — these kernels only feed the new stage.
    #
    # THE THREE RULES A CALLER HAS TO GET RIGHT, in one place, before reading any further:
    #   1. x and z are exactly WORLD_PERIOD-periodic and must stay that way. Reduce them with
    #      `np.mod(_, WORLD_PERIOD)` before calling, exactly as `render_region` already does for the
    #      2D path — mathematically a no-op, numerically the thing that keeps aliases bit-identical.
    #   2. y is NOT periodic and is NOT wrapped. It does not need to be: it is bounded and handed to
    #      the network directly.
    #   3. **y MUST BE ABSOLUTE WORLD Y.** Never a band index, never `y - terrain_height`. A carve
    #      field sampled relative to the surface moves with the surface, so the same block
    #      coordinate yields different occupancy depending on the terrain around it — which breaks
    #      the "every output cell is a pure function of (global block coordinate, seed)" guarantee
    #      that docs/poc-plan.md §1b makes and `export_onnx.verify_coordinate_purity` asserts, and
    #      puts the chunk-boundary seam back. Anchor the *band* to the height; index the *noise* by
    #      world y.

    @staticmethod
    def _hash_grad3(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, seed: int) -> np.ndarray:
        """Hashes an integer lattice corner to one of the 12 `_GRAD3` indices.

        Structurally the same integer avalanche as `_hash_angle`, with three differences:

        1. A third axis multiplier. All three (374761393, 2971215073, 668265263) are large and odd;
           an even multiplier would throw away low bits of that axis and make the field partially
           degenerate along it.
        2. **Every input to a multiply is masked so no product can reach 2**63.** `_hash_angle` does
           not do this and does not need to — its coordinates come from a 2D lattice that is wrapped
           on both axes — but here `iy` is deliberately *not* wrapped, and both mix rounds below
           would otherwise overflow: `(h ^ (h >> 16)) * 2654435761` with a full 32-bit `h` reaches
           1.1e19 against int64's 9.2e18 ceiling. numpy wraps int64 *array* overflow silently, so
           this cost nothing visible on the array path and the field it produced was perfectly
           deterministic — but on *scalar* inputs numpy raises `RuntimeWarning: overflow encountered
           in scalar multiply`, which is an error under `-W error::RuntimeWarning` or a pytest
           `filterwarnings = error` ini. Masking coordinates to 24 bits and the mix operands to 31
           bits keeps every intermediate well inside int64 with no behavioural asterisk.

           The 24-bit coordinate mask is not a restriction in practice: x and z arrive already
           wrapped to `lattice_period` (a few hundred cells at most) and y spans a 384-block world.
           It does mean the field technically repeats every 2**24 lattice cells in y; that is ~16.8
           million cells above the world ceiling and is not reachable.

           The `seed & 0xFFFFFF` fold is the same idea and matches `_hash_angle` exactly:
           `sample_params` draws world seeds up to 2**62 from its decorrelated seed stream, and
           multiplying one of those by 2246822519 overflows. 24 bits of seed entropy is ample — the
           seed's real job in this project is `seed_to_offset`'s *translation* of the canonical
           field, not reseeding the gradient table (docs/phase2-plan.md §1a).
        3. One extra mix round, and the reduction to 12 is the multiply-shift `(h * 12) >> 32`.
           This is a fixed-point "scale into [0, 12)" and it is chosen because it is the clean way
           to reduce a uniform 32-bit word to a non-power-of-two range — `h * 12 < 2**36` stays
           exactly representable, and the residual non-uniformity is 4/2**32 ~ 2.8e-9.

           **`h % 12` would have been fine too**, and the earlier version of this comment claimed
           otherwise. It asserted that the low bits of an xorshift-multiply are weak and that
           12 sharing the factor 4 with 2**32 biases a modulo toward the first four gradients.
           Measured on 10.17M real lattice corners, the low bits mix at least as well as the high
           bits and `h % 12` is statistically indistinguishable from the shift; the true modulo bias
           is the same ~2.8e-9. Keeping the shift because it is already here and marginally more
           principled, not because the alternative was broken. (Measured index distribution for the
           shift over 3M corners: 0.08314-0.08363 against an ideal 0.08333.)

        Note `iy` is hashed unwrapped — see `_perlin3d` for why y is not periodic.
        """
        seed_mixed = int(seed) & 0xFFFFFF
        h = (
            (ix.astype(np.int64) & 0xFFFFFF) * 374761393
            + (iy.astype(np.int64) & 0xFFFFFF) * 2971215073
            + (iz.astype(np.int64) & 0xFFFFFF) * 668265263
            + seed_mixed * 2246822519
        )  # max ~1.1e17, well inside int64
        h = h & 0xFFFFFFFF
        h = ((h ^ (h >> 13)) & 0x7FFFFFFF) * 1274126177  # max ~2.7e18
        h = h & 0xFFFFFFFF
        h = ((h ^ (h >> 16)) & 0x7FFFFFFF) * 2654435761  # max ~5.7e18
        h = h & 0xFFFFFFFF
        return (h * 12) >> 32

    @classmethod
    def _perlin3d(
        cls, x: np.ndarray, y: np.ndarray, z: np.ndarray, seed: int, lattice_period: int
    ) -> np.ndarray:
        """Classic 3D gradient (Perlin) noise, fully vectorized, evaluated at arbitrary
        (possibly non-integer, possibly domain-warped) coordinate arrays.

        **Tileable in x and z, with period `lattice_period` cells.** The eight integer lattice
        corners have their x and z indices reduced modulo `lattice_period` *before* they are hashed,
        so `noise(x + lattice_period, y, z) == noise(x, y, z)` holds identically — and, because the
        `_GRAD3` components are in {-1, 0, +1} and therefore multiply exactly, bit-for-bit rather
        than to within a rounding error. This is not a nicety. `model/positional.py`'s
        `CoordinateEncoder` builds its feature vector purely from sin/cos at wavelengths that all
        divide `WORLD_PERIOD`, so two positions `WORLD_PERIOD` apart are *literally the same input
        tensor*. A carve field that differed between them would be asking the network to emit two
        different occupancy volumes for one input; BCE resolves that the way MSE resolved it for
        height, by predicting the average — a uniform grey probability that thresholds back to
        exactly the plain heightfield this whole band exists to escape. See
        `spec_constants.WORLD_PERIOD` and docs/phase4-plan.md §2.2(a).

        `np.mod`, never C-style truncating `%`: lattice indices go negative for any sample left of
        the origin (region coordinates are signed, and a domain warp pushes samples further), and a
        truncating remainder maps -1 and +1 to gradients that are not `lattice_period` apart, which
        tears the field along the x<0 / z<0 planes. `np.mod` on int64 is floored, so
        `np.mod(-1, 8) == 7`, which is what tiling means.

        **y IS NOT PERIODIC AND MUST NOT BE WRAPPED.** The reason x and z need wrapping is that the
        network cannot distinguish aliased horizontal positions; y has no such problem. It is
        bounded (the world is 384 blocks tall) and it is handed to the consumer directly as a band
        index, so every y is a distinguishable input and there are no conflicting targets to
        average away. Wrapping it would additionally impose a spurious vertical repeat — stacked
        clones of the same cave at a fixed y interval, which reads as an obvious rendering artifact.

        **`y` MUST BE ABSOLUTE WORLD Y AT THE CALL SITE.** Never pass a band-relative index and
        never pass `y - terrain_height`. A carve field sampled relative to the surface *moves with
        the surface*: the same block coordinate then yields different occupancy depending on the
        terrain around it, which destroys the property that every output cell is a pure function of
        (global block coordinate, seed). That property is what docs/poc-plan.md §1b promises and
        what `export/export_onnx.py:verify_coordinate_purity` asserts against the real exported
        graph; breaking it makes neighbouring chunks disagree at their shared boundary and puts the
        seam back. Anchor the *band* to the height if you like — index the *noise* by world y.

        The offset vectors deliberately keep the unwrapped coordinates (`x - x0` etc.): only the
        gradient lookup wraps, the interpolation geometry does not.
        """
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        z0 = np.floor(z).astype(np.int64)
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

        # Only x and z wrap. y0/y1 are used raw.
        x0w = np.mod(x0, lattice_period)
        x1w = np.mod(x1, lattice_period)
        z0w = np.mod(z0, lattice_period)
        z1w = np.mod(z1, lattice_period)

        dx0, dx1 = x - x0, x - x1
        dy0, dy1 = y - y0, y - y1
        dz0, dz1 = z - z0, z - z1

        def dot(
            ix: np.ndarray, iy: np.ndarray, iz: np.ndarray,
            dx: np.ndarray, dy: np.ndarray, dz: np.ndarray,
        ) -> np.ndarray:
            g = cls._hash_grad3(ix, iy, iz, seed)
            return (
                np.take(_GRAD3_X, g) * dx
                + np.take(_GRAD3_Y, g) * dy
                + np.take(_GRAD3_Z, g) * dz
            )

        def fade(t: np.ndarray) -> np.ndarray:
            return t * t * t * (t * (t * 6 - 15) + 10)

        u, v, w = fade(dx0), fade(dy0), fade(dz0)

        # Trilinear interpolation, collapsed along x *as the corners are produced* rather than by
        # computing all eight first. This is a memory decision, not a style one. Callers evaluate
        # this over a [520, 520, 64] volume (17.3M cells); at float64 each full-volume temporary is
        # 138 MB, and holding eight corner arrays plus their gradient components at once peaks at
        # ~1.5 GB. Pairing them off immediately means at most two corner arrays are live, and the
        # measured peak for a [8, 520, 520] y-slab through the 4-octave `_fbm3d` is 190 MB — see
        # `_fbm3d`'s docstring for the full slab table.
        n0 = dot(x0w, y0, z0w, dx0, dy0, dz0)
        n1 = dot(x1w, y0, z0w, dx1, dy0, dz0)
        c00 = n0 + u * (n1 - n0)
        n0 = dot(x0w, y1, z0w, dx0, dy1, dz0)
        n1 = dot(x1w, y1, z0w, dx1, dy1, dz0)
        c10 = n0 + u * (n1 - n0)
        n0 = dot(x0w, y0, z1w, dx0, dy0, dz1)
        n1 = dot(x1w, y0, z1w, dx1, dy0, dz1)
        c01 = n0 + u * (n1 - n0)
        n0 = dot(x0w, y1, z1w, dx0, dy1, dz1)
        n1 = dot(x1w, y1, z1w, dx1, dy1, dz1)
        c11 = n0 + u * (n1 - n0)
        del n0, n1

        c0 = c00 + v * (c10 - c00)
        c1 = c01 + v * (c11 - c01)
        return (c0 + w * (c1 - c0)) * PERLIN3D_NORM  # see PERLIN3D_NORM: the factor is 1.0

    @classmethod
    def _fbm3d(
        cls, x: np.ndarray, y: np.ndarray, z: np.ndarray, seed: int, base_freq: float,
        octaves: int = 4, persistence: float = 0.5, lacunarity: float = 2.0, ridged: bool = False,
    ) -> np.ndarray:
        """Fractal-Brownian-motion sum of tileable 3D Perlin octaves — the volumetric sibling of
        `_fbm`, with the identical signature shape and the identical periodicity contract.

        Every octave shares the *same* horizontal world period of exactly `WORLD_PERIOD` blocks, so
        their sum does too. An octave's world period is `lattice_period / freq` blocks, which is
        only exactly `WORLD_PERIOD` when `WORLD_PERIOD * freq` is an integer, and a requested
        `freq` generally is not. So each octave's frequency is **quantized** to
        `n_cells / WORLD_PERIOD` for the nearest integer `n_cells`, and **the quantized value is
        what gets sampled**. Rounding `n_cells` for the lattice but sampling at the *requested*
        frequency is the tempting shortcut and it is wrong: the period then misses by a fraction of
        a cell, which looks negligible on one alias and accumulates into visible drift over the 16-32
        aliases a training run actually visits — silently reintroducing the exact conflicting-target
        problem the tiling exists to remove (identical `CoordinateEncoder` input, different carve).

        **The cost is a real shift in feature wavelength and it is NOT sub-percent at carve
        frequencies.** Effective frequencies land on a `1/WORLD_PERIOD` grid, so the relative shift
        is at most `0.5 / (WORLD_PERIOD * freq)` — i.e. it shrinks as the requested frequency rises,
        and the low frequencies are the ones that hurt. Measured, first octave:

            base_freq          cells/period   n_cells   shift
            4.37 / 512             17.48         17      2.75%
            3.10 / 512             12.40         12      3.23%
            1.90 / 512              7.60          8      5.26%
            0.70 / 512              2.80          3      7.14%

        (The identical "sub-percent" claim in the 2D `_fbm` above is inherited optimism from a
        higher-frequency regime; it is wrong there too.) This matters for A3's per-archetype carve
        frequencies specifically because the *large* carve features — a walk-in cave mouth, an arch
        spanning a canyon — are exactly the low-frequency ones sitting in the 3-7% band. Periodicity
        is still exact; what you asked for is not quite what you get. If a pinned carve frequency
        must be hit precisely, choose it ON the grid (`n / WORLD_PERIOD` for integer `n`) rather
        than expecting the quantizer to be invisible.

        `n_cells` is floored at 2 because a 1-cell period is a constant field — all eight corners
        wrap to the same gradient. Note this floor is itself a large shift for any requested
        frequency below `1.5 / WORLD_PERIOD`.

        y is scaled by the same quantized frequency (the carve is isotropic) but is otherwise left
        alone: it is not wrapped and has no period. **Pass absolute world y** — see `_perlin3d` for
        why a terrain-relative y silently breaks coordinate purity.

        **Broadcasting.** `x`, `y` and `z` may have any mutually broadcastable shapes; the output has
        their broadcast shape. This is the intended way to evaluate a volume: pass
        `x` as `(1, CANVAS, 1)`, `z` as `(1, 1, CANVAS)` and `y` as `(slab, 1, 1)` and the
        coordinate arrays and their fade curves stay tiny while only the corner dot products are
        full size. Note this is why the accumulator is `np.zeros(np.broadcast_shapes(...))` and not
        `np.zeros_like(x)` as in the 2D `_fbm`, whose inputs are always a matched meshgrid.

        **Do not evaluate [520, 520, 64] in one call.** Measured on this machine, full canvas
        volume, 4 octaves, by y-slab thickness (wall clock for the whole volume, and peak Python
        allocation):

            slab  4   9.6 s    95 MB      slab 16   9.9 s   381 MB
            slab  8   9.8 s   190 MB      slab 32  10.1 s   762 MB
                                          slab 64  12.1 s  1523 MB

        Throughput is flat-to-slightly-*better* at small slabs (they stay in cache; the big ones
        just thrash), so there is no speed argument for a big one — the 1.5 GB single-call figure
        is strictly worse on both axes. **Slab y in blocks of 8**, or 4 if several carve fields are
        being held simultaneously. ~10 s per region per field also means the 3D pass dominates
        region generation; budget ~1.7 h for 600 regions and ~22 h for 8000, per carve field
        (docs/phase4-plan.md §2.3).
        """
        shape = np.broadcast_shapes(np.shape(x), np.shape(y), np.shape(z))
        total = np.zeros(shape, dtype=np.float64)
        amp, freq, norm = 1.0, base_freq, 0.0
        for o in range(octaves):
            n_cells = max(2, int(round(WORLD_PERIOD * freq)))
            # Exact in binary: WORLD_PERIOD is a power of two, so q_freq is an integer scaled by a
            # negative power of two and `coord * q_freq` introduces no rounding of its own. That is
            # what lets the x/z periodicity below be an `==` rather than an `allclose`.
            q_freq = n_cells / WORLD_PERIOD
            layer = cls._perlin3d(x * q_freq, y * q_freq, z * q_freq, seed + o * 1013, n_cells)
            if ridged:
                layer = 1.0 - np.abs(layer)
            total += amp * layer
            norm += amp
            amp *= persistence
            freq *= lacunarity
        return total / norm

    # --- the occupancy band rasterizer (Phase 4 — docs/phase4-plan.md §2) ---------------------
    def _render_occupancy_band(
        self,
        params: RegionParams,
        X: np.ndarray,
        Z: np.ndarray,
        heightfield: np.ndarray,
        slope_norm: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Second stage of `render_region`: turns the 2D heightfield into a
        `[CANVAS, CANVAS, BAND_HEIGHT]` occupancy raster, carved by the shared 3D field.

        This is the direct implementation of the carve-parameter contract block at the top of this
        file — read that first; it is authoritative and this docstring does not restate its
        derivations. What follows is only the mapping from that contract onto this code.

        Args:
            params: the region's parameter vector. Only the carve half is read here.
            X, Z: the FOLDED global coordinates `render_region` already built, `[z][x]`, i.e.
                `np.mod(global + seed_offset, WORLD_PERIOD)`. Reused rather than rebuilt so the 3D
                field is sampled at bit-identical coordinates to the 2D one — coordinate rule 1.
                The domain-WARPED `Xw`/`Zw` are deliberately NOT used: the warp is a property of the
                landform, and warping the carve too would drag cavities along the valley meanders
                and re-correlate two fields that are more useful independent.
            heightfield: PRE-hoodoo `h`, float32, already clipped. Both gates read this, never the
                shaved version — see the contract's measurement of what post-hoodoo gates do.
            slope_norm: `np.clip(|grad h| / SLOPE_NORM_SCALE, 0, 1)` from the same pre-hoodoo `h`.
                Passed in rather than recomputed because `render_region`'s profile map needs the
                identical array, and computing `np.gradient` twice is both wasted work and a place
                for the two copies to drift.

        Returns:
            `(band, h_carved)`. `band` is `[CANVAS, CANVAS, BAND_HEIGHT]` bool, `[z][x][i]`, True =
            solid, `i` ascending from the band bottom — the layout `viz.render_cross_section` and
            `scripts/diagnose_overhangs.py` both call the internal one. `h_carved` is the
            POST-hoodoo heightfield and is what `render_region` returns as `heightfield`, because
            after the hoodoo shave the pre-hoodoo `h` is no longer the surface.

        ORDER OF OPERATIONS, which the contract pins and which is not interchangeable:
        gates from pre-hoodoo `h` -> hoodoo heightfield edit -> anchor from the carved `h` -> 3D
        features carve the band.

        THE ANCHOR CANCELS OUT OF THE FILL RULE, and that is worth seeing written down because it
        is what makes this cheap and what makes the §1.4 invariant checkable at a glance:

            world_y = anchor + BAND_BOTTOM_OFFSET + i
            depth   = anchor - world_y = -(BAND_BOTTOM_OFFSET + i) = 61 - i
            solid   = depth >= 0  <=>  i <= 61

        So the *base* band is the same 64 cells in every column of every world (exactly what
        `diagnose_overhangs.heightfield_band` observes about the v1.1.1 world), and every column's
        entire individuality comes from the 3D carve and from where its anchor sits. Two consequences
        used below: the depth window of each feature is a fixed set of BAND INDICES, so the slabs
        outside the union of the windows need no field evaluation at all; and the depth comparison is
        a length-`slab` vector rather than a `[CANVAS, CANVAS, slab]` array.

        THE §1.4 INVARIANT IS STRUCTURAL HERE, not a hope. Every depth window starts at depth >= 3,
        so no 3D feature can touch depth 0/1/2; index 61 (depth 0) is therefore solid in every
        column and indices 62/63 (depth -1/-2) are air in every column. Topmost solid cell
        == anchor == `np.rint(h_carved)`, always. The `world_y < WORLD_FLOOR_Y` re-solidification
        cannot break it either, since it only ever fires below the anchor. Asserted at the end of
        this method on every region, and again over many regions in
        `tests/test_model.py::test_band_top_solid_cell_is_the_returned_heightfield`.

        MEASURED, GATE 0 (task A4; 4 regions per archetype at generator seed 7, metrics from
        `scripts/diagnose_overhangs.py` over the full 520x520 canvas, connected components over a
        256x256x64 window with 6-connectivity):

            archetype                 overhang%  multirun%  cavity p50/p90/p95/p99/max  carved%  cc
            valley_mountains_craters    0.680      5.99        5 / 14 / 17 / 24 / 35     0.799   102
            deep_canyon                 0.792      5.65        6 / 18 / 21 / 29 / 36     0.769   123
            fjord_rift                  1.015      7.28        7 / 17 / 21 / 28 / 36     0.875   111
            snow_peaks                  0.982      7.82        6 / 16 / 19 / 25 / 36     0.960   150
            tropical_archipelago        0.724      5.53        6 / 16 / 20 / 26 / 35     0.807   128
            taiga_forest                0.446      3.36        6 / 16 / 20 / 28 / 35     0.599   113
            mesa_plateaus               0.140      1.29        5 / 14 / 17 / 23 / 31     0.141   227
            desert_dunes                0          0           -                         0        0
            rolling_grassland           0          0           -                         0        0
            swamp                       0          0           -                         0        0
            savanna_plains              0          0           -                         0        0

        Three things to read off that table, because they are the things it was measured to decide:

          * **The four flat archetypes are EXACTLY zero**, not nearly zero — every band cell equals
            the plain heightfield expansion, pinned by
            `tests/test_model.py::test_flat_archetypes_have_no_3d_structure`.
          * **The 64-block band is the right size and is not the binding constraint.** Max cavity
            height is 31-36 against the depth windows' arithmetic cap of 36, p99 is 23-29, p90 is
            14-18. docs/phase4-plan.md §1.2 says to grow the band if voids routinely exceed ~48;
            nothing here comes close, and no cavity is clipped by the band floor because the deepest
            window (38) stops 23 cells above it.
          * **No percolation.** Largest connected component of the carved set is 15-35% of carved
            cells and 0.02-0.26% of band VOLUME, over 102-227 components per 256x256 window — the
            regime the contract's level-set comparison calls the good one, and three orders of
            magnitude off the rejected sheet operator's 21%-of-volume single component.

        TWO MEASURED IMPERFECTIONS, recorded rather than fixed, both small:

          * **A weak lattice-phase modulation of cavity density.** Pooled over 5 `deep_canyon`
            regions, carved density against the first octave's 51.2-block lattice phase runs
            0.78 / 1.03 / 1.08 / 1.22 / 1.21 / 1.06 / 1.05 / 0.94 / 0.87 / 0.75 (1.0 = flat), i.e.
            cavities are ~1.6x more likely mid-cell than on a lattice plane. That is intrinsic to
            Perlin noise — every corner's dot product vanishes at its own corner, so |F| is
            systematically small near lattice points and the *tail* sublevel set avoids them. Per
            region it is invisible under sampling noise (the five regions individually peak at
            different phases), and it is not visible by eye in a rendered cross-section. Worth
            knowing about only if a future carve threshold moves much closer to the field's median,
            where the effect grows.
          * **The arch gate is very slightly window-dependent near the canvas edge.**
            `gaussian_filter` uses reflected boundaries, so `prominence` within ~3*sigma = 36 columns
            of the canvas edge is computed against a mirrored neighbourhood rather than the real
            one. Measured by re-cropping a region's canvas 40 blocks in on all sides and recomputing
            the gate: the arch gate flips for 3.96% / 2.23% / 0.62% / 0.03% / 0.00% of columns at
            0-4 / 4-12 / 12-24 / 24-36 / 36+ blocks from the new edge (`deep_canyon`; `mesa_plateaus`
            1.19% and `fjord_rift` 0.13% in the outermost ring). So the outer ~30 of 520 columns
            carry an arch gate that depends on which region window rendered them, which is a small
            hole in "every output cell is a pure function of (global block coordinate, seed)". It
            does NOT affect the WORLD_PERIOD periodicity guarantee — a whole-period shift moves the
            canvas by whole regions, so the two canvases are identical arrays — and shards never
            overlap, since every region draws its own world seed. Closing it properly means
            evaluating the 2D pipeline on a canvas padded by 3*sigma and cropping, which costs a
            ~1.3x wider 2D pass on every region; that was judged not worth it for a term that only
            gates one of three features. Revisit if a future gate uses a much larger sigma.
        """
        seed = CANONICAL_NOISE_SEED
        h = heightfield

        # --- hoodoo: a heightfield edit, and it runs FIRST ------------------------------------
        # `min(_, 1.0)` is the contract's clamp: past a threshold of ~0.85 the floor swallows the
        # spires and the feature inverts into "shave everything", and the gain is a product of two
        # sampled ranges so it can exceed 1 without either factor being unreasonable.
        hoodoo_gain = min(1.0, float(params.carve_strength) * float(params.hoodoo_strength))
        if hoodoo_gain > CARVE_STRENGTH_EPS:
            threshold = HOODOO_THRESHOLD_OFF + HOODOO_THRESHOLD_SPAN * hoodoo_gain
            plan = self._fbm(
                X, Z, seed + HOODOO_SEED_OFFSET, params.hoodoo_frequency / REGION_BLOCKS,
                octaves=HOODOO_OCTAVES, ridged=True,
            )
            ramp = np.clip((threshold - plan) / HOODOO_SOFT, 0.0, 1.0)
            # The re-clip is required by the contract even though it is not reachable in practice
            # (measured min h over 120 regions: 15.57, against TERRAIN_CLIP_MIN -48). It is here so
            # the anchor obeys the pipeline's invariant rather than depending on that margin.
            h_carved = np.clip(
                h - HOODOO_DEPTH * ramp, TERRAIN_CLIP_MIN, TERRAIN_CLIP_MAX
            ).astype(np.float32)
        else:
            # Skipped rather than computed-and-discarded. This is provably a no-op and not an
            # approximation: HOODOO_THRESHOLD_OFF (-0.05) is below the infimum of the 2D ridged
            # kernel's output (`_perlin2d` is bounded by construction, so `1 - |layer| >= 9.6e-6`),
            # so at gain 0 the ramp is `clip(negative, 0, 1) == 0` everywhere and
            # `h - HOODOO_DEPTH * 0.0` is bit-identically `h`. Skipping saves the four flat
            # archetypes — 35% of regions by weight — a 2D fBm they would throw away.
            h_carved = h

        anchor = np.rint(h_carved).astype(np.int64)  # half-to-even, matching viz.py. Rule 3.

        # --- the base band: the fill rule, with the anchor already cancelled ------------------
        band_i = np.arange(BAND_HEIGHT, dtype=np.int64)
        depth_of_i = -(BAND_BOTTOM_OFFSET + band_i)          # 61 at index 0 down to -2 at index 63
        band = np.broadcast_to(depth_of_i >= 0, h.shape + (BAND_HEIGHT,)).copy()

        # --- which 3D features are live in this region ----------------------------------------
        # The gain is a product, so `carve_strength == 0` switches all three off at once, and that
        # is the contract's proof rather than a calibration: the operator is `gain * F < -0.35`, and
        # at gain 0 that is `0 < -0.35`, false for every cell of every field.
        master = float(params.carve_strength)
        features: List[Tuple[str, float, int, int]] = []
        if master > CARVE_STRENGTH_EPS:
            for name, strength, roof, reach in (
                ("undercut", params.undercut_strength, UNDERCUT_ROOF, UNDERCUT_REACH),
                ("arch", params.arch_strength, ARCH_SPAN_CLEARANCE, ARCH_REACH),
                ("cave_mouth", params.cave_mouth_strength, CAVE_ROOF, CAVE_REACH),
            ):
                gain = master * float(strength)
                if gain > CARVE_STRENGTH_EPS:
                    features.append((name, gain, roof, roof + reach))  # windows are INCLUSIVE
        if not features:
            self._assert_band_top_is_anchor(band)
            return band, h_carved

        # --- terrain gates, all from PRE-hoodoo `h` -------------------------------------------
        gates: Dict[str, np.ndarray] = {}
        for name, _gain, _d0, _d1 in features:
            if name == "undercut":
                gates[name] = slope_norm >= UNDERCUT_SLOPE_MIN
            elif name == "cave_mouth":
                gates[name] = slope_norm >= CAVE_SLOPE_MIN
            else:
                # Prominence: how far a column stands above its own neighbourhood. Positive on a fin
                # or ridge crest, negative in a gully — what makes an arch an arch rather than a
                # hole in a hillside. Computed only when the arch is live: a sigma-12 gaussian over
                # a 520x520 canvas is ~25 ms measured, small against the 3D pass but pure waste in
                # the 6 archetypes that have no arches.
                prominence = h - gaussian_filter(h, sigma=ARCH_PROMINENCE_SIGMA)
                gates[name] = (slope_norm >= ARCH_SLOPE_MIN) & (prominence >= ARCH_PROMINENCE_MIN)

        # --- the band indices any live feature can reach --------------------------------------
        # Derived from the windows rather than hardcoded, so a future edit to a depth constant
        # cannot leave a stale range behind that silently stops carving the deepest cells. With the
        # shipped constants the union is depth 3..38, i.e. indices 23..58: 36 of the 64 indices, so
        # the 3D field is evaluated over 56% of the band's volume and the other 44% is known to be
        # untouched solid (or the two cells of headroom) before any noise is sampled.
        # `depth = -(BAND_BOTTOM_OFFSET + i)` inverts to `i = -BAND_BOTTOM_OFFSET - depth`, so a
        # deeper window is a LOWER band index and the two bounds swap.
        i_lo = max(0, min(-BAND_BOTTOM_OFFSET - d1 for _n, _g, _d0, d1 in features))
        i_hi = min(BAND_HEIGHT - 1, max(-BAND_BOTTOM_OFFSET - d0 for _n, _g, d0, _d1 in features))

        carve_freq = params.carve_frequency / REGION_BLOCKS
        # `[CANVAS, CANVAS, 1]`: x and z are constant along the band axis and broadcast against a
        # full-size y. See the contract's BROADCASTING NOTE — the band is anchored per column, so
        # `_fbm3d`'s "keep the coordinate arrays tiny" trick does not apply and world y is genuinely
        # `[CANVAS, CANVAS, slab]`.
        x3 = X[:, :, None]
        z3 = Z[:, :, None]

        # Slabbed in y per CARVE_Y_SLAB, which the contract chooses on MEMORY and not on time.
        for i0 in range(i_lo, i_hi + 1, CARVE_Y_SLAB):
            i1 = min(i0 + CARVE_Y_SLAB, i_hi + 1)
            idx = band_i[i0:i1]
            depth = depth_of_i[i0:i1]                                   # [slab], per band index
            # ABSOLUTE world y — coordinate rule 2. A band-relative y here would move the field with
            # the terrain and destroy the property that every output cell is a pure function of
            # (global block coordinate, seed).
            world_y = anchor[:, :, None] + (BAND_BOTTOM_OFFSET + idx)   # [CANVAS, CANVAS, slab]

            field = self._fbm3d(
                x3, world_y.astype(np.float64), z3,
                seed + CARVE_SEED_OFFSET, carve_freq,
                octaves=CARVE_OCTAVES, persistence=0.5, lacunarity=2.0,
            )

            air = np.zeros(world_y.shape, dtype=bool)
            for name, gain, d0, d1 in features:
                in_window = (depth >= d0) & (depth <= d1)               # [slab]
                if not in_window.any():
                    continue
                # ONE operator, three gates. `gain * field < CARVE_THRESHOLD` and not
                # `field < CARVE_THRESHOLD / gain`: the two are not bit-identical in floating point,
                # and every tail fraction in the contract is calibrated against the multiply form.
                air |= (
                    gates[name][:, :, None]
                    & in_window[None, None, :]
                    & (gain * field < CARVE_THRESHOLD)
                )
            del field

            # UNION, and nothing ever re-solidifies a carved cell — except the world floor, which is
            # not a carve rule but a representation rule (see WORLD_FLOOR_Y). It can only fire below
            # the anchor, so it cannot disturb the §1.4 invariant.
            band[:, :, i0:i1] = (band[:, :, i0:i1] & ~air) | (world_y < WORLD_FLOOR_Y)

        self._assert_band_top_is_anchor(band)
        return band, h_carved

    @staticmethod
    def _assert_band_top_is_anchor(band: np.ndarray) -> None:
        """docs/phase4-plan.md §1.4, checked on every rendered region.

        The returned `heightfield` must be *the topmost solid cell of the band*. That holds by
        construction (every 3D depth window starts strictly below depth 0), so this assertion is
        cheap insurance against a future edit to a depth constant rather than a real risk today —
        which is exactly why it is worth having: a window that reached depth 0 would produce
        perfectly plausible terrain whose heightmap is one block wrong in some columns, and nothing
        else in the pipeline would notice. Costs three slices over 1.4M cells, ~1 ms measured.
        """
        surface_i = BAND_HEIGHT - 1 - BAND_TOP_OFFSET  # 61: the cell at depth 0
        assert band[:, :, surface_i].all(), (
            "a 3D feature removed the anchor cell (depth 0) from some column, so the returned "
            "heightfield is no longer the topmost solid cell of the band and docs/phase4-plan.md "
            "§1.4 is false. Check that every depth window still starts strictly below depth 0"
        )
        assert not band[:, :, surface_i + 1:].any(), (
            "the band has solid cells above the anchor, which the fill rule `depth >= 0` cannot "
            "produce — the BAND_TOP_OFFSET headroom is being filled by something"
        )

    def render_region(self, params: RegionParams) -> Dict[str, Any]:
        """Renders the (CANVAS x CANVAS) heightfield/profile/biome rasters for one macro-region,
        plus the `[CANVAS, CANVAS, BAND_HEIGHT]` occupancy band Phase 4 added.
        Returns arrays covering the full region plus its HALO_MARGIN border on every side, so
        `data/dataset.py` can slice any of the region's 32x32 chunks (including edge chunks) with
        a full 24x24 halo window with no further padding.

        Two stages. The first is the Phase 2/3 heightfield and is untouched. The second is
        `_render_occupancy_band`, which shaves the heightfield where hoodoos stand and then carves
        overhangs, arches and cave mouths out of a 64-block band anchored to it — the thing that
        makes this dataset able to teach geometry no heightmap can express.

        **`heightfield` changes meaning here** (docs/phase4-plan.md §1.4): it is now *the topmost
        solid cell of the returned band*, not an independent quantity the band was derived from. The
        two cannot drift, because the band's construction guarantees it; see
        `_render_occupancy_band`'s assertion.
        """
        # GLOBAL block coordinates (Phase 2 — docs/phase2-plan.md §1a). Day 1 sampled noise at
        # *region-local* coordinates with a *per-region* Perlin seed, which made every macro-region
        # an independent pseudorandom field: unlearnable by a coordinate-conditioned network, and
        # discontinuous at every region boundary. Sampling one canonical field at global coordinates
        # fixes both — terrain now continues smoothly from region to region, so a valley can span
        # hundreds of blocks, and the target becomes a function the network can actually represent.
        local = np.arange(CANVAS, dtype=np.float64) - HALO_MARGIN
        gx = params.region_x * REGION_BLOCKS + local
        gz = params.region_z * REGION_BLOCKS + local
        Z, X = np.meshgrid(gz, gx, indexing="ij")  # [z][x], matches heightmap axis order

        # The world seed translates the canonical field rather than reseeding it.
        off_x, off_z = seed_to_offset(params.seed)
        X = X + off_x
        Z = Z + off_z

        # Fold into the canonical period. The noise below is exactly WORLD_PERIOD-periodic by
        # construction (see `_perlin2d`/`_fbm`), so this is a mathematical no-op — but it is not a
        # numerical one, and that is the point. Global coordinates reach ~35000 (64 regions x 512
        # blocks, plus the seed offset); evaluating the *same* landscape at x and x+2048 through
        # float64 addition, domain warping and floor() would agree only to ~1e-11, which is not the
        # bit-exact identity the training target needs. Reducing the coordinate first makes aliased
        # positions literally the same input, so the periodicity is exact rather than merely close.
        # It also keeps every downstream product small and well away from float64's exact-integer
        # limit. `np.mod`, not `%`-on-negatives semantics: region_x/region_z may be negative.
        X = np.mod(X, WORLD_PERIOD)
        Z = np.mod(Z, WORLD_PERIOD)

        # Canonical, fixed layer ids — deliberately NOT a function of seed/region_x/region_z.
        seed = CANONICAL_NOISE_SEED

        # Domain warp: erosion drives how much the sampling coordinates meander, producing
        # winding valleys rather than perfectly radial noise contours.
        #
        # The warp is itself an fbm, so it is WORLD_PERIOD-periodic on exactly the same terms as the
        # field it warps. DO NOT "optimize" this into a non-periodic offset (a linear ramp, a raw
        # hash, an unwrapped gradient): the composition `f(x + w(x))` is P-periodic *only* because
        # both `f` and `w` are — `f(x+P + w(x+P)) = f(x+P + w(x)) = f(x + w(x))`. A non-periodic
        # warp of a periodic field is not periodic, and would put the whole tiling guarantee back
        # where it was.
        warp_freq = 1.0 / 220.0
        wx = self._fbm(X, Z, seed + 101, warp_freq, octaves=2)
        wz = self._fbm(X, Z, seed + 202, warp_freq, octaves=2)
        warp_amp = 40.0 * params.erosion
        Xw = X + wx * warp_amp
        Zw = Z + wz * warp_amp

        base_freq = params.relief_frequency / REGION_BLOCKS
        raw = self._fbm(Xw, Zw, seed, base_freq, octaves=5, persistence=0.5, lacunarity=2.0)
        ridged = self._fbm(Xw, Zw, seed + 303, base_freq, octaves=5, persistence=0.5, lacunarity=2.0, ridged=True)
        ridged_signed = ridged * 2.0 - 1.0
        shaped = (1.0 - params.ridge_sharpness) * raw + params.ridge_sharpness * ridged_signed
        shaped = np.clip(shaped, -1.3, 1.3)

        # Plateau quantization: stair-step the shape toward discrete terraces (mesa plateaus).
        if params.plateau_quantization > 1e-3:
            levels = max(2.0, 14.0 * (1.0 - params.plateau_quantization) + 2.0)
            quantized = np.round(shaped * levels) / levels
            shaped = (1.0 - params.plateau_quantization) * shaped + params.plateau_quantization * quantized

        heightfield = params.base_height + shaped * params.relief_amplitude
        # Erosion emphasizes valley carving: push already-low areas further down. Phase 2 raises the
        # carving coefficient from 0.35 to 0.60 and squares the mask, so the deepest lows plunge
        # while shallow dips are left alone — Day-1 ground truth never went below y=27.7 anywhere in
        # 200 regions, which is why "deep valleys" was unlearnable regardless of model quality
        # (docs/phase2-plan.md §0). Squaring rather than raising the coefficient further keeps
        # canyon floors varied instead of clipping them flat against TERRAIN_CLIP_MIN.
        # `VALLEY_MASK_SCALE` normalizes by the *realistic* negative excursion of `shaped` (which
        # only reaches about -0.22 for ridge-heavy archetypes, not -1) — without it the squared mask
        # caps at 0.048 and the whole carving term is a measured no-op. See its definition above.
        valley_mask = np.clip(-shaped / VALLEY_MASK_SCALE, 0.0, 1.0) ** 2
        heightfield -= params.erosion * 0.60 * params.relief_amplitude * valley_mask

        heightfield = np.clip(heightfield, TERRAIN_CLIP_MIN, TERRAIN_CLIP_MAX).astype(np.float32)

        # --- slope, shared by the profile map and the Phase 4 carve gates ----------------------
        # HOISTED OUT OF THE `relief_amplitude > 25` BRANCH in Phase 4. It used to live inside the
        # profile-map branch below, which meant the gentle archetypes never computed it at all — and
        # `_render_occupancy_band` needs it unconditionally, because `taiga_forest` and
        # `tropical_archipelago` carve cave mouths (gated on slope) while sitting at
        # relief_amplitude 14-38, i.e. on both sides of that threshold. Computing it here also makes
        # the profile map and the carve gates read the SAME array rather than two copies that can
        # drift apart under an edit to SLOPE_NORM_SCALE.
        #
        # NB: distinct names from the `gx`/`gz` coordinate vectors above — these are the
        # heightfield's partial derivatives, not block coordinates.
        dh_dz, dh_dx = np.gradient(heightfield)
        slope = np.sqrt(dh_dx ** 2 + dh_dz ** 2)
        # Fixed normalizer, never `slope.max()` — see SLOPE_NORM_SCALE for why a per-region
        # statistic here produces a material seam at region boundaries.
        slope_norm = np.clip(slope / SLOPE_NORM_SCALE, 0.0, 1.0)

        # --- profile map: base profile everywhere, stone override on steep/high terrain --------
        profile_map = np.full(heightfield.shape, params.surface_profile_id, dtype=np.uint8)
        if params.relief_amplitude > 25.0:
            peak_thresh = params.base_height + 0.62 * params.relief_amplitude
            stone_score = 0.6 * slope_norm + 0.4 * np.clip(
                (heightfield - peak_thresh) / (params.relief_amplitude + 1e-6), 0.0, 1.0
            )
            stone_score = gaussian_filter(stone_score, sigma=1.5)
            profile_map = np.where(stone_score > 0.35, np.uint8(4), profile_map)

        # --- the occupancy band (Phase 4) ------------------------------------------------------
        # Returns the POST-hoodoo heightfield too: after the shave, the pre-hoodoo `h` is no longer
        # the surface, and `heightfield` is contractually the topmost solid cell of the band
        # (docs/phase4-plan.md §1.4). Everything above this line — the profile map included — reads
        # the PRE-hoodoo heightfield, which is the contract's rule for the carve gates and applies
        # to the profile map for the same reason: the hoodoo is a texture applied to the landform,
        # and its 18-block spire walls read as |grad h| ~ 18, which would repaint them stone and
        # roughly double the archetype's rock coverage (the contract measures the gate shares
        # 9.5%/1.8%/13.1% pre-hoodoo against 17.5%/9.8%/20.1% post- on three `mesa_plateaus`
        # regions). Only `mesa_plateaus` and `deep_canyon` have hoodoos at all, so for the other
        # nine archetypes `h_carved is heightfield` and the distinction is moot.
        band, heightfield = self._render_occupancy_band(
            params, X, Z, heightfield, slope_norm
        )

        # --- biome map: base biome everywhere, ocean under water -------------------------------
        # Read from the POST-hoodoo heightfield, unlike the profile map above: a spire-field floor
        # shaved 18 blocks down past sea level really is submerged, and labelling it land would put
        # a grass biome under water in the one place the terrain and the label are guaranteed to
        # disagree.
        biome_map = np.full(heightfield.shape, params.biome_class, dtype=np.uint8)
        underwater = heightfield < (params.water_level - 1.0)
        biome_map = np.where(underwater, np.uint8(7), biome_map)

        assert profile_map.max() < NUM_PROFILES
        assert biome_map.max() < NUM_BIOMES

        return {
            "heightfield": heightfield,
            "band": band,
            "profile_map": profile_map,
            "biome_map": biome_map,
            "water_level": np.float32(params.water_level),
            "params": asdict(params),
        }

    def generate_shards(self, num_regions: int, out_dir: str, region_layout_width: int = 64) -> List[str]:
        """Samples `num_regions` macro-regions and writes one `.npz` shard each to `out_dir`.
        Region (x, z) coordinates are laid out on a grid purely for bookkeeping/filenames. Since
        `sample_params` draws an independent *world seed* per region (and the world seed translates
        the canonical noise field, per docs/phase2-plan.md §1a), grid-adjacent shards are not
        continuations of each other — each is a differently-translated, differently-parameterized
        window onto the one shared field. Continuity holds between any two regions that share a
        world seed and parameter vector, which is the property the trained model has to reproduce
        at inference time (and which `render_region`'s global-coordinate sampling guarantees).

        Because the field is WORLD_PERIOD-periodic, every shard — wherever it sits on the layout
        grid — samples the same 2048x2048 canonical domain. 600 regions of 512x512 therefore cover
        that domain ~37x over, and 8000 regions ~500x. The alternative (a 65536-block offset window)
        spread the same shards over a 1024x larger domain at ~3% coverage, which is a memorization
        problem rather than a regression problem."""
        import os

        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for region_id in range(num_regions):
            rx = region_id % region_layout_width
            rz = region_id // region_layout_width
            params = self.sample_params(rx, rz)
            region = self.render_region(params)
            path = os.path.join(out_dir, f"region_{region_id:05d}.npz")
            np.savez_compressed(
                path,
                heightfield=region["heightfield"],
                band=region["band"],
                profile_map=region["profile_map"],
                biome_map=region["biome_map"],
                water_level=region["water_level"],
                **{f"param_{k}": v for k, v in region["params"].items()},
            )
            paths.append(path)
        return paths
