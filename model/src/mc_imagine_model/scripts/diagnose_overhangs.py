#!/usr/bin/env python3
"""
Overhang Diagnosis Script (Mc-Imagine Phase 4 — docs/phase4-plan.md §2.4).

Computes the four volumetric metrics §9's measurements log is filled in from, on a
surface-anchored occupancy band: overhang cells, multi-run columns, the cavity-height
distribution (which is what validates the 64-block `BAND_HEIGHT`, §1.2), and the walkable-void
count. Plus the marginal-probability baseline that Gate 2's go/no-go is judged against.

THE POINT OF THIS FILE IS THAT THERE IS ONE SET OF METRIC FUNCTIONS. §2.4:

    "run it on ground truth and on model output with the same code, so 'how much of the data's
     3D structure survived training' is one ratio and not two incomparable numbers"

So `compute_overhang_metrics` is called on both sides and `compute_retention` divides them here,
in the tool, rather than leaving a human to divide two printouts and get the normalization wrong.
Every headline metric is therefore a *percentage or a density*, never a raw count: ground truth is
a 520x520 region and model output is a handful of 16x16 chunks, and a retention ratio built from
raw counts would be a ratio of areas wearing a ratio of terrain quality's clothes.

Usage — Gate 0 (data only, no model exists yet):

    PYTHONPATH=model/src python model/src/mc_imagine_model/scripts/diagnose_overhangs.py \
        --ground-truth model/data/regions/region_00000.npz \
        --output overhangs_gate0.json

Usage — Gate 2/3 (paired; the retention column and the go/no-go become real):

    PYTHONPATH=model/src python model/src/mc_imagine_model/scripts/diagnose_overhangs.py \
        --ground-truth model/data/regions/ --max-regions 32 \
        --model model/training_runs/run1/band_output.npy \
        --model-heightfield model/training_runs/run1/heightmap.npy \
        --output overhangs_gate2.png

Same command, different arguments. `--model` may also be produced in-process from a checkpoint via
`--checkpoint` once the occupancy head (docs/phase4-plan.md §1.3) exists.

PASS `--model-heightfield` (or use `--checkpoint`, which gets it for free). The walkable-void count
joins floors in absolute world y, and the band is anchored per column — so without a heightfield the
model side can only be measured in band-index space, which overcounts voids by up to 24x on sloped
terrain. When the two sides disagree the tool re-measures BOTH in the weaker space rather than
reporting a retention ratio that is mostly an artefact of a missing file.

Gate 2's go/no-go is a conjunction of three conditions (§7, and `evaluate_baseline_gate`):
multi-run rate above the marginal baseline, walkable-void retention, and p90 cavity height. Its
fourth condition — relief retention not regressed from v1.1.1 — is measured by the terrain tooling,
not here.
"""

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from mc_imagine_model.spec_constants import (
    BAND_BOTTOM_OFFSET,
    BAND_HEIGHT,
    BAND_TOP_OFFSET,
)

# Keys a shard might store the band under. `data/world_generator.py` writes one of these; the list
# exists so this tool does not have to be edited in lockstep with a rename, and so a rename that
# lands while nobody is looking produces the loud heightfield-fallback banner below rather than a
# KeyError three gates later.
BAND_KEY_CANDIDATES = ("band", "occupancy_band", "occupancy", "band_raster")

# docs/phase4-plan.md §1.2: "if the 3D generator routinely produces voids deeper than ~48 blocks,
# either the band grows or the generator is over-carving". That sentence is the entire reason the
# cavity-height distribution is a headline metric, so the threshold it names is a constant here and
# is reported as a percentage rather than left for a reader to eyeball off a histogram.
BAND_SIZE_WARNING_HEIGHT = 48

# §2.4's walkable void: "cavities >=3 blocks tall with >=2 blocks of floor".
DEFAULT_MIN_WALKABLE_HEIGHT = 3
DEFAULT_MIN_WALKABLE_FLOOR = 2
# A neighbouring column's floor may sit one block off and still be the same walkable surface (a
# 1-block step is walkable in Minecraft). Applied in WORLD Y where a heightfield is available —
# see `compute_overhang_metrics` metric 4 for why that distinction is worth 24x on a cliff.
DEFAULT_FLOOR_STEP_TOLERANCE = 1

# Gate 2's go/no-go is "meaningfully above the marginal-probability baseline". "Meaningfully" has to
# be a number or it is not a gate. 5 percentage points is the starting value; it is deliberately not
# a ratio, because the baseline is 0.00% in the common case and every ratio against zero is either
# infinite or undefined.
DEFAULT_BASELINE_MARGIN_PTS = 5.0
# Gate 2's second condition (§7, as amended): the model must retain at least this fraction of ground
# truth's walkable voids. 0.25 is the plan's number, not this tool's invention.
DEFAULT_MIN_VOID_RETENTION = 0.25

DEFAULT_PROMPT = "a towering sandstone cliff with a deep undercut you can shelter beneath"

# The retention table, as (json path, printed label, format). One list, so the printed report and
# the JSON cannot drift apart.
RETENTION_KEYS: Tuple[Tuple[Tuple[str, ...], str, str], ...] = (
    (("overhang_cell_pct",), "Overhang cells (% of band)", "{:.4f}"),
    (("multi_run_column_pct",), "Multi-run columns (%)", "{:.4f}"),
    (("cavity", "count_per_1000_columns"), "Cavities per 1000 columns", "{:.3f}"),
    (("cavity", "mean_height"), "Mean cavity height (blocks)", "{:.3f}"),
    (("walkable", "voids_per_1000_columns"), "Walkable voids per 1000 cols", "{:.3f}"),
    (("walkable", "floor_cells_per_1000_columns"), "Walkable floor per 1000 cols", "{:.3f}"),
)
# Deliberately NOT in the table: `cavity.percentiles.p95`. A ratio of two percentiles is not a
# retention in §2.4's sense — it is a ratio of two order statistics from differently-shaped
# distributions, and it reads like a fraction of structure retained when it is not one. The
# percentiles are reported in full in the per-side structure block instead, and the gate check uses
# p90 as an absolute threshold rather than as a ratio.


# ---------------------------------------------------------------------------------------------
# Layout normalization — the transposition guard
# ---------------------------------------------------------------------------------------------

def _normalize_band(
    band: np.ndarray,
    layout: str = "auto",
    band_height: Optional[int] = None,
    threshold: float = 0.5,
    chunk_grid: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Normalizes any accepted band layout to the internal one: `[Z, X, I]` bool, 1 = solid.

    INTERNAL LAYOUT, stated once and relied on everywhere below: `[z][x][i]`, `i` ascending from
    the band bottom, `True` = solid. That is the ground-truth raster's own layout
    (`data/world_generator.py`, `[CANVAS, CANVAS, BAND_HEIGHT]`), so the ground-truth path is a
    no-op and only model output is moved.

    Accepted inputs:
      - `[Z, X, I]`  — ground-truth raster layout, `[z][x][i]`.
      - `[I, Z, X]`  — one chunk of occupancy head output, `[BAND_HEIGHT, 16, 16]`.
      - `[B, I, Z, X]` — a batch of chunks straight off the head (`[B, 64, 16, 16]`). Tiled into a
        `chunk_grid = (rows, cols)` arrangement, row-major, matching
        `diagnose_speckle.render_biome_area` and `inference_utils.render_area`. When `chunk_grid` is
        None it defaults to `sqrt(B) x sqrt(B)` and B must be a perfect square.

        PASS `chunk_grid` IF YOUR TILE IS NOT SQUARE. B alone cannot distinguish a 2x8 strip from a
        4x4 block — both have B=16 and both pass a perfect-square check — and assembling a 2x8 strip
        as 4x4 fabricates horizontal adjacency at four seams and destroys it at others. The
        walkable-void metric is a *connectivity* metric, so that is not a cosmetic error: it invents
        and severs exactly the joins the number counts. There is no way to detect this from the
        array, so the square default is a convention, not an inference.

    WHY THIS FUNCTION EXISTS AND IS TESTED SEPARATELY. `[Z, X, I]` and `[I, Z, X]` both have 64 in
    them and both produce plausible-looking numbers under the wrong reading — a transposed band
    reports the *world* transposed, which is not visible in any summary statistic.
    `docs/model-spec.md` has a whole section on why that class of bug is the dangerous one.
    `tests/test_diagnosis_tools.py` asserts the two layouts give bit-identical metric dicts on the
    same logical volume.

    Auto-detection is by matching an axis against `band_height` (default `BAND_HEIGHT`). A cube
    (`[H, H, H]`) is genuinely ambiguous and raises rather than guessing; pass `layout=` explicitly.

    Float input is thresholded at `threshold` (so occupancy *probabilities* work directly; for raw
    logits pass `threshold=0.0`). Integer/bool input is compared `!= 0` and `threshold` is ignored,
    because thresholding a uint8 0/1 raster at 0.5 and at 0.0 differ, and only one of those is what
    anyone means by "a band".
    """
    h = BAND_HEIGHT if band_height is None else int(band_height)
    a = np.asarray(band)

    if a.ndim == 4:
        if a.shape[1] != h:
            raise ValueError(
                f"4D input is read as [B, BAND_HEIGHT, Z, X] (the occupancy head's output); "
                f"axis 1 is {a.shape[1]}, expected {h}"
            )
        b = a.shape[0]
        if chunk_grid is None:
            span = int(round(float(np.sqrt(b))))
            if span * span != b:
                raise ValueError(
                    f"batch of {b} chunks is not a square grid, so horizontal chunk adjacency is "
                    f"undefined; pass chunk_grid=(rows, cols) or assemble the chunks into a "
                    f"[Z, X, {h}] grid before calling"
                )
            rows = cols = span
        else:
            rows, cols = int(chunk_grid[0]), int(chunk_grid[1])
            if rows * cols != b:
                raise ValueError(f"chunk_grid {rows}x{cols} does not describe {b} chunks")
        _, _, cz, cx = a.shape
        grid = np.empty((rows * cz, cols * cx, h), dtype=a.dtype)
        for idx in range(b):
            row, col = idx // cols, idx % cols
            grid[row * cz:(row + 1) * cz, col * cx:(col + 1) * cx, :] = np.moveaxis(a[idx], 0, 2)
        a = grid
        # The assembled grid is [Z, X, I] BY CONSTRUCTION, so say so instead of re-detecting. A
        # 4x4 tile of 16x16 chunks is [64, 64, 64] and auto-detection rightly calls that ambiguous
        # — which made a perfectly ordinary Gate 2 invocation (`--model band.npy` straight off the
        # head) fail with "pass layout= explicitly". Caught running the paired CLI end-to-end.
        layout = "zxi"

    if a.ndim != 3:
        raise ValueError(f"band must be 3D ([Z,X,I] or [I,Z,X]) or 4D ([B,I,Z,X]), got {a.shape}")

    if layout == "auto":
        first_is_i = a.shape[0] == h
        last_is_i = a.shape[2] == h
        if last_is_i and not first_is_i:
            layout = "zxi"
        elif first_is_i and not last_is_i:
            layout = "izx"
        elif first_is_i and last_is_i:
            raise ValueError(
                f"band shape {a.shape} is ambiguous (both the first and last axis are "
                f"{h}); pass layout='zxi' or layout='izx' explicitly rather than letting this "
                f"guess — a transposed band reports a transposed world and looks fine"
            )
        else:
            raise ValueError(
                f"neither the first nor the last axis of {a.shape} is the band depth {h}; "
                f"pass band_height= if this is a deliberately non-standard band"
            )
    if layout not in ("zxi", "izx"):
        raise ValueError(f"layout must be 'auto', 'zxi' or 'izx', got {layout!r}")
    if layout == "izx":
        a = np.moveaxis(a, 0, 2)

    if a.shape[2] != h:
        raise ValueError(f"band depth {a.shape[2]} != expected {h} after layout normalization")

    if a.dtype.kind == "f":
        # LOGITS AT THE PROBABILITY THRESHOLD ARE A PLAUSIBLE WRONG ANSWER, NOT A CRASH. Measured on
        # a synthetic head output in the range [-7.2, 8.4]: thresholding at the 0.5 default instead
        # of the correct 0.0 moved overhang cells from 31.64% to 39.80% and walkable voids per 1000
        # columns from 203.6 to 345.0 — biased toward AIR, i.e. toward a PASS, which is the
        # direction a diagnostic must never be silently wrong in. There is no way to tell a
        # probability from a logit by dtype, but a value outside [0, 1] settles it.
        lo, hi = (float(a.min()), float(a.max())) if a.size else (0.0, 0.0)
        if (lo < 0.0 or hi > 1.0) and threshold == 0.5:
            print(
                f"WARNING: float band spans [{lo:.3f}, {hi:.3f}], which is outside [0, 1] — these "
                f"look like LOGITS, not probabilities, and the threshold is the 0.5 probability "
                f"default. Pass --threshold 0.0 (or sigmoid the array first). Continuing at 0.5, "
                f"which biases the result toward air and toward a PASS.",
                file=sys.stderr,
            )
        return a >= threshold
    return a != 0


def heightfield_band(heightfield: np.ndarray) -> np.ndarray:
    """Builds the degenerate band a pure heightfield world has: `y <= h -> solid`, in band space.

    This is exactly what v1.1.1 produces and what `export/export_onnx.py:141-146` produced forever
    (docs/phase4-plan.md §0.2). Written out in band coordinates it is startling and worth stating:

        world_y = rint(h) + BAND_BOTTOM_OFFSET + i,  solid iff world_y <= rint(h)
                                                    iff BAND_BOTTOM_OFFSET + i <= 0
                                                    iff i <= 61

    — the anchor `h` cancels completely, so the v1.1.1 band is the *same 64 cells in every column
    of every world*: solid for i in [0, 61], air for i in {62, 63} (the BAND_TOP_OFFSET headroom).
    Zero overhang cells, zero multi-run columns, zero cavities, by construction and not by accident.
    That is the number Gate 0's new generator has to beat, and it is why this tool's degenerate case
    is worth a code path rather than a shrug.
    """
    h = np.asarray(heightfield)
    if h.ndim != 2:
        raise ValueError(f"heightfield must be 2D [Z, X], got {h.shape}")
    i = np.arange(BAND_HEIGHT)
    solid_i = (BAND_BOTTOM_OFFSET + i) <= 0
    assert int(solid_i.sum()) == BAND_HEIGHT - BAND_TOP_OFFSET, (
        "the plain heightfield expansion must fill everything up to the anchor and leave exactly "
        "BAND_TOP_OFFSET cells of headroom; if this fires, the band offsets moved"
    )
    return np.broadcast_to(solid_i, h.shape + (BAND_HEIGHT,)).copy()


# ---------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------

def _percentiles_from_counts(counts: np.ndarray, qs: Sequence[int]) -> Dict[str, int]:
    """Nearest-rank percentiles of an integer-valued distribution given as a bincount.

    Cavity heights are integers in `[1, BAND_HEIGHT]`, so a length-`BAND_HEIGHT+1` bincount is an
    *exact*, tiny representation of the whole distribution. Two consequences, both deliberate:
    percentiles over thousands of regions are exact rather than sampled (see
    `aggregate_overhang_metrics`), and the reported value is always an observed cavity height. No
    interpolation: 4.5 blocks of headroom is not a thing a player can stand in, and a p95 that
    reads 47.5 against §1.2's ~48-block ceiling would be a made-up number in the one place this
    tool exists to be believed.
    """
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    out: Dict[str, int] = {}
    if total == 0:
        return {f"p{q}": 0 for q in qs}
    cum = np.cumsum(counts)
    for q in qs:
        rank = max(1, int(np.ceil(q / 100.0 * total)))
        out[f"p{q}"] = int(np.searchsorted(cum, rank, side="left"))
    return out


def _walkable_components(
    cav_col: np.ndarray,
    cav_level: np.ndarray,
    nz: int,
    nx: int,
    tolerance: int,
) -> np.ndarray:
    """Groups tall cavities into walkable voids and returns each component's floor-cell count.

    A void is a connected component of the relation "4-adjacent columns (x OR z) whose floors are
    within `tolerance` blocks of each other". `cav_level` is the floor level of each cavity — WORLD
    Y when a heightfield was supplied, band index otherwise; see `compute_overhang_metrics`.

    WHY THIS IS A SPARSE GRAPH AND NOT `scipy.ndimage.label`. The first version of this function
    labelled a `[Z, X, I]` boolean mask with a 3x3x3 structuring element. That has two defects and
    both were shipped:

      1. A 3x3x3 element only reaches +/-1 along the level axis, so `floor_step_tolerance` SILENTLY
         SATURATED AT 1. Values of 2, 3, 5, 10 and 50 all produced byte-identical results while the
         report echoed the value back as if it had been applied — a measurement reported as made
         that was not made, and the one documented workaround for defect (2) therefore did nothing.
      2. It cannot express world-y adjacency at all, because a dense array indexed by band index
         forces the comparison into band-index space.

    A pair list plus `scipy.sparse.csgraph.connected_components` fixes both, is exact for any
    tolerance, and drops peak memory substantially (the dense label array was int32 over the whole
    `[Z, X, I]` volume — ~70 MB of labels for a 17 MB band, and it dominated peak RSS).

    Vectorized: for each of the two positive neighbour directions and each offset in
    `[-tolerance, +tolerance]`, one `searchsorted` against the sorted (column, level) key array.
    That is `2 * (2 * tolerance + 1)` passes over the cavity list, no Python loop over columns.
    """
    n = int(cav_col.size)
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    tol = max(0, int(tolerance))
    level = cav_level.astype(np.int64)
    lo = int(level.min())
    # Stride must exceed the level span plus both tolerance skirts, or a probe that walks off one
    # column's slot lands inside the next column's and joins two cavities that are nowhere near
    # each other. That is a silent over-merge, so the stride is derived rather than guessed.
    span = int(level.max()) - lo + 1
    stride = span + 2 * tol + 1
    key = cav_col.astype(np.int64) * stride + (level - lo + tol)
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]

    src: List[np.ndarray] = []
    dst: List[np.ndarray] = []
    # Only the POSITIVE directions: the graph is undirected, so a->a+1 already covers a+1->a.
    # d = 1 is the x neighbour (invalid on the last column of a row, hence the modulo mask);
    # d = nx is the z neighbour. Both axes, deliberately — a two-column ledge running along z is as
    # walkable as one running along x, and requiring a specific axis would make the headline number
    # depend on world orientation.
    x_of = cav_col % nx
    for d, valid in ((1, x_of < nx - 1), (nx, cav_col + nx < nz * nx)):
        base = (cav_col + d).astype(np.int64) * stride + (level - lo + tol)
        for dy in range(-tol, tol + 1):
            probe = base + dy
            pos = np.searchsorted(sorted_key, probe)
            in_range = pos < n
            hit = np.zeros(n, dtype=bool)
            hit[in_range] = sorted_key[pos[in_range]] == probe[in_range]
            hit &= valid
            if not hit.any():
                continue
            src.append(np.flatnonzero(hit))
            dst.append(order[pos[hit]])

    if src:
        rows = np.concatenate(src)
        cols = np.concatenate(dst)
    else:
        rows = cols = np.zeros(0, dtype=np.int64)
    graph = coo_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return np.bincount(labels).astype(np.int64)


def compute_overhang_metrics(
    band: np.ndarray,
    heightfield: Optional[np.ndarray] = None,
    layout: str = "auto",
    band_height: Optional[int] = None,
    threshold: float = 0.5,
    min_walkable_height: int = DEFAULT_MIN_WALKABLE_HEIGHT,
    min_walkable_floor: int = DEFAULT_MIN_WALKABLE_FLOOR,
    floor_step_tolerance: int = DEFAULT_FLOOR_STEP_TOLERANCE,
    chunk_grid: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Computes docs/phase4-plan.md §2.4's four metrics on one occupancy band.

    Args:
        band: occupancy, 1 = solid. `[Z,X,I]`, `[I,Z,X]` or `[B,I,Z,X]` — see `_normalize_band`.
        heightfield: `[Z, X]` absolute world y per column — the band's anchor. STRONGLY RECOMMENDED:
            without it the walkable-void metric degrades to band-index space and overcounts voids
            up to 24x on exactly the sloped terrain this release is about (see metric 4).
        layout / band_height / threshold / chunk_grid: forwarded to `_normalize_band`.
        min_walkable_height: minimum cavity height for "a player can enter it" (§2.4: 3).
        min_walkable_floor: minimum floor cells for the same (§2.4: 2).
        floor_step_tolerance: vertical slack, in blocks, when joining neighbouring columns' floors.

    Returns a dict of counts, percentages and distributions; see the module docstring for why every
    headline number is scale-free.

    THE FOUR DEFINITIONS, AND WHAT WAS AMBIGUOUS ABOUT THEM
    ------------------------------------------------------
    All four are defined **over the band only**, with two boundary rules that are not symmetric,
    because the band's two ends are not symmetric (spec_constants: below the band is solid rock,
    deterministically; above it is air or water):

      * BELOW index 0 there is a known-solid virtual cell, and it participates. A cavity whose floor
        is the band floor is a real cavity with a real floor — you can stand on it — so it counts,
        and the solid→air step into it counts as a transition. This costs the degenerate heightfield
        band nothing (its index 0 is always solid, so the virtual cell adds no transition).
      * ABOVE index BAND_HEIGHT-1 there is nothing known. An air run that reaches the top of the
        band is open to whatever is above and is NOT a cavity. Without this rule every column's
        ordinary above-ground sky counts as a 2-block cavity and the entire distribution is noise.

    1. OVERHANG CELLS. Air cells with at least one solid cell *strictly above them, inside the
       band*, as a percentage of band cells. Cells at the top index can never qualify. This is the
       cheap volumetric mass measure: it counts sheltered air, which is the thing you are standing
       in when the screenshot works.

    2. MULTI-RUN COLUMNS. Percentage of columns with more than one solid↔air transition, counting
       the virtual solid floor below index 0 as the sequence's first element. A plain heightfield
       column has exactly one transition; every column with two or more contains geometry no
       heightmap can express. This is Gate 2's explicit go/no-go statistic, so it is reported
       alongside `max_transitions_per_column` — a column with three solid runs must read 5
       transitions, not "2, saturated", and the test suite pins that.

    3. CAVITY HEIGHT. A cavity is a maximal run of air cells *within one column* that is roofed
       (bounded above by a solid cell inside the band). Its height is the number of air cells: the
       headroom. Deliberately per-column and 1D, not a 3D connected component: a sloping cave
       passage's 3D vertical extent is not headroom, and §1.2 wants to know "can a player-sized void
       fit in 64 blocks", which is a per-column question. Reported as a full distribution
       (percentiles + an exact bincount), plus `clipped_at_band_floor` — cavities whose bottom is
       band index 0 are *truncated* by the band and their measured height is a lower bound, which
       is the single number that says "the band is too shallow" as opposed to "the voids are big".

    4. WALKABLE VOIDS. §2.4: "cavities >=3 blocks tall with >=2 blocks of floor — the ones a player
       can actually enter, which is the only number that corresponds to the screenshot".

       THE AMBIGUITY, AND HOW IT WAS RESOLVED. A cavity as defined in (3) is one column wide, so its
       floor is exactly one cell and "≥2 blocks of floor" cannot be a statement about that column.
       It has to be horizontal extent. Resolved as:

         - "Floor" = the solid cell directly beneath a cavity's lowest air cell. Every cavity has
           one by construction (a maximal air run is bounded below by solid, or by the virtual
           band-floor cell, which is solid too), so the *vertical* half of "floor" carries no
           information and the whole condition is horizontal.
         - Two tall cavities share a floor when their columns are 4-adjacent (in x OR z) and their
           floors are within `floor_step_tolerance` blocks of each other IN WORLD Y.
         - A walkable void is a connected component of that relation; it qualifies when it has
           `min_walkable_floor` or more floor cells. The reported count is the number of
           **components**, not of cavities, so a 10x10 cave mouth is one walkable void and not a
           hundred — which is what "corresponds to the screenshot" means. `floor_cells` reports the
           area separately for anyone who wants it.

       WORLD Y, NOT BAND INDEX, AND THIS IS THE WHOLE BALLGAME. The band is anchored per column
       (`world_y = rint(h) + BAND_BOTTOM_OFFSET + i`), so equal band index means equal world y only
       where the ANCHOR is equal. A physically continuous floor reads N indices apart wherever
       `rint(h)` moves N blocks — and that is precisely the terrain this release exists to produce.
       Measured over the 600 v1.1.1 shards in `model/data/` (adjacent column pairs in x and z,
       |Δ rint(h)| > 1):

           snow_peaks               28.5%     (p99 delta 4, max 16)
           valley_mountains_craters 22.0%     (p99 4, max 21)
           fjord_rift               21.1%     (p99 4, max 21)
           deep_canyon               8.4%     (p99 9, max 46)
           mesa_plateaus             3.4%     (p99 6, max 43)

       Those are §2.1's cliff-undercut and arch archetypes: the ones the headline screenshot comes
       from. The flat archetypes are all under 1%, which is why this defect hides on easy terrain.

       Measuring in band-index space SEVERS components that are physically joined, and severing a
       component RAISES the component count. Measured on a 2:1 cliff (anchor drops 2 blocks per
       column) carrying one continuous 8x24 shelter 3 blocks tall whose floor is flat in world y —
       the plan's own "definition of done" scene, and `test_walkable_voids_cliff_fixture_world_y`:

           band index, tol=0    voids=24   floor_cells=192   largest=  8
           band index, tol=1    voids=24   floor_cells=192   largest=  8
           band index, tol=2    voids= 1   floor_cells=192   largest=192
           world y,    tol=0    voids= 1   floor_cells=192   largest=192
           world y,    tol=1    voids= 1   floor_cells=192   largest=192   <- truth: 1 shelter,
                                                                              192 floor cells

       A 24x overcount of §9's "the number that matches the screenshot". Note that band index at
       tol=2 happens to recover this scene — because the slope is exactly 2 — which is the point:
       in band-index space the tolerance that works is the local grade, and there is no single
       correct value. In world y, tol=0 is already right. On a gently-rolling synthetic region
       (mean |Δh| 0.34, nearly the friendliest possible case) band index gives 173 voids where
       world y gives 70 — still 2.5x.

       AN EARLIER VERSION OF THIS DOCSTRING CLAIMED THE OPPOSITE — that band-index space
       "UNDERCOUNTS on sloped ground and never overcounts", and that this was the safe direction.
       That was false, and it was false about the headline number: it undercounts floor *cells per
       void* while overcounting *voids*. Recorded here because the wrong version was confidently
       written and survived review once.

       When no heightfield is supplied the metric falls back to band-index space, `floor_space`
       reads `"band_index"`, and the number is an inflated upper bound on the void count. It is not
       silently comparable with a world-y number; `main` refuses to mix the two.

       Two remaining sub-choices:
         - 4-connectivity in BOTH x and z, not one axis. A two-column ledge running along z is as
           walkable as one running along x; requiring a specific axis would make the headline number
           depend on world orientation.
         - `floor_step_tolerance` defaults to 1, not 0, because a 1-block step is walkable in
           Minecraft. Unlike the previous implementation it is now actually applied at any value
           (see `_walkable_components`, which documents the saturation bug it replaces).
    """
    solid = _normalize_band(band, layout=layout, band_height=band_height, threshold=threshold,
                            chunk_grid=chunk_grid)
    nz, nx, h = solid.shape
    anchors = None
    if heightfield is not None:
        hf = np.asarray(heightfield, dtype=np.float64)
        if hf.shape != (nz, nx):
            raise ValueError(
                f"heightfield {hf.shape} does not match the band's column grid {(nz, nx)}"
            )
        # `np.rint` is half-to-even, matching the spec_constants formula and `viz.py`'s anchor.
        # Rounding one way here and the other there shifts every floor by a block.
        anchors = np.rint(hf).astype(np.int64).reshape(-1)
    n_cols = nz * nx
    band_cells = n_cols * h
    cols = solid.reshape(n_cols, h)
    air = ~cols

    # --- 1. overhang cells -------------------------------------------------------------------
    # Suffix-max along the band index: `any_at_or_above[c, i]` = is there solid at index >= i.
    # Shifting it down by one gives "strictly above", which is the definition. Same construction as
    # `viz.render_cross_section`'s sheltered-air mask, deliberately — if these two ever disagree,
    # the picture and the number disagree, and it is not obvious from either which is wrong.
    any_at_or_above = np.maximum.accumulate(cols[:, ::-1], axis=1)[:, ::-1]
    solid_above = np.zeros_like(cols)
    solid_above[:, :-1] = any_at_or_above[:, 1:]
    overhang = air & solid_above
    overhang_cells = int(overhang.sum())

    # --- 2. multi-run columns ----------------------------------------------------------------
    internal = (cols[:, 1:] != cols[:, :-1]).sum(axis=1).astype(np.int64)
    floor_step = (~cols[:, 0]).astype(np.int64)   # the virtual solid cell below index 0
    transitions = internal + floor_step
    multi_run = transitions > 1
    multi_run_columns = int(multi_run.sum())
    # Bucketed so "how multi" is visible: 1 is a heightfield, 2 is one cavity, 5 is three solid runs.
    t_hist_raw = np.bincount(transitions, minlength=6)
    transition_histogram = {
        "0_all_one_material": int(t_hist_raw[0]),
        "1_heightfield": int(t_hist_raw[1]),
        "2": int(t_hist_raw[2]),
        "3": int(t_hist_raw[3]),
        "4": int(t_hist_raw[4]),
        "5_plus": int(t_hist_raw[5:].sum()),
    }

    # --- 3. cavities -------------------------------------------------------------------------
    # Vectorized run extraction: pad each column's air mask with a False cell at each end and
    # flatten. The padding is what stops runs merging across the column boundary in the flat view,
    # so the whole [520,520,64] region is three numpy passes and no Python loop over 270k columns.
    padded = np.zeros((n_cols, h + 2), dtype=bool)
    padded[:, 1:h + 1] = air
    flat = padded.ravel()
    d = np.diff(flat.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1)
    lengths = ends - starts + 1
    run_col = starts // (h + 2)
    start_i = starts % (h + 2) - 1
    end_i = ends % (h + 2) - 1
    # Roofed = there is a band cell above the run's top, and it is solid (the run is maximal, so
    # the cell above it is solid whenever it exists). end_i == h-1 means the run reaches the band
    # top and is open to whatever is above: not a cavity.
    roofed = end_i <= h - 2
    cav_height = lengths[roofed].astype(np.int64)
    cav_col = run_col[roofed]
    cav_floor_i = start_i[roofed].astype(np.int64)

    height_counts = np.bincount(cav_height, minlength=h + 1)[: h + 1]
    n_cavities = int(cav_height.size)
    clipped = int((cav_floor_i == 0).sum())
    over_warn = int((cav_height > BAND_SIZE_WARNING_HEIGHT).sum())

    # --- 4. walkable voids -------------------------------------------------------------------
    tall = cav_height >= min_walkable_height
    tall_col = cav_col[tall]
    if anchors is not None:
        # The floor's ABSOLUTE WORLD Y. This is the fix that took the cliff fixture from 21 voids
        # to 1; see the metric-4 discussion above for the measurements.
        tall_level = anchors[tall_col] + BAND_BOTTOM_OFFSET + cav_floor_i[tall]
        floor_space = "world_y"
    else:
        tall_level = cav_floor_i[tall]
        floor_space = "band_index"
    comp_sizes = _walkable_components(tall_col, tall_level, nz, nx, floor_step_tolerance)
    if comp_sizes.size:
        qualifying = comp_sizes >= min_walkable_floor
        walkable_void_count = int(qualifying.sum())
        walkable_floor_cells = int(comp_sizes[qualifying].sum())
        largest_void = int(comp_sizes.max())
    else:
        walkable_void_count = 0
        walkable_floor_cells = 0
        largest_void = 0

    def _pct(num: float, den: float) -> float:
        return float(100.0 * num / den) if den > 0 else 0.0

    return {
        "grid_shape": [int(nz), int(nx)],
        "band_height": int(h),
        "num_columns": int(n_cols),
        "band_cells": int(band_cells),
        "solid_fraction": float(cols.mean()) if band_cells else 0.0,
        "solid_fraction_by_band_index": [float(v) for v in cols.mean(axis=0)],
        # 1
        "overhang_cells": overhang_cells,
        "overhang_cell_pct": _pct(overhang_cells, band_cells),
        # 2
        "multi_run_columns": multi_run_columns,
        "multi_run_column_pct": _pct(multi_run_columns, n_cols),
        "mean_transitions_per_column": float(transitions.mean()) if n_cols else 0.0,
        "max_transitions_per_column": int(transitions.max()) if n_cols else 0,
        "transition_histogram": transition_histogram,
        # 3
        "cavity": {
            "count": n_cavities,
            "count_per_1000_columns": float(1000.0 * n_cavities / n_cols) if n_cols else 0.0,
            "height_counts": [int(v) for v in height_counts],
            "mean_height": float(cav_height.mean()) if n_cavities else 0.0,
            "max_height": int(cav_height.max()) if n_cavities else 0,
            "percentiles": _percentiles_from_counts(height_counts, (50, 90, 95, 99)),
            "clipped_at_band_floor": clipped,
            "clipped_at_band_floor_pct": _pct(clipped, n_cavities),
            f"taller_than_{BAND_SIZE_WARNING_HEIGHT}": over_warn,
            f"taller_than_{BAND_SIZE_WARNING_HEIGHT}_pct": _pct(over_warn, n_cavities),
        },
        # 4
        "walkable": {
            "void_count": walkable_void_count,
            "voids_per_1000_columns": float(1000.0 * walkable_void_count / n_cols) if n_cols else 0.0,
            "floor_cells": walkable_floor_cells,
            # NOT a percentage. One column can hold several tall cavities and contribute a floor
            # cell for each, so this exceeded 100% "of columns" on a stacked-cavity fixture (300%
            # was observed) while being labelled a percentage of columns. A rate per 1000 columns
            # says the same thing and cannot be misread as a fraction.
            "floor_cells_per_1000_columns": float(1000.0 * walkable_floor_cells / n_cols) if n_cols else 0.0,
            # Gate 2 condition 3's quantity: how big the average walkable void actually is. Void
            # COUNT and total floor AREA can both be retained by a model that shatters ground
            # truth's caverns into hundreds of minimum-size pockets; the mean does not survive that.
            # See `evaluate_baseline_gate` for the adversarial band this exists to catch.
            "mean_floor_cells_per_void": (
                float(walkable_floor_cells / walkable_void_count) if walkable_void_count else 0.0
            ),
            "largest_void_floor_cells": largest_void,
            "min_height": int(min_walkable_height),
            "min_floor_cells": int(min_walkable_floor),
            "floor_step_tolerance": int(floor_step_tolerance),
            "floor_space": floor_space,
        },
    }


def aggregate_overhang_metrics(metrics_list: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pools per-region metrics into one dict with the same schema, for Gate 3's thousands of
    regions. Counts add; percentages are recomputed from the pooled counts (never averaged —
    averaging percentages across regions of different sizes is the classic way to report a number
    nobody can reproduce). Percentiles are recomputed from the pooled height bincount and are
    therefore exact, not approximated from per-region percentiles.

    Regions are pooled but never *joined*: walkable voids are counted within each region and summed,
    so a void straddling two regions' shared edge is counted twice (or, if it is only 1 floor cell
    wide on each side, not at all). Regions are independently-seeded windows on the canonical field
    (`world_generator.generate_shards`) and are not spatial neighbours, so there is no shared edge to
    join in practice; this note exists so nobody later assumes there is.
    """
    if not metrics_list:
        raise ValueError("nothing to aggregate")
    if len(metrics_list) == 1:
        # Same schema for one region as for many — an early `return metrics_list[0]` left every
        # single-region Gate 0 JSON without `num_regions_pooled`, so a consumer had to handle two
        # shapes for no reason.
        single = dict(metrics_list[0])
        single["num_regions_pooled"] = 1
        return single
    heights = {m["band_height"] for m in metrics_list}
    if len(heights) != 1:
        raise ValueError(f"cannot pool regions with different band depths: {sorted(heights)}")
    h = heights.pop()
    spaces = {m["walkable"]["floor_space"] for m in metrics_list}
    if len(spaces) != 1:
        raise ValueError(
            f"cannot pool regions measured in different floor spaces: {sorted(spaces)} — some "
            f"shards supplied a heightfield and some did not, so their void counts are not "
            f"commensurable (see compute_overhang_metrics metric 4)"
        )

    n_cols = sum(int(m["num_columns"]) for m in metrics_list)
    band_cells = sum(int(m["band_cells"]) for m in metrics_list)
    overhang_cells = sum(int(m["overhang_cells"]) for m in metrics_list)
    multi_run = sum(int(m["multi_run_columns"]) for m in metrics_list)
    solid_cells = sum(float(m["solid_fraction"]) * int(m["band_cells"]) for m in metrics_list)
    profile = np.zeros(h, dtype=np.float64)
    t_hist: Dict[str, int] = {}
    height_counts = np.zeros(h + 1, dtype=np.int64)
    transitions_weighted = 0.0
    for m in metrics_list:
        w = int(m["num_columns"])
        profile += np.asarray(m["solid_fraction_by_band_index"], dtype=np.float64) * w
        transitions_weighted += float(m["mean_transitions_per_column"]) * w
        for k, v in m["transition_histogram"].items():
            t_hist[k] = t_hist.get(k, 0) + int(v)
        height_counts += np.asarray(m["cavity"]["height_counts"], dtype=np.int64)
    profile /= max(n_cols, 1)

    n_cav = int(height_counts.sum())
    mean_height = float((height_counts * np.arange(h + 1)).sum() / n_cav) if n_cav else 0.0
    max_height = int(np.flatnonzero(height_counts)[-1]) if n_cav else 0
    clipped = sum(int(m["cavity"]["clipped_at_band_floor"]) for m in metrics_list)
    over_key = f"taller_than_{BAND_SIZE_WARNING_HEIGHT}"
    over_warn = sum(int(m["cavity"][over_key]) for m in metrics_list)
    voids = sum(int(m["walkable"]["void_count"]) for m in metrics_list)
    floor_cells = sum(int(m["walkable"]["floor_cells"]) for m in metrics_list)
    first_walk = metrics_list[0]["walkable"]

    def _pct(num: float, den: float) -> float:
        return float(100.0 * num / den) if den > 0 else 0.0

    return {
        "grid_shape": [int(sum(m["grid_shape"][0] for m in metrics_list)), int(metrics_list[0]["grid_shape"][1])],
        "band_height": int(h),
        "num_columns": n_cols,
        "band_cells": band_cells,
        "num_regions_pooled": len(metrics_list),
        "solid_fraction": float(solid_cells / band_cells) if band_cells else 0.0,
        "solid_fraction_by_band_index": [float(v) for v in profile],
        "overhang_cells": overhang_cells,
        "overhang_cell_pct": _pct(overhang_cells, band_cells),
        "multi_run_columns": multi_run,
        "multi_run_column_pct": _pct(multi_run, n_cols),
        "mean_transitions_per_column": float(transitions_weighted / n_cols) if n_cols else 0.0,
        "max_transitions_per_column": max(int(m["max_transitions_per_column"]) for m in metrics_list),
        "transition_histogram": t_hist,
        "cavity": {
            "count": n_cav,
            "count_per_1000_columns": float(1000.0 * n_cav / n_cols) if n_cols else 0.0,
            "height_counts": [int(v) for v in height_counts],
            "mean_height": mean_height,
            "max_height": max_height,
            "percentiles": _percentiles_from_counts(height_counts, (50, 90, 95, 99)),
            "clipped_at_band_floor": clipped,
            "clipped_at_band_floor_pct": _pct(clipped, n_cav),
            over_key: over_warn,
            f"{over_key}_pct": _pct(over_warn, n_cav),
        },
        "walkable": {
            "void_count": voids,
            "voids_per_1000_columns": float(1000.0 * voids / n_cols) if n_cols else 0.0,
            "floor_cells": floor_cells,
            "floor_cells_per_1000_columns": float(1000.0 * floor_cells / n_cols) if n_cols else 0.0,
            "largest_void_floor_cells": max(int(m["walkable"]["largest_void_floor_cells"]) for m in metrics_list),
            "min_height": int(first_walk["min_height"]),
            "min_floor_cells": int(first_walk["min_floor_cells"]),
            "floor_step_tolerance": int(first_walk["floor_step_tolerance"]),
            "floor_space": first_walk["floor_space"],
        },
    }


# ---------------------------------------------------------------------------------------------
# Baseline and retention — the two comparisons this tool exists to make
# ---------------------------------------------------------------------------------------------

def marginal_baseline_band(
    band: np.ndarray,
    layout: str = "auto",
    band_height: Optional[int] = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """Builds the marginal-probability baseline band: the answer a model that learned nothing except
    the average band would give.

    docs/phase4-plan.md §3.1 states the failure this guards against, and it has shipped three times
    in 2D already:

        "the BCE-optimal answer for an unpredictable binary quantity is its marginal probability. A
         band that is 70% solid on average is best answered — pointwise — with 0.7 everywhere.
         Threshold that at 0.5 and you get solid everywhere: a heightfield. [...] Both score well.
         Both are worthless."

    So: take the ground truth's per-band-index marginal `p[i]` (mean over all columns), threshold at
    0.5, and tile the resulting single template column across the whole grid. Running the same
    metrics on *that* gives the score a model gets for learning nothing, and Gate 2's go/no-go is
    whether the real model beats it.

    A PROPERTY OF THIS BASELINE THAT READERS SHOULD KNOW: every column is identical, so its
    multi-run column percentage is 0.00% or 100.00% and nothing in between. In the normal case the
    marginal is monotonically decreasing in `i` (deep is solid, high is air), the template has one
    transition, and the baseline reads 0.00%. A baseline of 100.00% means the *average* band itself
    has multiple solid runs — a strong statement about the data, and a sign that this comparison has
    stopped being informative and needs a look rather than a rubber stamp.
    """
    solid = _normalize_band(band, layout=layout, band_height=band_height, threshold=threshold)
    p = solid.reshape(-1, solid.shape[2]).mean(axis=0)
    return marginal_baseline_from_profile(p, solid.shape[:2])


def marginal_baseline_from_profile(
    profile: Sequence[float],
    grid_shape: Tuple[int, int],
) -> np.ndarray:
    """`marginal_baseline_band` from an already-computed per-band-index marginal, so the baseline
    over many pooled regions uses the pooled marginal rather than the first region's.

    `grid_shape` only affects the walkable-void metric, which is a connectivity measure and needs
    somewhere to be connected; any representative region shape gives the same per-column
    percentages, because every column of the baseline is the same column."""
    template = np.asarray(profile, dtype=np.float64) >= 0.5
    return np.broadcast_to(template, (int(grid_shape[0]), int(grid_shape[1]), template.size)).copy()


def _dig(d: Dict[str, Any], path: Sequence[str]) -> Any:
    for k in path:
        d = d[k]
    return d


def compute_retention(
    gt_metrics: Dict[str, Any],
    model_metrics: Optional[Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    """model / ground truth, for every headline metric, as a first-class result.

    UNDEFINED IS A VALUE HERE, SPELLED `None`. When the ground-truth metric is 0 the ratio does not
    exist, and reporting it as `inf` (numpy's answer), `nan` (numpy's answer for 0/0) or `0.0` (the
    lazy answer) all read as *results* in a table that gets copied into docs/phase4-plan.md §9. The
    degenerate case is not hypothetical: on a v1.1.1 heightfield band every one of these
    denominators is exactly 0, and "retention: 0%" would be a false claim about a model that may
    well have produced overhangs the ground truth never had.
    """
    out: Dict[str, Optional[float]] = {}
    for path, _label, _fmt in RETENTION_KEYS:
        key = ".".join(path)
        if model_metrics is None:
            out[key] = None
            continue
        gt_v = float(_dig(gt_metrics, path))
        m_v = float(_dig(model_metrics, path))
        out[key] = float(m_v / gt_v) if gt_v != 0.0 else None
    return out


def evaluate_baseline_gate(
    observed_metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Any],
    margin_pts: float = DEFAULT_BASELINE_MARGIN_PTS,
    gt_metrics: Optional[Dict[str, Any]] = None,
    min_void_retention: float = DEFAULT_MIN_VOID_RETENTION,
) -> Dict[str, Any]:
    """Gate 2's go/no-go (docs/phase4-plan.md §7), which is a **conjunction of three conditions**:

        1. multi-run column rate >= marginal baseline + `margin_pts`
        2. walkable-void retention >= `min_void_retention` of ground truth
        3. p90 cavity height >= `min_walkable_height` (the observed side's own setting)

    ALL THREE, or the verdict is FAIL. It is not a headline number with warnings attached, because
    a warning next to the word PASS is a warning nobody acts on.

    WHY IT IS A CONJUNCTION, MEASURED. Condition 1 alone is gameable by the exact failure §3.1
    predicts. A random 1-block-speckle band — uniformly carved single-cell pockets at 18% density
    over a 64x64 column tile, no structure of any kind — scores a **perfect** 100.00% multi-run
    columns against a 0.00% baseline (+100.00 points, the most emphatic possible PASS) while
    containing nothing a player can enter: every cavity is exactly 1 block tall (p50 = p90 = max =
    1) and there are 0 walkable voids. Under the conjunction it now reads FAIL on conditions 2 and
    3. `test_baseline_gate_speckle_fails_the_conjunction` reproduces the fixture and pins all of it.

    Condition 3 is what kills it, and it is deliberately p90 rather than "are there any walkable
    voids at all". An earlier version of this check keyed on `voids == 0`, and adding ONE 4x4 room
    to 0.39% of the columns flipped that to false and silenced the entire warning while the band was
    still 99.6% speckle. A percentile cannot be bought off that cheaply.

    Note also that `overhang_cell_pct` does NOT discriminate here and must not be used as a
    substitute. It measures how much air has solid above it, which says nothing about whether that
    air is enterable: the speckle band above scores 14.06% while a structured carved band scored
    6.07% on the same 64-deep geometry. But the ordering is NOT a stable property — it is purely a
    function of the two densities, and on a 12-deep fixture with a 4-tall room the structured band
    scores double the speckle (33.3% vs 16.7%). An earlier draft of this docstring asserted the
    ordering was stable; the test fixture disproved it. The claim that survives is the weaker and
    sufficient one: a band with ZERO enterable structure can post a large overhang percentage, so
    the metric cannot stand in for conditions 2 and 3.

    NOT IN SCOPE HERE: §7's fourth condition, "relief retention not regressed from v1.1.1". That is
    measured by the terrain tooling, not this one. `main` prints a line saying so, because a PASS
    from this tool is three quarters of the gate and must not be read as all of it.

    `gt_metrics` is optional. Without it (a Gate 0 ground-truth-only run) there is no model and no
    retention, so condition 2 is not evaluable; conditions are then reported as measured on the
    ground truth itself and `is_gate2_verdict` is False. That run is a check that the DATA has
    structure the average band does not — useful, and not a Gate 2 decision.
    """
    observed = float(observed_metrics["multi_run_column_pct"])
    baseline = float(baseline_metrics["multi_run_column_pct"])
    delta = observed - baseline
    margin_met = bool(delta >= margin_pts)

    walk = observed_metrics.get("walkable", {})
    min_h = int(walk.get("min_height", DEFAULT_MIN_WALKABLE_HEIGHT))
    percentiles = observed_metrics.get("cavity", {}).get("percentiles", {})
    p50 = int(percentiles.get("p50", 0))
    p90 = int(percentiles.get("p90", 0))
    voids = float(walk.get("voids_per_1000_columns", 0.0))

    height_met = bool(p90 >= min_h)

    retention: Optional[float] = None
    if gt_metrics is not None:
        gt_voids = float(gt_metrics["walkable"]["voids_per_1000_columns"])
        # Undefined, not 0.0 and not inf: ground truth with no walkable voids cannot say whether a
        # model retained them. An undefined condition is NOT met — the gate authorizes a 9-hour run
        # and must not do so on an unmeasurable criterion.
        retention = float(voids / gt_voids) if gt_voids != 0.0 else None
    retention_met = bool(retention is not None and retention >= min_void_retention)

    is_gate2 = gt_metrics is not None
    # Flipping any condition in or out of the verdict is this one line, by design.
    passed = margin_met and height_met and (retention_met if is_gate2 else True)

    return {
        "observed_multi_run_column_pct": observed,
        "baseline_multi_run_column_pct": baseline,
        "delta_pts": delta,
        "required_margin_pts": float(margin_pts),
        "observed_p50_cavity_height": p50,
        "observed_p90_cavity_height": p90,
        "required_p90_cavity_height": min_h,
        "observed_voids_per_1000_columns": voids,
        "walkable_void_retention": retention,
        "required_void_retention": float(min_void_retention),
        "conditions": {
            "multi_run_margin": margin_met,
            "walkable_void_retention": retention_met if is_gate2 else None,
            "p90_cavity_height": height_met,
        },
        "is_gate2_verdict": is_gate2,
        "verdict": "PASS" if passed else "FAIL",
        "baseline_is_degenerate": baseline >= 100.0,
    }


# ---------------------------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------------------------

def save_diagnostic_plot(
    gt_metrics: Dict[str, Any],
    model_metrics: Optional[Dict[str, Any]],
    baseline_metrics: Dict[str, Any],
    output_path: str,
    title: str = "Overhang Diagnostic Report",
) -> None:
    """Saves a three-panel figure: the cavity-height distribution that validates the band size, the
    per-band-index occupancy marginal the baseline is built from, and the headline comparison.

    Driven entirely by metrics dicts and never by the raw band, so a figure can be redrawn from a
    saved JSON report without the 17 MB array — and so there is no second, subtly different
    computation of the same quantity living in the plotting code.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- panel 0: cavity height distribution -------------------------------------------------
    h = int(gt_metrics["band_height"])
    heights = np.arange(h + 1)
    gt_counts = np.asarray(gt_metrics["cavity"]["height_counts"], dtype=float)
    axes[0].bar(heights, gt_counts, width=0.9, color="sienna", alpha=0.75,
                edgecolor="black", linewidth=0.3, label="ground truth")
    if model_metrics is not None:
        m_counts = np.asarray(model_metrics["cavity"]["height_counts"], dtype=float)
        axes[0].step(heights, m_counts, where="mid", color="steelblue", linewidth=1.6, label="model")
    axes[0].axvline(BAND_SIZE_WARNING_HEIGHT, color="crimson", linestyle="--", linewidth=1.2,
                    label=f"§1.2 concern ({BAND_SIZE_WARNING_HEIGHT})")
    axes[0].axvline(h, color="black", linestyle=":", linewidth=1.2, label=f"band depth ({h})")
    axes[0].set_xlabel("Cavity height (blocks of headroom)")
    axes[0].set_ylabel("Cavity count")
    axes[0].set_yscale("symlog")
    axes[0].set_title("Cavity height distribution\n(validates BAND_HEIGHT — §1.2)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", linestyle="--", alpha=0.4)

    # --- panel 1: the occupancy marginal, i.e. the baseline, drawn ---------------------------
    prof_gt = np.asarray(gt_metrics["solid_fraction_by_band_index"], dtype=float)
    idx = np.arange(prof_gt.size)
    axes[1].plot(prof_gt, idx, color="sienna", linewidth=1.8, label="ground truth")
    if model_metrics is not None:
        axes[1].plot(np.asarray(model_metrics["solid_fraction_by_band_index"], dtype=float), idx,
                     color="steelblue", linewidth=1.8, label="model")
    axes[1].axvline(0.5, color="crimson", linestyle="--", linewidth=1.2, label="0.5 threshold")
    baseline_template = np.asarray(baseline_metrics["solid_fraction_by_band_index"], dtype=float)
    axes[1].fill_betweenx(idx, 0.0, baseline_template, color="grey", alpha=0.25,
                          step="mid", label="thresholded baseline band")
    axes[1].set_xlabel("Solid fraction")
    axes[1].set_ylabel("Band index i (0 = band bottom)")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_title("Per-band-index occupancy marginal\n(the baseline a collapsed model returns)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    # --- panel 2: the headline comparison ----------------------------------------------------
    labels = ["Overhang\ncells (%)", "Multi-run\ncolumns (%)", "Walkable voids\nper 1000 cols"]
    def _triple(m: Optional[Dict[str, Any]]) -> List[float]:
        if m is None:
            return [0.0, 0.0, 0.0]
        return [float(m["overhang_cell_pct"]), float(m["multi_run_column_pct"]),
                float(m["walkable"]["voids_per_1000_columns"])]
    series = [("ground truth", _triple(gt_metrics), "sienna"),
              ("marginal baseline", _triple(baseline_metrics), "grey")]
    if model_metrics is not None:
        series.insert(1, ("model", _triple(model_metrics), "steelblue"))
    x = np.arange(len(labels))
    width = 0.8 / len(series)
    for k, (name, vals, color) in enumerate(series):
        axes[2].bar(x + k * width - 0.4 + width / 2, vals, width=width, color=color,
                    edgecolor="black", linewidth=0.4, alpha=0.85, label=name)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=9)
    axes[2].set_yscale("symlog", linthresh=0.01)
    axes[2].set_title("Headline metrics (§2.4)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------

class LoadedBand(NamedTuple):
    """One loaded band plus everything a caller needs to avoid misreporting it.

    `is_heightfield_fallback` is a BOOLEAN, not a substring of `source`. The first version only had
    the string, and every check was `"DERIVED-HEIGHTFIELD-FALLBACK" in sources[0]` — so a directory
    where only the *second* shard was stale printed no warning at all and pooled its zeros into the
    total, and the `--model` path never checked at all. A stale v1.1.1 shard passed as `--model`
    then printed `Multi-run columns  41.96 | 0.00 | 0.00%` and `VERDICT: FAIL`, which reads as "the
    model learned nothing": the single most expensive wrong conclusion this tool can produce, and
    exactly docs/phase4-plan.md §5.1's silent-downgrade risk arriving through the diagnostic instead
    of through the mod.
    """
    band: np.ndarray
    heightfield: Optional[np.ndarray]
    source: str
    is_heightfield_fallback: bool


def load_band(path: str, band_key: Optional[str] = None) -> LoadedBand:
    """Loads one band (and its heightfield anchor, when the source has one) from a `.npy` or a
    `region_*.npz` shard.

    The heightfield is loaded and RETURNED, not discarded: it is what makes the walkable-void metric
    world-y-correct rather than band-index-approximate (a 24x difference on a cliff — see
    `compute_overhang_metrics` metric 4). It is right there in the shard next to the band.

    If the shard has no band array but does have a heightfield, the degenerate v1.1.1 band is
    derived (`heightfield_band`) and `is_heightfield_fallback` is True.
    """
    base = os.path.basename(path)
    if path.endswith(".npy"):
        return LoadedBand(np.load(path), None, f"{base} (raw array, no heightfield)", False)
    with np.load(path) as npz:
        keys = list(npz.files)
        hf = npz["heightfield"] if "heightfield" in keys else None
        hf_note = "" if hf is not None else ", no heightfield"
        if band_key is not None:
            if band_key not in keys:
                raise KeyError(f"{path} has no array named {band_key!r}; it has {keys}")
            return LoadedBand(npz[band_key], hf, f"{base} key '{band_key}'{hf_note}", False)
        for k in BAND_KEY_CANDIDATES:
            if k in keys:
                return LoadedBand(npz[k], hf, f"{base} key '{k}'{hf_note}", False)
        if hf is not None:
            return LoadedBand(
                heightfield_band(hf), hf,
                f"{base} DERIVED-HEIGHTFIELD-FALLBACK (no band array in shard)", True,
            )
    raise KeyError(
        f"{path} contains neither a band array ({BAND_KEY_CANDIDATES}) nor a heightfield; "
        f"pass --band-key if it is stored under another name"
    )


def resolve_band_paths(path: str, max_regions: Optional[int] = None) -> List[str]:
    """Expands a file, a directory of `region_*.npz` shards, or a glob into a sorted path list.

    A directory is globbed for `region_*.npz` and then for `*.npz` — never for `*.npy`. Diagnostic
    arrays get dropped in shard directories, and sweeping them into the pool silently changes the
    denominator of every percentage in the report. Point the tool at an explicit glob if you really
    do want to pool raw arrays.
    """
    if os.path.isdir(path):
        paths = sorted(glob.glob(os.path.join(path, "region_*.npz")))
        if not paths:
            paths = sorted(glob.glob(os.path.join(path, "*.npz")))
        if not paths:
            raise FileNotFoundError(f"no region_*.npz or *.npz shards found in {path}")
    elif os.path.isfile(path):
        paths = [path]
    else:
        paths = sorted(glob.glob(path))
        if not paths:
            raise FileNotFoundError(f"no such file, directory or glob match: {path}")
    if max_regions is not None:
        paths = paths[:max_regions]
    return paths


def render_band_area(
    model: Any,
    prompt: str,
    seed: int,
    chunk_span: int = 4,
    origin_chunk: Tuple[int, int] = (0, 0),
    device: str = "cpu",
    max_tokens: int = 128,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Runs the model over a `chunk_span x chunk_span` tile of chunks and returns
    `(band, heightfield)` — the band as `[chunk_span*16, chunk_span*16, BAND_HEIGHT]` in the
    internal `[Z, X, I]` layout, and the heightmap head's output as `[Z, X]` so the model side can
    be measured in world y exactly like the ground-truth side.

    Sibling of `diagnose_speckle.render_biome_area`, same row-major chunk layout. Requires the
    occupancy head from docs/phase4-plan.md §1.3, which is Task B1 — a checkpoint without it fails
    here with an explicit message rather than a `KeyError`, because "the model has no occupancy
    head" and "the model produced no overhangs" are very different findings and Gate 2 must not
    confuse them.
    """
    import torch
    from mc_imagine_model.tokenizer_utils import tokenize

    model.eval()
    tokens = tokenize(prompt, max_tokens=max_tokens)
    b = chunk_span * chunk_span
    ox, oz = origin_chunk
    with torch.no_grad():
        out = model(
            torch.tensor([tokens] * b, dtype=torch.int32, device=device),
            torch.tensor([ox + (i % chunk_span) for i in range(b)], dtype=torch.int32, device=device),
            torch.tensor([oz + (i // chunk_span) for i in range(b)], dtype=torch.int32, device=device),
            torch.tensor([seed] * b, dtype=torch.int64, device=device),
        )
    key = next((k for k in ("occupancy_logits", "occupancy", "band_logits") if k in out), None)
    if key is None:
        raise KeyError(
            "this checkpoint's forward() returns no occupancy output "
            f"(saw {sorted(out.keys())}) — it is a pre-v1.1.2 heightfield-only model, not a "
            "volumetric one that failed to produce overhangs"
        )
    logits = out[key].detach().to("cpu").float().numpy()   # [B, BAND_HEIGHT, 16, 16]
    # Logits, so the decision boundary is 0.0, not 0.5. `_normalize_band` is told this explicitly
    # by thresholding here rather than passing floats downstream with an implied convention.
    # `chunk_grid` is passed even though the tile is square: the span is KNOWN here, so there is no
    # reason to hand `_normalize_band` a shape it has to guess from B.
    band = _normalize_band(logits >= 0.0, layout="izx", chunk_grid=(chunk_span, chunk_span))

    hf = None
    if "heightmap" in out:
        hm = out["heightmap"].detach().to("cpu").float().numpy()   # [B, 16, 16]
        cz, cx = hm.shape[1], hm.shape[2]
        hf = np.zeros((chunk_span * cz, chunk_span * cx), dtype=np.float32)
        for idx in range(b):
            row, col = idx // chunk_span, idx % chunk_span
            hf[row * cz:(row + 1) * cz, col * cx:(col + 1) * cx] = hm[idx]
    return band, hf


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------

def _fmt_retention(value: Optional[float]) -> str:
    return "undefined" if value is None else f"{value * 100:.2f}%"


def print_report(
    gt: Dict[str, Any],
    model: Optional[Dict[str, Any]],
    baseline: Dict[str, Any],
    retention: Dict[str, Optional[float]],
    gate: Dict[str, Any],
) -> None:
    """Prints the report a human reads at Gate 0 and Gate 2. Layout mirrors diagnose_terracing.py
    and diagnose_speckle.py so the three are scannable as a set."""
    print("\n" + "=" * 78)
    print("                     QUANTITATIVE OVERHANG METRICS                     ")
    print("=" * 78)
    print(f"{'':<34}{'GROUND TRUTH':>14}{'MODEL':>14}{'RETENTION':>14}")
    for path, label_text, fmt in RETENTION_KEYS:
        gt_s = fmt.format(float(_dig(gt, path)))
        m_s = fmt.format(float(_dig(model, path))) if model is not None else "—"
        r_s = _fmt_retention(retention.get(".".join(path)))
        print(f"{label_text:<34}{gt_s:>14}{m_s:>14}{r_s:>14}")

    for name, m in (("GROUND TRUTH", gt), ("MODEL", model)):
        if m is None:
            continue
        cav = m["cavity"]
        over_key = f"taller_than_{BAND_SIZE_WARNING_HEIGHT}"
        print(f"\n--- {name}: STRUCTURE ---")
        print(f"  Columns / band cells:        {m['num_columns']:,} / {m['band_cells']:,}"
              f"  (band depth {m['band_height']})")
        print(f"  Solid fraction:              {m['solid_fraction'] * 100:.2f}%")
        print(f"  Overhang cells:              {m['overhang_cells']:,} ({m['overhang_cell_pct']:.4f}% of band)")
        print(f"  Multi-run columns:           {m['multi_run_columns']:,} ({m['multi_run_column_pct']:.4f}%)")
        print(f"  Transitions per column:      mean {m['mean_transitions_per_column']:.4f}, "
              f"max {m['max_transitions_per_column']}")
        for k, v in m["transition_histogram"].items():
            print(f"    {k:<22}: {v:>10,} columns")
        print(f"  --- cavity heights (validates BAND_HEIGHT = {BAND_HEIGHT}, §1.2) ---")
        print(f"    Cavities:                  {cav['count']:,} ({cav['count_per_1000_columns']:.3f} per 1000 cols)")
        print(f"    Mean / max height:         {cav['mean_height']:.3f} / {cav['max_height']} blocks")
        p = cav["percentiles"]
        print(f"    p50 / p90 / p95 / p99:     {p['p50']} / {p['p90']} / {p['p95']} / {p['p99']} blocks")
        print(f"    Taller than {BAND_SIZE_WARNING_HEIGHT}:             {cav[over_key]:,} "
              f"({cav[f'{over_key}_pct']:.2f}% of cavities)")
        print(f"    Clipped at band floor:     {cav['clipped_at_band_floor']:,} "
              f"({cav['clipped_at_band_floor_pct']:.2f}% — heights are lower bounds for these)")
        w = m["walkable"]
        print(f"  --- walkable voids (>={w['min_height']} tall, >={w['min_floor_cells']} floor cells, "
              f"step tol {w['floor_step_tolerance']}, floors joined in {w['floor_space']}) ---")
        print(f"    Walkable voids:            {w['void_count']:,} "
              f"({w['voids_per_1000_columns']:.3f} per 1000 cols)")
        print(f"    Walkable floor cells:      {w['floor_cells']:,} "
              f"({w['floor_cells_per_1000_columns']:.3f} per 1000 cols)")
        print(f"    Largest void floor area:   {w['largest_void_floor_cells']:,} cells")
        if w["floor_space"] == "band_index":
            print("    !! NO HEIGHTFIELD SUPPLIED, so floors were joined in band-index space and")
            print("    !! this void count is an INFLATED UPPER BOUND — measured 24x on a cliff and")
            print("    !! 2.5x on gently-rolling terrain. See compute_overhang_metrics metric 4.")

    is_gate2 = bool(gate["is_gate2_verdict"])
    header = "GATE 2 GO/NO-GO (§7)" if is_gate2 else "GATE 0 DATA CHECK (not a Gate 2 verdict)"
    print(f"\n--- {header} ---")
    print("  Baseline = ground-truth per-band-index marginal thresholded at 0.5, tiled everywhere:")
    print("  the score a model gets for learning nothing except the average band (§3.1).")
    print(f"  Baseline overhang cells:       {baseline['overhang_cell_pct']:.4f}%")
    who = "model" if model is not None else "ground truth"
    cond = gate["conditions"]

    def _mark(ok: Optional[bool]) -> str:
        return "  n/a" if ok is None else (" MET " if ok else " NOT MET")

    print(f"  Conditions, measured on {who}:")
    print(f"    1. multi-run columns   {gate['observed_multi_run_column_pct']:8.4f}% vs baseline "
          f"{gate['baseline_multi_run_column_pct']:.4f}% "
          f"[{gate['delta_pts']:+.4f} pts, need >= {gate['required_margin_pts']:.2f}]"
          f" ->{_mark(cond['multi_run_margin'])}")
    ret = gate["walkable_void_retention"]
    ret_s = "undefined" if ret is None else f"{ret * 100:.2f}%"
    print(f"    2. walkable-void retention {ret_s:>10} "
          f"(need >= {gate['required_void_retention'] * 100:.0f}%)"
          f" ->{_mark(cond['walkable_void_retention'])}")
    print(f"    3. p90 cavity height    {gate['observed_p90_cavity_height']:>3} blocks "
          f"(need >= {gate['required_p90_cavity_height']}; p50 is "
          f"{gate['observed_p50_cavity_height']})"
          f" ->{_mark(cond['p90_cavity_height'])}")
    if gate["baseline_is_degenerate"]:
        print("  NOTE: baseline multi-run rate is 100% — the AVERAGE band itself has multiple solid")
        print("        runs, so condition 1 is no longer informative. Look at it before trusting")
        print("        the verdict either way (see marginal_baseline_band's docstring).")
    if is_gate2:
        failed = [n for n, ok in cond.items() if ok is False]
        why = "all three conditions met" if gate["verdict"] == "PASS" else \
            "failed: " + ", ".join(failed)
        print(f"  VERDICT: {gate['verdict']} — {why}")
        print("  §7 has a FOURTH condition — relief retention must not regress from v1.1.1 — which")
        print("  is measured by the terrain tooling, not this one. A PASS here is three quarters of")
        print("  the gate.")
    else:
        print(f"  DATA CHECK: {gate['verdict']} — this is Gate 0's check that the DATA has structure")
        print("  the average band does not. It is NOT the Gate 2 verdict: condition 2 needs a model")
        print("  to retain anything. Gate 2 reruns these same three lines against model output.")
    print("=" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Overhang Diagnosis Tool — the §2.4 volumetric metrics, on ground truth and "
                    "model output with the same code, with retention and the marginal baseline."
    )
    parser.add_argument("--ground-truth", required=True,
                        help="Band source: a region_*.npz shard, a .npy band array, a directory of "
                             "shards, or a glob. Required for both Gate 0 and Gate 2 — the baseline "
                             "is defined from ground truth.")
    parser.add_argument("--model", default=None,
                        help="Optional model-output band (.npy/.npz). Omit for a Gate 0 "
                             "ground-truth-only run; the retention column then reads '—'.")
    parser.add_argument("--band-key", default=None,
                        help=f"Array name inside .npz shards (default: first of {BAND_KEY_CANDIDATES})")
    parser.add_argument("--max-regions", type=int, default=None,
                        help="Cap on shards read when --ground-truth is a directory/glob")
    parser.add_argument("--layout", default="auto", choices=("auto", "zxi", "izx"),
                        help="Ground-truth band axis order: zxi = [z][x][i] (ground truth), "
                             "izx = [i][z][x] (one chunk of model output). Default auto-detects "
                             "from the band-depth axis, and refuses to guess on a cube.")
    parser.add_argument("--model-layout", default=None, choices=("auto", "zxi", "izx"),
                        help="Axis order for --model, when it differs from --layout. Defaults to "
                             "--layout. Needed because the two sides genuinely can differ: ground "
                             "truth is [z][x][i] and a saved head output is [i][z][x].")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for float bands (probabilities: 0.5; raw logits: 0.0). "
                             "Ignored for integer/bool bands, which are compared != 0.")
    parser.add_argument("--min-walkable-height", type=int, default=DEFAULT_MIN_WALKABLE_HEIGHT,
                        help=f"Cavity height for 'a player can enter it' (default: {DEFAULT_MIN_WALKABLE_HEIGHT})")
    parser.add_argument("--min-walkable-floor", type=int, default=DEFAULT_MIN_WALKABLE_FLOOR,
                        help=f"Floor cells for the same (default: {DEFAULT_MIN_WALKABLE_FLOOR})")
    parser.add_argument("--floor-step-tolerance", type=int, default=DEFAULT_FLOOR_STEP_TOLERANCE,
                        help="Vertical slack in BLOCKS when joining neighbouring columns' floors "
                             f"(default: {DEFAULT_FLOOR_STEP_TOLERANCE}; 0 = exactly level). Any "
                             "value is honoured — see _walkable_components.")
    parser.add_argument("--model-heightfield", default=None,
                        help="Optional .npy heightfield [Z, X] for --model, so the model side's "
                             "walkable voids are joined in world y like the ground truth's. Without "
                             "it BOTH sides are downgraded to band-index space to stay comparable.")
    parser.add_argument("--baseline-margin", type=float, default=DEFAULT_BASELINE_MARGIN_PTS,
                        help="Gate 2 condition 1: percentage points above the marginal baseline "
                             f"required (default: {DEFAULT_BASELINE_MARGIN_PTS})")
    parser.add_argument("--min-void-retention", type=float, default=DEFAULT_MIN_VOID_RETENTION,
                        help="Gate 2 condition 2: fraction of ground truth's walkable voids the "
                             f"model must retain (default: {DEFAULT_MIN_VOID_RETENTION}). "
                             "Condition 3 is --min-walkable-height.")
    parser.add_argument("--checkpoint", default=None,
                        help="Alternative to --model: run inference to produce the model band. "
                             "Requires the occupancy head (Task B1).")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt for --checkpoint inference")
    parser.add_argument("--chunk-span", type=int, default=4, help="Chunk grid size for --checkpoint")
    parser.add_argument("--seed", type=int, default=12345, help="World seed for --checkpoint")
    parser.add_argument("--device", default="cpu", help="PyTorch device for --checkpoint")
    parser.add_argument("--output", default=None,
                        help="Optional output file (.json for the full report, .png for the plot)")

    args = parser.parse_args()

    kw = dict(
        layout=args.layout,
        threshold=args.threshold,
        min_walkable_height=args.min_walkable_height,
        min_walkable_floor=args.min_walkable_floor,
        floor_step_tolerance=args.floor_step_tolerance,
    )

    print("=== Overhang Diagnosis Script ===")
    try:
        gt_paths = resolve_band_paths(args.ground_truth, args.max_regions)
    except (FileNotFoundError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    def _load_side(paths: List[str], layout: str, use_heightfield: bool = True,
                   override_hf: Optional[np.ndarray] = None):
        """Loads and measures one side. Returns (per-region metrics, sources, any_fallback,
        all_have_heightfield)."""
        metrics_out: List[Dict[str, Any]] = []
        srcs: List[str] = []
        fallback = False
        have_hf = True
        for path in paths:
            loaded = load_band(path, args.band_key)
            srcs.append(loaded.source)
            fallback = fallback or loaded.is_heightfield_fallback
            hf = override_hf if override_hf is not None else loaded.heightfield
            have_hf = have_hf and hf is not None
            metrics_out.append(compute_overhang_metrics(
                loaded.band, heightfield=hf if use_heightfield else None,
                **dict(kw, layout=layout)))
        return metrics_out, srcs, fallback, have_hf

    def _print_fallback_banner(where: str) -> None:
        print("  " + "!" * 72)
        print(f"  !! NO BAND ARRAY IN AT LEAST ONE {where} SHARD. That band was DERIVED from the")
        print("  !! heightfield, i.e. it is the degenerate v1.1.1 volume: its metrics are 0 by")
        print("  !! construction, and that is a fact about the DATA, not about the terrain.")
        if where == "MODEL":
            print("  !! A zero here reads as 'the model learned nothing'. It is not that. This is")
            print("  !! docs/phase4-plan.md §5.1's silent downgrade arriving through the diagnostic.")
        print("  !! Regenerate shards with the Phase 4 band (docs/phase4-plan.md §2.1) first.")
        print("  " + "!" * 72)

    t0 = time.perf_counter()
    try:
        per_region, sources, gt_fallback, gt_has_hf = _load_side(gt_paths, args.layout)
    except (KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    gt_elapsed = time.perf_counter() - t0

    print(f"Ground truth: {args.ground_truth}  ({len(gt_paths)} region(s))")
    print(f"  source:     {sources[0]}" + (f"  [+{len(sources) - 1} more]" if len(sources) > 1 else ""))
    if gt_fallback:
        _print_fallback_banner("GROUND TRUTH")
    print(f"Computed in {gt_elapsed:.3f}s over {sum(m['num_columns'] for m in per_region):,} columns")

    model_regions: Optional[List[Dict[str, Any]]] = None
    model_source = None
    model_has_hf = False
    try:
        if args.model is not None:
            m_paths = resolve_band_paths(args.model)
            m_layout = args.model_layout if args.model_layout is not None else args.layout
            override = np.load(args.model_heightfield) if args.model_heightfield else None
            model_regions, m_sources, m_fallback, model_has_hf = _load_side(
                m_paths, m_layout, override_hf=override)
            model_source = m_sources[0] + (f"  [+{len(m_sources) - 1} more]" if len(m_sources) > 1 else "")
            if override is not None:
                model_source += f"  +heightfield {os.path.basename(args.model_heightfield)}"
            if m_fallback:
                print(f"Model:        {model_source}")
                _print_fallback_banner("MODEL")
        elif args.checkpoint is not None:
            if not os.path.isfile(args.checkpoint):
                print(f"ERROR: Checkpoint file not found at {args.checkpoint}", file=sys.stderr)
                return 1
            from mc_imagine_model.export.export_onnx import load_imagine_net
            net = load_imagine_net(args.checkpoint)
            net.to(args.device)
            print(f"Inference:    prompt='{args.prompt}' span={args.chunk_span} seed={args.seed}")
            band, m_hf = render_band_area(net, args.prompt, args.seed, args.chunk_span,
                                          device=args.device)
            model_has_hf = m_hf is not None
            # `render_band_area` returns the internal [Z, X, I] layout, so layout is KNOWN here.
            # Re-sniffing it was a real bug: a 4x4 tile of 16x16 chunks is [64, 64, 64], which
            # auto-detection correctly calls ambiguous, so `--checkpoint` at its own default
            # --chunk-span died with an uncaught ValueError before reaching any metric.
            model_regions = [compute_overhang_metrics(band, heightfield=m_hf,
                                                      **dict(kw, layout="zxi"))]
            model_source = f"{os.path.basename(args.checkpoint)} inference"
    except (KeyError, ValueError, FileNotFoundError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # --- floor-space symmetry: never compare a world-y number against a band-index one -------
    # The two sides' walkable-void counts differ by 2.5-24x purely on which space the floors were
    # joined in (24x on a cliff), so a paired run measuring ground truth in world y and the model
    # in band index would report a retention ratio that is mostly an artefact of the model file not
    # carrying a heightfield. Both sides are therefore recomputed in the WEAKER space when they disagree, with
    # the reason stated, rather than silently mixed or silently refused.
    mixed = model_regions is not None and gt_has_hf and not model_has_hf
    if mixed:
        print("\n  " + "!" * 72)
        print("  !! MODEL OUTPUT HAS NO HEIGHTFIELD, so its floors can only be joined in")
        print("  !! band-index space. Ground truth is being RE-MEASURED the same way so the")
        print("  !! retention ratio is apples-to-apples. Both void counts are now inflated upper")
        print("  !! bounds (24x on a cliff fixture, 2.5x on rolling terrain). Pass")
        print("  !! --model-heightfield to get the world-y numbers §9 should record.")
        print("  " + "!" * 72)
        per_region, _, _, _ = _load_side(gt_paths, args.layout, use_heightfield=False)

    gt_metrics = aggregate_overhang_metrics(per_region)
    model_metrics = aggregate_overhang_metrics(model_regions) if model_regions else None
    if model_metrics is not None:
        print(f"Model:        {model_source}")
    else:
        print("Model:        (none — Gate 0 ground-truth-only run)")

    # The baseline is ALWAYS built from ground truth: it is what a model that learned only the
    # data's average band would say. Building it from model output would compare the model to
    # itself, which passes trivially.
    baseline_band = marginal_baseline_from_profile(
        gt_metrics["solid_fraction_by_band_index"], tuple(per_region[0]["grid_shape"])
    )
    # `marginal_baseline_from_profile` always returns the internal [Z, X, I] layout, so the
    # baseline is measured with layout="zxi" REGARDLESS of --layout. Passing the user's --layout
    # through here would transpose the baseline whenever the inputs are model-layout bands, and a
    # transposed baseline is still a perfectly plausible-looking number.
    baseline_kw = dict(kw, layout="zxi")
    baseline_metrics = compute_overhang_metrics(baseline_band, **baseline_kw)

    retention = compute_retention(gt_metrics, model_metrics)
    gate = evaluate_baseline_gate(
        model_metrics if model_metrics is not None else gt_metrics,
        baseline_metrics,
        args.baseline_margin,
        gt_metrics=gt_metrics if model_metrics is not None else None,
        min_void_retention=args.min_void_retention,
    )
    print_report(gt_metrics, model_metrics, baseline_metrics, retention, gate)

    if args.output:
        report = {
            "ground_truth_path": args.ground_truth,
            "ground_truth_sources": sources,
            "ground_truth_is_heightfield_fallback": gt_fallback,
            "model_path": args.model or args.checkpoint,
            "model_source": model_source,
            "regions": len(gt_paths),
            "settings": dict(kw, model_layout=args.model_layout,
                             min_void_retention=args.min_void_retention),
            "floor_space": gt_metrics["walkable"]["floor_space"],
            "floor_space_downgraded_to_match_model": mixed,
            "ground_truth": gt_metrics,
            "model": model_metrics,
            "marginal_baseline": baseline_metrics,
            "retention": retention,
            "gate": gate,
            "seconds": gt_elapsed,
        }
        if args.output.endswith(".png"):
            save_diagnostic_plot(gt_metrics, model_metrics, baseline_metrics, args.output)
            print(f"\nSaved diagnostic plot image to: {args.output}")
        else:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nSaved diagnostic metrics JSON report to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
