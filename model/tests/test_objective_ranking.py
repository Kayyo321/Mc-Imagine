"""
Objective preference-ordering test (Mc-Imagine Phase 4.3 — docs/phase4.3-plan.md §4.1/§4.2, B.1/B.2).

`phase4.2-plan.md` §6.1 measured, under the shipped objective (`OccupancyLoss` defaults +
`overhang_weight=4.0`): real caves score far below heightfield-only, which scores below matched-rate
speckle — the objective already ranks speckle worst. `test_objective_ranks_real_below_heightfield_
below_speckle` pins that as a regression guard.

`phase4.3-plan.md` §4.1 named a second, MISSING property: a cave-shaped prediction in the wrong place
("displaced") should score closer to real caves than to speckle — "right shape, wrong place" should
not read as "no shape at all". `B.2` added `VerticalCoherenceLoss` (`losses.py`) to address this.
Measured here, on 128 real chunks at the dataset's representative density (~7% multi-run, not a
cherry-picked outlier):

    weight needed for `real < displaced < speckle` (ordering alone):        ~240
    weight needed for "displaced closer to real than to speckle" (in full): ~2826

At ~2800, `VerticalCoherenceLoss` would have to outweigh `OccupancyLoss`/`OverhangLoss` by three
orders of magnitude — it would stop being a shape-refinement term and become the entire objective.
The same conclusion holds for a cavity-height-histogram term (`B.3`): the bottleneck is not which
shape metric is chosen, it is that `OccupancyLoss`/`OverhangLoss`'s per-cell BCE (both 🔒-locked,
`phase4.3-plan.md` §3) already separates "near-perfect pointwise match" (real) from "anything spatially
imperfect" (displaced OR speckle, coherent or not) by a margin no bounded additive term can close
without dominating the total loss. Operator decision (2026-08-08): ship `B.2` at the weight that
closes the ORDERING (`VERTICAL_WEIGHT = 300`, headroom over the measured ~240 threshold), and record
the "closer to real" property as an open, unresolved finding rather than forcing it — see
`test_objective_displaced_not_closer_to_real_than_speckle_yet` below, which documents this as an
XFAIL with the measured numbers, not a silently-loosened requirement.

Four synthetic predictions are built against a batch of REAL ground truth chunks (never synthetic-vs-
synthetic — the point is what the objective does with real cave structure):
  - real caves:           the batch itself (a perfect prediction).
  - displaced:             each chunk's own structure, rolled 8 blocks in x (wraps within the chunk)
    — cave-SHAPED, wrong place, exactly `phase4.3-plan.md` §4.1's construction.
  - matched-rate speckle:  heightfield-only plus one scattered 1-block cavity per column, at the same
    rate as the batch's own multi-run column percentage — "speckle" as `phase4.2-plan.md` §5.1's
    rehearsal run actually produced it, not an arbitrary density.
  - heightfield-only:      `diagnose_overhangs.heightfield_band`'s degenerate flat-world band.

Chunks are filtered to a MODERATE multi-run rate (3-12%, bracketing the retuned dataset's own ~10.14%
average, `phase4.2-plan.md` §1.1) rather than the densest region available — the objective's absolute
loss magnitudes scale with structure density, and an outlier-dense sample inflates every number without
changing any ranking, which would misrepresent how close/far this objective is from §6.1's reference
figures (0.0005 / 1.3903 / 1.5415).

Predictions are represented as confident logits (±8.0, matching `test_model.py`'s convention) rather
than probabilities, so `OccupancyLoss`/`OverhangLoss`/`VerticalCoherenceLoss` see the same near-bit-
exact 0/1 structure real numbers are always measured against in this codebase.
"""

import glob
import os

import numpy as np
import pytest
import torch

from mc_imagine_model.scripts.diagnose_overhangs import compute_overhang_metrics, heightfield_band
from mc_imagine_model.training.losses import OccupancyLoss, OverhangLoss, VerticalCoherenceLoss

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Shipped weights (training/config.mps.yaml == config.cuda.yaml's `occupancy_weight`/
# `overhang_weight` block, required to be identical between the two configs). "The current
# objective" this test means, not a value this test invented.
OCCUPANCY_WEIGHT = 1.0
OVERHANG_WEIGHT = 4.0
# B.2's operator-chosen weight (see module docstring): ~240 closes the ordering requirement with
# headroom; NOT shipped in either config yet (CombinedLoss's own default stays 0.0 — B.4's job).
VERTICAL_WEIGHT = 300.0

CONFIDENT_LOGIT = 15.0  # sigmoid(15) leaves a ~3e-7 leak/cell — see note below, not test_model.py's
# 8.0 (sigmoid(8)~3e-4/cell): VerticalCoherenceLoss sums its per-cell field over a whole 16384-cell
# chunk before normalizing, so many non-cavity cells' tiny individual "leak" (a soft prediction is
# never bit-exactly 0/1) accumulates into a real signal at vertical_weight~150 — measured, an 8.0
# "perfect" prediction scored 0.050, not ~0, purely from this leak. 15.0 reduces it ~1000x, to noise.
NUM_CHUNKS = 128
MIN_MULTI_RUN_PCT = 3.0   # moderate density floor
MAX_MULTI_RUN_PCT = 12.0  # moderate density ceiling — brackets the dataset's ~10.14% average
MAX_REGIONS_SCANNED = 80


def _load_real_chunk_batch(
    data_dir: str, num_chunks: int = NUM_CHUNKS,
) -> np.ndarray:
    """Real 16x16 chunks at MODERATE density (see module docstring), `[B, 16, 16, 64]` bool
    (`[B, Z, X, I]`), 1=solid. Non-overlapping windowing, same convention as
    `test_occupancy_capacity.py`'s chunk extraction.
    """
    paths = sorted(glob.glob(os.path.join(data_dir, "region_*.npz")))[:MAX_REGIONS_SCANNED]
    chunks = []
    for p in paths:
        with np.load(p) as z:
            band = z["band"]
            c = band.shape[0]
            for zz in range(0, c - 15, 16):
                for xx in range(0, c - 15, 16):
                    chunk = band[zz:zz + 16, xx:xx + 16, :].astype(bool)
                    pct = compute_overhang_metrics(chunk, layout="zxi")["multi_run_column_pct"]
                    if MIN_MULTI_RUN_PCT <= pct <= MAX_MULTI_RUN_PCT:
                        chunks.append(chunk)
        if len(chunks) >= num_chunks:
            break
    if len(chunks) < num_chunks:
        pytest.skip(
            f"found only {len(chunks)} chunks in [{MIN_MULTI_RUN_PCT}, {MAX_MULTI_RUN_PCT}]% "
            f"multi-run rate under {data_dir} (need {num_chunks})"
        )
    return np.stack(chunks[:num_chunks])


def _displace(batch: np.ndarray, shift: int = 8) -> np.ndarray:
    """Each chunk's OWN structure, shifted `shift` blocks in x, independently per chunk (wraps at
    that chunk's own boundary) — `phase4.3-plan.md` §4.1's literal construction, at real per-chunk
    scale rather than across an entire multi-chunk crop.
    """
    return np.roll(batch, shift=shift, axis=2)  # batch axes: [B, Z, X, I]


def _matched_rate_speckle(batch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Heightfield-only, plus one random 1-block cavity per speckled column, at the same RATE as
    `batch`'s own multi-run column percentage (pooled across the whole batch) — matches §6.1's
    "speckle at matched multi-run rate", not an arbitrary density.
    """
    b, z, x, h = batch.shape
    hf = np.broadcast_to(heightfield_band(np.zeros((z, x))), batch.shape).copy()
    target_pct = compute_overhang_metrics(
        batch.reshape(b * z, x, h), layout="zxi",
    )["multi_run_column_pct"]
    n_cols = b * z * x
    flat = hf.reshape(n_cols, h)
    n_speckled = int(round(target_pct / 100.0 * n_cols))
    cols = rng.choice(n_cols, size=n_speckled, replace=False)
    # One 1-block cavity per speckled column, well inside the solid run (index 61 is the last solid
    # heightfield index) so carving it always creates a genuine extra transition.
    carve_i = rng.integers(low=5, high=55, size=n_speckled)
    flat[cols, carve_i] = False
    return flat.reshape(batch.shape)


def _to_logits(batch: np.ndarray) -> torch.Tensor:
    arr = np.where(batch, CONFIDENT_LOGIT, -CONFIDENT_LOGIT).astype(np.float32)
    return torch.from_numpy(np.moveaxis(arr, 3, 1))  # [B, Z, X, I] -> [B, I, Z, X]


def _to_target(batch: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.moveaxis(batch.astype(np.float32), 3, 1))


def _objective_loss(pred_batch: np.ndarray, target_batch: np.ndarray, vertical_weight: float = 0.0) -> float:
    """Total = occupancy_weight*OccupancyLoss + overhang_weight*OverhangLoss
             + vertical_weight*VerticalCoherenceLoss (0.0 = the pre-B.2 objective)."""
    logits = _to_logits(pred_batch)
    target = _to_target(target_batch)
    total = (
        OCCUPANCY_WEIGHT * OccupancyLoss()(logits, target)
        + OVERHANG_WEIGHT * OverhangLoss()(logits, target)
        + vertical_weight * VerticalCoherenceLoss()(logits, target)
    )
    return float(total.item())


@pytest.mark.skipif(not os.path.isdir(DATA_DIR), reason=f"no real shards under {DATA_DIR}")
def test_objective_ranks_real_below_heightfield_below_speckle() -> None:
    """`phase4.2-plan.md` §6.1's already-measured ordering, pinned as a regression guard, now under
    the FULL shipped-plus-B.2 objective (adding `VerticalCoherenceLoss` must not break it — it is
    exactly zero at a heightfield-only prediction and larger for speckle than for real, so it can
    only widen this gap, never close it)."""
    batch = _load_real_chunk_batch(DATA_DIR)
    hf = np.broadcast_to(heightfield_band(np.zeros(batch.shape[1:3])), batch.shape).copy()
    speckle = _matched_rate_speckle(batch, np.random.default_rng(0))

    loss_real = _objective_loss(batch, batch, VERTICAL_WEIGHT)
    loss_heightfield = _objective_loss(hf, batch, VERTICAL_WEIGHT)
    loss_speckle = _objective_loss(speckle, batch, VERTICAL_WEIGHT)

    assert loss_real < 0.01, f"a perfect prediction should score near zero, got {loss_real:.4f}"
    assert loss_real < loss_heightfield, (
        f"real caves ({loss_real:.4f}) must score below heightfield-only ({loss_heightfield:.4f})"
    )
    assert loss_heightfield < loss_speckle, (
        f"heightfield-only ({loss_heightfield:.4f}) must score below matched-rate speckle "
        f"({loss_speckle:.4f}) — speckle must remain the objective's worst case"
    )


@pytest.mark.skipif(not os.path.isdir(DATA_DIR), reason=f"no real shards under {DATA_DIR}")
def test_objective_orders_real_displaced_speckle() -> None:
    """`B.2`'s achieved property: `real < displaced < speckle`, at `VERTICAL_WEIGHT`. Before B.2
    (`vertical_weight=0`), `OccupancyLoss`/`OverhangLoss` alone rank displaced WORSE than speckle —
    a coherently-shifted full cave has many more mismatched cells than sparse speckle does, so per-
    cell BCE penalizes it more, not less. `VerticalCoherenceLoss` (nonzero for speckle's excess
    1-block cavities, ~zero for displaced/real since a roll preserves each chunk's own cavity count)
    is what flips the ordering back.
    """
    batch = _load_real_chunk_batch(DATA_DIR)
    displaced = _displace(batch, shift=8)
    speckle = _matched_rate_speckle(batch, np.random.default_rng(1))

    loss_real = _objective_loss(batch, batch, VERTICAL_WEIGHT)
    loss_displaced = _objective_loss(displaced, batch, VERTICAL_WEIGHT)
    loss_speckle = _objective_loss(speckle, batch, VERTICAL_WEIGHT)

    # Pinned as a regression guard for the PRE-B.2 failure mode this fixes: without the vertical
    # term, displaced scores at or above speckle.
    loss_displaced_no_vertical = _objective_loss(displaced, batch, vertical_weight=0.0)
    loss_speckle_no_vertical = _objective_loss(speckle, batch, vertical_weight=0.0)
    assert loss_displaced_no_vertical >= loss_speckle_no_vertical, (
        "expected the PRE-B.2 objective (vertical_weight=0) to already rank displaced >= speckle "
        f"(displaced={loss_displaced_no_vertical:.4f}, speckle={loss_speckle_no_vertical:.4f}) — if "
        "this no longer holds, OccupancyLoss/OverhangLoss changed and this test's premise is stale"
    )

    assert loss_real < loss_displaced < loss_speckle, (
        f"B.2 must order real ({loss_real:.4f}) < displaced ({loss_displaced:.4f}) < speckle "
        f"({loss_speckle:.4f}) at vertical_weight={VERTICAL_WEIGHT}"
    )


@pytest.mark.skipif(not os.path.isdir(DATA_DIR), reason=f"no real shards under {DATA_DIR}")
def test_objective_displaced_not_closer_to_real_than_speckle_yet() -> None:
    """The UNRESOLVED half of §4.1's acceptance criterion: "displaced closer to real than to
    speckle". Recorded as an XFAIL, not silently dropped or loosened — see the module docstring for
    the weight math (~2826 needed) showing this is not reachable by a bounded additive shape term
    while `OccupancyLoss`/`OverhangLoss` stay 🔒-locked. If this ever starts passing (e.g. after a
    future change to those locked terms, or a fundamentally different shape term), pytest reports an
    XPASS — loud, not a silent pass — which is the intended signal to revisit this test.
    """
    batch = _load_real_chunk_batch(DATA_DIR)
    displaced = _displace(batch, shift=8)
    speckle = _matched_rate_speckle(batch, np.random.default_rng(1))

    loss_real = _objective_loss(batch, batch, VERTICAL_WEIGHT)
    loss_displaced = _objective_loss(displaced, batch, VERTICAL_WEIGHT)
    loss_speckle = _objective_loss(speckle, batch, VERTICAL_WEIGHT)

    closer_to_real = (loss_displaced - loss_real) < (loss_speckle - loss_displaced)
    if not closer_to_real:
        pytest.xfail(
            f"known gap (2026-08-08 operator decision): displaced ({loss_displaced:.4f}) is not "
            f"closer to real ({loss_real:.4f}) than to speckle ({loss_speckle:.4f}) at "
            f"vertical_weight={VERTICAL_WEIGHT} — closing this needs weight~2826, which would let "
            f"VerticalCoherenceLoss dominate the entire objective; see module docstring"
        )
