#!/usr/bin/env python3
"""
Loss-mass diagnosis (Mc-Imagine Phase 4.2 — docs/phase4.2-plan.md §3.1).

Answers "where does the occupancy gradient actually go?" on real ground truth, without touching a
model. Every acceptance criterion in docs/phase4.2-plan.md §4 is expressed in this script's output:
multi-run column %, `OccupancyLoss`'s weight mass on overhang-bearing columns and on the extra-
transition cells that actually constitute an overhang, and each loss term's magnitude and share of
the total at a perfect heightfield-only prediction, for a range of `overhang_weight` settings.

Not a test — a measurement tool an operator runs before and after touching losses.py, the same way
`diagnose_overhangs.py` is run before and after touching the generator. See
`model/tests/test_occupancy_capacity.py` for the corresponding capacity (architecture) check, which
IS a test.

Usage:
    PYTHONPATH=model/src python model/src/mc_imagine_model/scripts/loss_mass.py
    PYTHONPATH=model/src python model/src/mc_imagine_model/scripts/loss_mass.py \\
        --data-dir model/data --num-regions 24 --num-chunks 512 \\
        --overhang-weights 0.005,0.05,0.5,1.0
"""

import argparse
import glob
import os

import numpy as np
import torch

from mc_imagine_model.training.losses import OccupancyLoss, OverhangLoss, VerticalCoherenceLoss

# Same layout as preflight.py: .../mc_imagine_model/scripts/loss_mass.py -> .../model
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../mc_imagine_model
MODEL_ROOT = os.path.dirname(os.path.dirname(_PKG_DIR))                  # .../model
DEFAULT_DATA_DIR = os.path.join(MODEL_ROOT, "data")


def load_chunks(data_dir: str, num_regions: int, num_chunks: int) -> torch.Tensor:
    """Loads real shards and slices them into [N, 64, 16, 16] (y, z, x) chunks.

    Mirrors the exact windowing the diagnosis used (`docs/phase4.2-plan.md` §3.1's provenance):
    non-overlapping 16x16 columns strided by 64 across each region's band.
    """
    paths = sorted(glob.glob(os.path.join(data_dir, "region_*.npz")))[:num_regions]
    if not paths:
        raise FileNotFoundError(f"no region_*.npz shards under {data_dir}")
    chunks = []
    for p in paths:
        with np.load(p) as z:
            b = z["band"]  # [CANVAS, CANVAS, 64]
            c = b.shape[0]
            for zz in range(0, c - 15, 64):
                for xx in range(0, c - 15, 64):
                    chunks.append(np.transpose(b[zz:zz + 16, xx:xx + 16, :], (2, 0, 1)))
    if not chunks:
        raise RuntimeError(f"loaded {len(paths)} shards from {data_dir} but extracted zero chunks")
    return torch.from_numpy(np.stack(chunks[:num_chunks])).float()


def build_matched_rate_speckle(band: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Heightfield-only, plus one random 1-block cavity per column at `band`'s own multi-run column
    rate — docs/phase4.3-plan.md §4.2's "speckle at matched multi-run rate", used to give
    `VerticalCoherenceLoss` a report point where it is actually nonzero (it is exactly zero at a
    heightfield-only prediction by construction: excess is only ever charged over-target, and
    heightfield predicts no cavities at all — see `test_objective_ranking.py`'s docstring).
    """
    n, h, z, x = band.shape
    dy = torch.zeros_like(band)
    dy[:, :-1] = (band[:, 1:] - band[:, :-1]).abs()
    n_trans = dy.sum(dim=1)
    multi_run_pct = float((n_trans >= 2).float().mean()) * 100.0

    hf = torch.zeros_like(band)
    hf[:, :62] = 1.0  # BAND_BOTTOM_OFFSET=-61 -> solid iff i<=61 (spec_constants), matches heightfield_band

    rng = np.random.default_rng(seed)
    n_cols = n * z * x
    flat = hf.permute(0, 2, 3, 1).reshape(n_cols, h).clone()
    n_speckled = int(round(multi_run_pct / 100.0 * n_cols))
    cols = rng.choice(n_cols, size=min(n_speckled, n_cols), replace=False)
    carve_i = rng.integers(low=5, high=55, size=cols.size)
    flat[cols, carve_i] = 0.0
    return flat.reshape(n, z, x, h).permute(0, 3, 1, 2).contiguous()


def report(band: torch.Tensor, overhang_weights) -> None:
    print(f"chunks: {tuple(band.shape)}   solid fraction: {band.mean():.4f}")

    # --- transition structure --------------------------------------------------------------
    dy = torch.zeros_like(band)
    dy[:, :-1] = (band[:, 1:] - band[:, :-1]).abs()
    n_trans = dy.sum(dim=1)  # transitions per column
    surface_like = (n_trans <= 1).float().mean()
    print(f"columns with <=1 transition (pure heightfield): {surface_like * 100:.3f}%")
    print(f"columns with >=2 transitions (overhang-bearing): {(1 - surface_like) * 100:.3f}%")

    # --- OccupancyLoss weight mass ----------------------------------------------------------
    # Uses the production weight/mask logic directly (OccupancyLoss.compute_weight /
    # transition_masks) rather than a hand-reimplemented copy, so this can never silently drift
    # from what the training loop actually weights.
    occupancy_loss = OccupancyLoss()
    w = occupancy_loss.compute_weight(band)
    _, is_extra = occupancy_loss.transition_masks(band)
    total_w = w.sum()
    multi = (n_trans >= 2).unsqueeze(1).expand_as(w)
    print(f"\nOccupancyLoss weight mass on overhang-bearing columns: "
          f"{(w * multi).sum() / total_w * 100:.3f}%")
    print(f"   ...of which the *extra* transition cells themselves: "
          f"{(w * is_extra.float()).sum() / total_w * 100:.4f}%")

    # --- term magnitudes at a heightfield-only optimum --------------------------------------
    # Best heightfield predictor: solid iff below the topmost solid cell.
    idx = torch.arange(64).view(1, 64, 1, 1).expand_as(band)
    top = torch.where(band > 0, idx, torch.full_like(idx, -1)).amax(dim=1, keepdim=True)
    hf = (idx <= top).float()  # the pure-heightfield prediction
    logits = torch.logit(hf.clamp(1e-4, 1 - 1e-4))

    occ = occupancy_loss(logits, band)
    ovh = OverhangLoss()(logits, band)
    print(f"\nAt a perfect *heightfield-only* prediction:")
    print(f"  occupancy term            = {occ:.6f}   x weight 1.0")
    print(f"  overhang term             = {ovh:.6f}")
    for wt in overhang_weights:
        share = ovh * wt / (occ + ovh * wt) * 100
        print(f"    at overhang_weight={wt:<8}: share = {share:6.2f}%")

    # --- VerticalCoherenceLoss (docs/phase4.3-plan.md §4.2, B.2) -----------------------------
    # Zero at heightfield-only by construction (it only ever charges an EXCESS of predicted
    # 1-tall cavities over the target's own count, and heightfield predicts none), so its share is
    # reported at a matched-rate speckle prediction instead — the case it exists to penalize.
    speckle_logits = torch.logit(build_matched_rate_speckle(band).clamp(1e-4, 1 - 1e-4))
    occ_speckle = occupancy_loss(speckle_logits, band)
    ovh_speckle = OverhangLoss()(speckle_logits, band)
    vert_speckle = VerticalCoherenceLoss()(speckle_logits, band)
    print(f"\nAt a matched-rate *1-block-speckle* prediction:")
    print(f"  occupancy term            = {occ_speckle:.6f}   x weight 1.0")
    print(f"  overhang term             = {ovh_speckle:.6f}   x weight 4.0 (shipped)")
    print(f"  vertical-coherence term   = {vert_speckle:.6f}   (docs/phase4.3-plan.md §4.2 B.2)")
    fixed = occ_speckle + 4.0 * ovh_speckle
    for vw in (0.0, 150.0, 240.0, 300.0, 500.0, 1000.0):
        share = vert_speckle * vw / (fixed + vert_speckle * vw) * 100
        print(f"    at vertical_weight={vw:<8}: share = {share:6.2f}%")
    print(
        "  Note (operator decision 2026-08-08, tests/test_objective_ranking.py): 300 closes "
        "`real < displaced < speckle` with headroom over the measured ~240 threshold; the stricter "
        "\"displaced closer to real than to speckle\" needs ~2826 and is not shipped — see that "
        "test's module docstring for the full finding."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report where OccupancyLoss/OverhangLoss's weight mass and gradient actually go, "
                    "on real ground truth — docs/phase4.2-plan.md §3.1.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help=f"Directory of region_*.npz shards (default {DEFAULT_DATA_DIR})")
    parser.add_argument("--num-regions", type=int, default=24,
                        help="Shards to read before slicing into chunks (default 24)")
    parser.add_argument("--num-chunks", type=int, default=512,
                        help="Chunks to keep after slicing (default 512)")
    parser.add_argument("--overhang-weights", default="0.005,0.05,0.5,1.0",
                        help="Comma-separated overhang_weight values to report the loss share at")
    args = parser.parse_args()

    weights = [float(w) for w in args.overhang_weights.split(",") if w.strip()]
    band = load_chunks(args.data_dir, args.num_regions, args.num_chunks)
    report(band, weights)


if __name__ == "__main__":
    main()
