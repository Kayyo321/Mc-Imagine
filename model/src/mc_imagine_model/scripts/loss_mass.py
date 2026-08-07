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

from mc_imagine_model.training.losses import OccupancyLoss, OverhangLoss

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
