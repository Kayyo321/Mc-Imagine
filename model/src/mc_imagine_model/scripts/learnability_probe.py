#!/usr/bin/env python3
"""
Single-region learnability probe (Mc-Imagine Phase 4.3 — docs/phase4.3-plan.md §3.1, A.1).

Answers the question `phase4.2-plan.md` §6.1 left open: can this architecture, conditioned the way
it actually is at inference time, LEARN a coordinate -> 3D-cave mapping at all? Trains the full
`ImagineNet` (every module: `PromptEncoder`, `SeedEncoder`, `CoordinateEncoder`, the conv stack,
`Occupancy3DHead` — nothing frozen, nothing swapped for random features) on every chunk of ONE real
carved region, through the real conditioning path (real prompt tokens from that region's own
caption, its real world seed, real global chunk coordinates), with every loss term except occupancy
zeroed out so nothing competes with it. Reports the sub-surface Pearson r on that SAME region (train
r — memorisation is the point here, not generalisation) plus the predicted vs. target cavity-height
distribution.

WHY THIS AND NOT `test_occupancy_capacity.py`. That test feeds `Occupancy3DHead` random FROZEN
features and overfits 8 chunks — it proves the head can REPRESENT structure. It says nothing about
whether the network can LEARN placement from (prompt, seed, coordinates), which is `phase4.2-plan.md`
§6.1's actual open problem: r = 0.031 on a real rehearsal run, sub-chance. This probe removes every
other variable (dataset scale, other loss terms competing for gradient, held-out generalisation) and
asks the narrowest possible version of that question: with all of SGD's budget pointed at ONE
region's occupancy loss and nothing else, does placement r rise at all?

THE FORK THIS DECIDES (`phase4.3-plan.md` §3.1):
    r >= 0.7  -> placement is learnable; Phase 4.2's failure was budget/scale, not design. -> §3.2.
    r <  0.3  -> the mapping is not learnable even when memorising one region. No amount of data or
                 steps fixes that. -> §3.3 (architectural localisation probes).
    0.3<=r<0.7-> partial; treat as the r<0.3 branch but also run §3.2 in parallel.

Placement is measured EXCLUSIVELY through `McImagineDataset` slicing, never by hand-slicing `band`
directly — `phase4.2-plan.md` §6.1's note: hand-slicing from canvas index 0 shifts the target 4
blocks (`HALO_MARGIN`) against the coordinates fed to the model, which corrupted the first pass at
this exact measurement (reported r=0.013 instead of the correct 0.031).

Usage:
    PYTHONPATH=model/src python model/src/mc_imagine_model/scripts/learnability_probe.py
    PYTHONPATH=model/src python model/src/mc_imagine_model/scripts/learnability_probe.py \\
        --region model/data/region_00366.npz --max-steps 4000 --batch-size 32 --device mps \\
        --output learnability_probe_report.json
"""

import argparse
import glob
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from mc_imagine_model.data.dataset import CHUNKS_PER_REGION_SIDE, McImagineDataset
from mc_imagine_model.data.world_generator import REGION_BLOCKS
from mc_imagine_model.model.imagine_net import ImagineNet
from mc_imagine_model.scripts.diagnose_overhangs import compute_overhang_metrics
from mc_imagine_model.spec_constants import BAND_HEIGHT
from mc_imagine_model.training.losses import CombinedLoss
from mc_imagine_model.training.train import (
    build_targets,
    make_lr_lambda,
    move_batch,
    resolve_device,
    seed_everything,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger.setLevel(logging.INFO)

# .../mc_imagine_model/scripts/learnability_probe.py -> .../model (same layout as loss_mass.py)
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ROOT = os.path.dirname(os.path.dirname(_PKG_DIR))
DEFAULT_DATA_DIR = os.path.join(MODEL_ROOT, "data")

# Chosen by scanning all 400 regions in the retuned dataset (`phase4.2-plan.md` §1.1) for the
# highest ground-truth multi-run column rate via `compute_overhang_metrics` — the region with the
# most 3D structure to memorise, so a null result here cannot be blamed on "the region was flat".
# Measured: 34.56% multi-run columns, mean cavity height 10.37, carve_strength 1.12. Override with
# --region to pick a different one, or pass --scan-regions to redo the search.
DEFAULT_REGION = os.path.join(DEFAULT_DATA_DIR, "region_00366.npz")

# `model:` block mirrored byte-for-byte from `training/config.mps.yaml` (also identical in
# config.cuda.yaml — the two must never diverge, see that file's header comment). This probe must
# train the SAME architecture the real runs use, not a shrunk stand-in, or a positive result here
# would not transfer and a negative one would be uninformative.
PROBE_MODEL_CONFIG: Dict[str, Any] = {
    "checkpoint_dir": None,
    "halo": 9,
    "patch": 16,
    "conv_channels": 192,
    "num_conv_layers": 6,
    "num_conv3d_layers": 3,
    "fusion_hidden": 512,
    "fusion_out": 256,
    "seed_freqs": 32,
    "seed_embed_dim": 64,
    "num_profiles": 8,
    "num_biomes": 12,
}

REGION_SIZE = CHUNKS_PER_REGION_SIDE * 16  # 512, matches REGION_BLOCKS


def scan_for_most_carved_region(data_dir: str, max_regions: Optional[int] = None) -> str:
    """Picks the `region_*.npz` shard with the highest ground-truth multi-run column rate.

    Reads the raw `band` array directly (NOT through `McImagineDataset`) — this is legitimate here
    because it is only choosing WHICH region file to train on, not measuring model/target placement
    agreement (the operation the module docstring's halo warning applies to).
    """
    paths = sorted(glob.glob(os.path.join(data_dir, "region_*.npz")))
    if max_regions is not None:
        paths = paths[:max_regions]
    if not paths:
        raise FileNotFoundError(f"no region_*.npz shards found in {data_dir}")
    best_path, best_pct = None, -1.0
    for p in paths:
        with np.load(p) as z:
            pct = compute_overhang_metrics(z["band"], layout="zxi")["multi_run_column_pct"]
        if pct > best_pct:
            best_path, best_pct = p, pct
    logger.info("Scanned %d regions; most-carved is %s (%.2f%% multi-run columns).",
                len(paths), best_path, best_pct)
    return best_path


def train_probe(
    region_path: str,
    device: torch.device,
    max_steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    log_every: int,
    max_tokens: int,
    warmup_steps: int = 0,
    resume_path: Optional[str] = None,
    checkpoint_out: Optional[str] = None,
    checkpoint_every: int = 0,
) -> Tuple[ImagineNet, McImagineDataset]:
    """Trains `ImagineNet` on every chunk of ONE region, occupancy loss only.

    `chunks_per_region` is set to every chunk position in the region (32x32=1024) — the memorisation
    target is the whole region, not a sample of it, since A.1's question is "can this be memorised
    at all", not "how much data is needed".

    `warmup_steps > 0` switches on the same cosine warmup/decay schedule `train.py` uses
    (`make_lr_lambda`), scheduled against `max_steps` as the total. The first A.1 run (4000 steps,
    flat LR) plateaued noisily (occupancy_loss bouncing 0.26-0.44 in the last 600 steps) rather than
    cleanly converging — a decaying LR is the standard fix for that kind of late-training bounce and
    costs nothing to try before concluding r=0.6686 is the ceiling.

    `resume_path`/`checkpoint_out` let an extension continue from a previously-saved probe run
    instead of re-paying the first N steps: `max_steps` is always the TOTAL step budget (not
    "additional steps"), and the LR schedule is computed against that same total on both the
    original and the resumed run so it stays one continuous curve.
    """
    loader_generator = seed_everything(seed)

    n_chunks = CHUNKS_PER_REGION_SIDE * CHUNKS_PER_REGION_SIDE
    ds = McImagineDataset(
        data_dir=os.path.dirname(region_path),
        shard_paths=[region_path],
        chunks_per_region=n_chunks,
        max_tokens=max_tokens,
        index_seed=0,
        shard_cache_size=1,
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=False, generator=loader_generator,
    )

    model = ImagineNet(PROBE_MODEL_CONFIG).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("ImagineNet: %d params, %d chunks in region, %d steps, batch_size=%d, device=%s",
                n_params, len(ds), max_steps, batch_size, device)

    # Occupancy loss only, per §3.1: "set every other head's weight to 0.0 so nothing competes".
    # terrain_weight/biome_weight default to 1.0 in CombinedLoss and are the two whose contribution
    # is NOT gated by an `if weight > 0.0` check (see losses.py CombinedLoss.forward) — they must be
    # passed explicitly, or they would silently keep competing with occupancy for gradient.
    criterion = CombinedLoss(
        terrain_weight=0.0,
        biome_weight=0.0,
        slope_weight=0.0,
        relief_weight=0.0,
        occupancy_weight=1.0,
        overhang_weight=0.0,
        consistency_weight=0.0,
        structure_weight=0.0,
        structure_graph_weight=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, make_lr_lambda(warmup_steps, max_steps)
        )

    global_step = 0
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        global_step = int(ckpt["global_step"])
        logger.info("Resumed probe from %s at global_step %d.", resume_path, global_step)

    def _save_checkpoint() -> None:
        if not checkpoint_out:
            return
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "global_step": global_step,
                "max_steps": max_steps,
                "region_path": region_path,
            },
            checkpoint_out,
        )
        logger.info("Saved probe checkpoint to %s at global_step %d.", checkpoint_out, global_step)

    model.train()
    t0 = time.time()
    while global_step < max_steps:
        for batch in loader:
            if global_step >= max_steps:
                break
            batch = move_batch(batch, device)
            preds = model(batch["prompt_tokens"], batch["chunk_x"], batch["chunk_z"], batch["seed"])
            targets = build_targets(batch)
            loss, loss_dict = criterion(preds, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

            if global_step == 1 or global_step % log_every == 0:
                dt = time.time() - t0
                lr_now = optimizer.param_groups[0]["lr"]
                logger.info(
                    "step %d/%d occupancy_loss=%.5f lr=%.2e (%.2f steps/s, %.1fs elapsed)",
                    global_step, max_steps, loss_dict["occupancy"].item(), lr_now,
                    global_step / dt if dt > 0 else 0.0, dt,
                )

            if checkpoint_every > 0 and global_step % checkpoint_every == 0:
                _save_checkpoint()
    logger.info("Training done in %.1fs.", time.time() - t0)
    _save_checkpoint()
    return model, ds


@torch.no_grad()
def render_region_bands(
    model: ImagineNet, ds: McImagineDataset, device: torch.device, batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs `model` over every chunk in `ds` (one region, through the real conditioning path) and
    assembles `(predicted_band, target_band)`, both `[REGION_SIZE, REGION_SIZE, BAND_HEIGHT]` bool
    in `[Z, X, I]` layout, 1=solid.

    Placed by each sample's own `(cx, cz)` from `ds.index` rather than by iteration order — the
    dataset's chunk order is a random permutation (`chunks_per_region` == every position, drawn via
    `rng.choice` in `McImagineDataset.__init__`), not row-major, so tiling by position in the loader
    would scramble the reassembled grid.
    """
    model.eval()
    pred_grid = np.zeros((REGION_SIZE, REGION_SIZE, BAND_HEIGHT), dtype=bool)
    target_grid = np.zeros((REGION_SIZE, REGION_SIZE, BAND_HEIGHT), dtype=bool)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    idx_cursor = 0
    for batch in loader:
        b = batch["chunk_x"].shape[0]
        sample_indices = list(range(idx_cursor, idx_cursor + b))
        idx_cursor += b

        moved = move_batch(batch, device)
        preds = model(moved["prompt_tokens"], moved["chunk_x"], moved["chunk_z"], moved["seed"])
        pred_solid = (preds["occupancy_logits"] >= 0.0).cpu().numpy()  # [B, 64, 16, 16], logits>=0
        target_solid = batch["occupancy_band"].numpy().astype(bool)     # [B, 64, 16, 16]

        for local_i, sample_i in enumerate(sample_indices):
            _, cx, cz = ds.index[sample_i]
            z0, x0 = cz * 16, cx * 16
            pred_grid[z0:z0 + 16, x0:x0 + 16, :] = np.moveaxis(pred_solid[local_i], 0, 2)
            target_grid[z0:z0 + 16, x0:x0 + 16, :] = np.moveaxis(target_solid[local_i], 0, 2)

    model.train()
    return pred_grid, target_grid


def subsurface_placement_stats(
    pred_solid: np.ndarray, target_solid: np.ndarray, control_seed: int = 0,
) -> Dict[str, Any]:
    """Sub-surface (below ground truth's own topmost solid cell) placement agreement between
    `pred_solid` and `target_solid`, both `[Z, X, I]` bool, 1=solid — `phase4.2-plan.md` §6.1's
    measurement, reproduced here against this probe's own training region.

    "Sub-surface" is defined by the TARGET's topmost solid cell per column, applied identically to
    both sides: this isolates "did the model place air correctly underground" from "does the model
    predict open sky above the surface", which every model gets right trivially and would dilute r
    toward 1 for free.
    """
    has_solid = target_solid.any(axis=2)
    top_i = BAND_HEIGHT - 1 - np.argmax(target_solid[:, :, ::-1], axis=2)
    idx = np.arange(BAND_HEIGHT).reshape(1, 1, BAND_HEIGHT)
    mask = (idx < top_i[:, :, None]) & has_solid[:, :, None]

    gt_air = (~target_solid)[mask]
    model_air = (~pred_solid)[mask]

    gt_rate = float(gt_air.mean()) if gt_air.size else 0.0
    model_rate = float(model_air.mean()) if model_air.size else 0.0

    gt_f = gt_air.astype(np.float64)
    model_f = model_air.astype(np.float64)
    if gt_f.std() > 0 and model_f.std() > 0:
        r = float(np.corrcoef(model_f, gt_f)[0, 1])
    else:
        r = float("nan")

    intersection = int((gt_air & model_air).sum())
    union = int((gt_air | model_air).sum())
    iou = float(intersection / union) if union > 0 else 0.0
    chance = gt_rate * model_rate
    lift = float((intersection / gt_air.size) / chance) if chance > 0 and gt_air.size else float("nan")

    rng = np.random.default_rng(control_seed)
    shuffled = model_f.copy()
    rng.shuffle(shuffled)
    if gt_f.std() > 0 and shuffled.std() > 0:
        r_control = float(np.corrcoef(shuffled, gt_f)[0, 1])
    else:
        r_control = float("nan")

    return {
        "sub_surface_cells": int(gt_air.size),
        "model_sub_surface_air_pct": model_rate * 100.0,
        "truth_sub_surface_air_pct": gt_rate * 100.0,
        "iou": iou,
        "pearson_r": r,
        "lift_over_chance": lift,
        "shuffled_pairing_control_r": r_control,
    }


def cavity_summary(metrics: Dict[str, Any]) -> Dict[str, Any]:
    cav = metrics["cavity"]
    return {
        "count": cav["count"],
        "count_per_1000_columns": cav["count_per_1000_columns"],
        "mean_height": cav["mean_height"],
        "max_height": cav["max_height"],
        "percentiles": cav["percentiles"],
        "height_counts": cav["height_counts"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 4.3 §3.1 A.1 — single-region learnability probe."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--region", default=None,
                         help=f"Region shard to train on (default: {DEFAULT_REGION}).")
    parser.add_argument("--scan-regions", action="store_true",
                         help="Re-scan --data-dir for the highest multi-run-rate region instead of "
                              "using the pinned default.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int, default=4000,
                         help="Plan caps this at 4000 (§3.1's A.1 row).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", default=None, help="Optional path to write a JSON report.")
    parser.add_argument("--warmup-steps", type=int, default=0,
                         help="Cosine warmup/decay schedule over --max-steps total steps (0 = flat "
                              "LR, the original A.1 protocol).")
    parser.add_argument("--resume", default=None,
                         help="Probe checkpoint (from --checkpoint-out) to resume from — --max-steps "
                              "is the TOTAL budget, not additional steps.")
    parser.add_argument("--checkpoint-out", default=None,
                         help="Path to save a probe checkpoint (model+optimizer+scheduler+step) at "
                              "the end of training, for a later --resume extension.")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                         help="Also checkpoint every N steps (0 = only at the end).")
    args = parser.parse_args()

    if args.max_steps > 4000:
        logger.warning(
            "--max-steps %d exceeds the plan's original 4000-step cap for A.1 (an authorized "
            "extension, not the base protocol).", args.max_steps,
        )

    if args.region:
        region_path = args.region
    elif args.scan_regions:
        region_path = scan_for_most_carved_region(args.data_dir)
    else:
        region_path = DEFAULT_REGION
    if not os.path.isfile(region_path):
        raise FileNotFoundError(f"region shard not found: {region_path}")

    device = resolve_device(args.device)

    with np.load(region_path) as z:
        region_gt_metrics = compute_overhang_metrics(z["band"], layout="zxi")
        carve_strength = float(z["param_carve_strength"]) if "param_carve_strength" in z else None

    logger.info(
        "Region: %s (carve_strength=%s, ground-truth multi-run %.2f%%, mean cavity height %.2f)",
        region_path, carve_strength, region_gt_metrics["multi_run_column_pct"],
        region_gt_metrics["cavity"]["mean_height"],
    )

    model, ds = train_probe(
        region_path=region_path,
        device=device,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        log_every=args.log_every,
        max_tokens=args.max_tokens,
        warmup_steps=args.warmup_steps,
        resume_path=args.resume,
        checkpoint_out=args.checkpoint_out,
        checkpoint_every=args.checkpoint_every,
    )

    logger.info("Rendering full-region predicted vs. target bands through McImagineDataset...")
    pred_grid, target_grid = render_region_bands(model, ds, device, args.eval_batch_size)

    placement = subsurface_placement_stats(pred_grid, target_grid)
    pred_metrics = compute_overhang_metrics(pred_grid, layout="zxi")
    target_metrics = compute_overhang_metrics(target_grid, layout="zxi")

    r = placement["pearson_r"]
    if not np.isnan(r) and r >= 0.7:
        verdict = "r >= 0.7: PLACEMENT IS LEARNABLE. Budget/scale, not design. Proceed to §3.2."
    elif np.isnan(r) or r < 0.3:
        verdict = "r < 0.3 (or undefined): NOT LEARNABLE even memorising one region. Proceed to §3.3."
    else:
        verdict = "0.3 <= r < 0.7: PARTIAL. Treat as the r<0.3 branch; run §3.2 in parallel."

    print("\n" + "=" * 78)
    print("            PHASE 4.3 §3.1 A.1 — SINGLE-REGION LEARNABILITY PROBE            ")
    print("=" * 78)
    print(f"Region:                     {region_path}")
    print(f"Ground-truth multi-run %:   {region_gt_metrics['multi_run_column_pct']:.2f}%")
    print(f"Steps trained:              {args.max_steps}")
    print("-" * 78)
    print(f"Sub-surface cells compared: {placement['sub_surface_cells']:,}")
    print(f"Model sub-surface air %:    {placement['model_sub_surface_air_pct']:.4f}%")
    print(f"Truth sub-surface air %:    {placement['truth_sub_surface_air_pct']:.4f}%")
    print(f"IoU:                        {placement['iou']:.4f}")
    print(f"PEARSON R (train, this region): {r:.4f}")
    print(f"Lift over chance:           {placement['lift_over_chance']:.4f}")
    print(f"Shuffled-pairing control r: {placement['shuffled_pairing_control_r']:.4f}")
    print("-" * 78)
    print("Cavity height distribution — PREDICTED:")
    print(f"  count={pred_metrics['cavity']['count']}, mean={pred_metrics['cavity']['mean_height']:.2f}, "
          f"p50={pred_metrics['cavity']['percentiles']['p50']}, "
          f"p90={pred_metrics['cavity']['percentiles']['p90']}, "
          f"max={pred_metrics['cavity']['max_height']}")
    print("Cavity height distribution — TARGET (ground truth, via McImagineDataset):")
    print(f"  count={target_metrics['cavity']['count']}, mean={target_metrics['cavity']['mean_height']:.2f}, "
          f"p50={target_metrics['cavity']['percentiles']['p50']}, "
          f"p90={target_metrics['cavity']['percentiles']['p90']}, "
          f"max={target_metrics['cavity']['max_height']}")
    print("=" * 78)
    print(verdict)
    print("=" * 78)

    if args.output:
        report = {
            "region_path": region_path,
            "carve_strength": carve_strength,
            "region_ground_truth_multi_run_pct": region_gt_metrics["multi_run_column_pct"],
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "device": str(device),
            "placement": placement,
            "predicted_cavity": cavity_summary(pred_metrics),
            "target_cavity": cavity_summary(target_metrics),
            "verdict": verdict,
        }
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Wrote report to %s", args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
