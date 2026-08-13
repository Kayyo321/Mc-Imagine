#!/usr/bin/env python3
"""
Scale ladder probe (Mc-Imagine Phase 4.3 — docs/phase4.3-plan.md §3.2, A.2).

Only meaningful once A.1 has answered r>=0.7 (measured 2026-08-08: r=0.8836, 12000 steps, single
region — see `learnability_probe.py`). This is A.1's protocol repeated at 1 -> 8 -> 64 region rungs,
a FIXED step budget per region (so total steps scale with region count), recording TRAIN r (on the
training regions' own chunks) and HELD-OUT r (on a region set never trained on, fixed across every
rung) at each rung — "the rental decision in numeric form" (§3.2): if train r stays high while
held-out r collapses, the model is memorising and more regions will not generalise; if both degrade
gracefully, the trend projects toward 8000 regions.

BOUNDED TO 64 REGIONS LOCALLY, NOT 400 (operator decision, 2026-08-08). The plan's literal ladder is
1/8/64/400 — at this machine's measured MPS throughput (~2.6 steps/s, though A.1's extended run saw
an unexplained 10x slowdown partway through), a 400-region rung at any informative per-region step
budget is many hours to potentially days. §3.2's own text permits projecting the trend rather than
measuring every point ("If both degrade gracefully, extrapolate to 8000 regions and report the
projected r"); this script measures 1/8/64 and reports the trend, leaving 400/8000 as a projection
(or a task for the eventual rented box, alongside the confirmatory run).

Held-out regions are FIXED across every rung (never touched by training at any rung, any N): the top
`--num-holdout` cave-dense regions by ground-truth multi-run rate. Training rungs are drawn from the
NEXT most cave-dense regions after the held-out set, NESTED (the 8-region rung's regions are a
superset of the 1-region rung's, the 64-region rung's a superset of the 8-region rung's) — a clean,
standard scaling-curve design, not required by the plan but avoids rung-to-rung region composition
being a confound.

Usage:
    PYTHONPATH=model/src python model/src/mc_imagine_model/scripts/scale_ladder_probe.py \\
        --region-counts 1,8,64 --steps-per-region 300 --device mps \\
        --output model/scale_ladder_report.json
"""

import argparse
import glob
import json
import logging
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from mc_imagine_model.data.dataset import CHUNKS_PER_REGION_SIDE, McImagineDataset
from mc_imagine_model.model.imagine_net import ImagineNet
from mc_imagine_model.scripts.diagnose_overhangs import compute_overhang_metrics
from mc_imagine_model.scripts.learnability_probe import (
    PROBE_MODEL_CONFIG,
    subsurface_placement_stats,
)
from mc_imagine_model.training.losses import CombinedLoss
from mc_imagine_model.training.train import build_targets, make_lr_lambda, move_batch, resolve_device, seed_everything

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger.setLevel(logging.INFO)

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ROOT = os.path.dirname(os.path.dirname(_PKG_DIR))
DEFAULT_DATA_DIR = os.path.join(MODEL_ROOT, "data")

DEFAULT_HOLDOUT_COUNT = 8
DEFAULT_HOLDOUT_CHUNKS = 400   # chunks drawn (across the whole held-out set) for the held-out r measurement
DEFAULT_TRAIN_EVAL_CHUNKS = 400  # chunks drawn from the rung's OWN training regions for train r


def rank_regions_by_multi_run(data_dir: str) -> List[Tuple[str, float]]:
    """All `region_*.npz` shards, sorted by ground-truth multi-run column rate, most cave-dense
    first. Reused as the single source of region density for both the held-out reservation and the
    nested training rungs, so the two can never accidentally overlap by construction (see `main`).
    """
    paths = sorted(glob.glob(os.path.join(data_dir, "region_*.npz")))
    if not paths:
        raise FileNotFoundError(f"no region_*.npz shards under {data_dir}")
    ranked = []
    for p in paths:
        with np.load(p) as z:
            pct = compute_overhang_metrics(z["band"], layout="zxi")["multi_run_column_pct"]
        ranked.append((p, pct))
    ranked.sort(key=lambda t: -t[1])
    return ranked


def build_dataset(region_paths: List[str], chunks_per_region: int, max_tokens: int, index_seed: int) -> McImagineDataset:
    return McImagineDataset(
        data_dir=os.path.dirname(region_paths[0]),
        shard_paths=region_paths,
        chunks_per_region=chunks_per_region,
        max_tokens=max_tokens,
        index_seed=index_seed,
        shard_cache_size=len(region_paths),
    )


def train_rung(
    region_paths: List[str],
    device: torch.device,
    total_steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    warmup_steps: int,
    seed: int,
    log_every: int,
    max_tokens: int,
    chunks_per_region: int,
) -> ImagineNet:
    """A.1's protocol (full ImagineNet, real conditioning path, occupancy loss only), but sampling
    `chunks_per_region` chunks per region — train.py's own real default — rather than A.1's full
    1024-per-region enumeration, since this ladder's point is scale behaviour under a REALISTIC
    per-region sampling density held fixed across rungs, not single-region memorisation.
    """
    loader_generator = seed_everything(seed)
    ds = build_dataset(region_paths, chunks_per_region, max_tokens, index_seed=0)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False, generator=loader_generator)

    model = ImagineNet(PROBE_MODEL_CONFIG).to(device)
    criterion = CombinedLoss(
        terrain_weight=0.0, biome_weight=0.0, slope_weight=0.0, relief_weight=0.0,
        occupancy_weight=1.0, overhang_weight=0.0, consistency_weight=0.0,
        structure_weight=0.0, structure_graph_weight=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(max(1, warmup_steps), total_steps)
    )

    logger.info(
        "Rung: %d regions, %d chunks/region (%d samples), %d total steps, batch_size=%d",
        len(region_paths), chunks_per_region, len(ds), total_steps, batch_size,
    )

    model.train()
    global_step = 0
    t0 = time.time()
    while global_step < total_steps:
        for batch in loader:
            if global_step >= total_steps:
                break
            batch = move_batch(batch, device)
            preds = model(batch["prompt_tokens"], batch["chunk_x"], batch["chunk_z"], batch["seed"])
            targets = build_targets(batch)
            loss, loss_dict = criterion(preds, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step == 1 or global_step % log_every == 0:
                dt = time.time() - t0
                logger.info(
                    "  step %d/%d occupancy_loss=%.5f (%.2f steps/s, %.1fs elapsed)",
                    global_step, total_steps, loss_dict["occupancy"].item(),
                    global_step / dt if dt > 0 else 0.0, dt,
                )
    logger.info("Rung trained in %.1fs.", time.time() - t0)
    return model


@torch.no_grad()
def evaluate_placement_r(
    model: ImagineNet, ds: McImagineDataset, device: torch.device, batch_size: int, max_chunks: int,
) -> Dict[str, Any]:
    """Sub-surface Pearson r over up to `max_chunks` chunks drawn from `ds`, through the real
    conditioning path. Cavity/multi-run metrics are per-COLUMN (see `compute_overhang_metrics`'s own
    docstring), so chunks need not be spatially assembled into a true region grid to be scored
    together — arbitrary chunks are reshaped into one `[B*16, 16, 64]` "column pool" directly. This
    differs from `learnability_probe.render_region_bands`, which assembles a true `[512, 512, 64]`
    grid because A.1 needed genuine within-region cavity-height statistics for ONE region; A.2 only
    needs the pooled sub-surface Pearson r across possibly-disjoint regions.
    """
    model.eval()
    n = min(max_chunks, len(ds))
    rng = np.random.default_rng(0)
    indices = rng.choice(len(ds), size=n, replace=False) if len(ds) > n else np.arange(len(ds))
    loader = DataLoader(torch.utils.data.Subset(ds, indices.tolist()), batch_size=batch_size, shuffle=False)

    pred_chunks, target_chunks = [], []
    for batch in loader:
        moved = move_batch(batch, device)
        preds = model(moved["prompt_tokens"], moved["chunk_x"], moved["chunk_z"], moved["seed"])
        pred_chunks.append((preds["occupancy_logits"] >= 0.0).cpu().numpy())
        target_chunks.append(batch["occupancy_band"].numpy().astype(bool))
    model.train()

    pred = np.concatenate(pred_chunks, axis=0)    # [N, 64, 16, 16] = [N, I, Z, X]
    target = np.concatenate(target_chunks, axis=0)
    n_actual = pred.shape[0]
    # [N, I, Z, X] -> [N, Z, X, I] -> [N*16, 16, 64], matching subsurface_placement_stats's [Z,X,I]
    pred_zxi = np.moveaxis(pred, 1, 3).reshape(n_actual * 16, 16, 64)
    target_zxi = np.moveaxis(target, 1, 3).reshape(n_actual * 16, 16, 64)

    stats = subsurface_placement_stats(pred_zxi, target_zxi)
    stats["n_chunks"] = int(n_actual)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4.3 §3.2 A.2 — scale ladder probe.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--region-counts", default="1,8,64",
                         help="Comma-separated training-rung region counts (plan default 1,8,64,400; "
                              "bounded to 64 locally per the 2026-08-08 operator decision).")
    parser.add_argument("--steps-per-region", type=int, default=300,
                         help="Fixed per-region step budget C — total steps for a rung = C * N.")
    parser.add_argument("--chunks-per-region", type=int, default=100,
                         help="Chunks sampled per region (train.py's own real default).")
    parser.add_argument("--num-holdout", type=int, default=DEFAULT_HOLDOUT_COUNT)
    parser.add_argument("--holdout-eval-chunks", type=int, default=DEFAULT_HOLDOUT_CHUNKS)
    parser.add_argument("--train-eval-chunks", type=int, default=DEFAULT_TRAIN_EVAL_CHUNKS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-fraction", type=float, default=0.1,
                         help="Warmup steps as a fraction of each rung's total steps.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    region_counts = [int(x) for x in args.region_counts.split(",") if x.strip()]
    device = resolve_device(args.device)

    ranked = rank_regions_by_multi_run(args.data_dir)
    logger.info("Ranked %d regions by multi-run rate; top: %.2f%%, %dth: %.2f%%",
                len(ranked), ranked[0][1], args.num_holdout, ranked[args.num_holdout - 1][1])

    holdout_paths = [p for p, _ in ranked[:args.num_holdout]]
    pool_paths = [p for p, _ in ranked[args.num_holdout:]]
    max_n = max(region_counts)
    if max_n > len(pool_paths):
        raise ValueError(f"largest rung ({max_n}) exceeds the available non-held-out pool ({len(pool_paths)})")

    logger.info("Held-out set (%d regions, never trained on any rung): %s",
                len(holdout_paths), [os.path.basename(p) for p in holdout_paths])

    holdout_ds = build_dataset(holdout_paths, args.chunks_per_region * 4, args.max_tokens, index_seed=99)

    results = []
    for n in region_counts:
        train_paths = pool_paths[:n]  # nested: rung N's regions are a prefix of rung N+1's
        total_steps = args.steps_per_region * n
        warmup_steps = max(1, int(total_steps * args.warmup_fraction))
        logger.info("=== Rung: %d region(s), %d total steps, warmup %d ===", n, total_steps, warmup_steps)

        model = train_rung(
            region_paths=train_paths, device=device, total_steps=total_steps,
            batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
            warmup_steps=warmup_steps, seed=args.seed, log_every=args.log_every,
            max_tokens=args.max_tokens, chunks_per_region=args.chunks_per_region,
        )

        train_ds = build_dataset(train_paths, args.chunks_per_region, args.max_tokens, index_seed=0)
        train_stats = evaluate_placement_r(model, train_ds, device, args.eval_batch_size, args.train_eval_chunks)
        holdout_stats = evaluate_placement_r(model, holdout_ds, device, args.eval_batch_size, args.holdout_eval_chunks)

        logger.info(
            "Rung %d regions: train r=%.4f (n=%d chunks), held-out r=%.4f (n=%d chunks)",
            n, train_stats["pearson_r"], train_stats["n_chunks"],
            holdout_stats["pearson_r"], holdout_stats["n_chunks"],
        )
        results.append({
            "num_regions": n,
            "total_steps": total_steps,
            "train": train_stats,
            "holdout": holdout_stats,
        })

    print("\n" + "=" * 78)
    print("        PHASE 4.3 §3.2 A.2 — SCALE LADDER (train r vs. held-out r)        ")
    print("=" * 78)
    print(f"{'regions':>8}{'steps':>10}{'train r':>12}{'holdout r':>12}{'train IoU':>12}{'holdout IoU':>13}")
    for r in results:
        print(f"{r['num_regions']:>8}{r['total_steps']:>10}"
              f"{r['train']['pearson_r']:>12.4f}{r['holdout']['pearson_r']:>12.4f}"
              f"{r['train']['iou']:>12.4f}{r['holdout']['iou']:>13.4f}")
    print("=" * 78)

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "region_counts": region_counts,
                "steps_per_region": args.steps_per_region,
                "num_holdout": args.num_holdout,
                "holdout_regions": [os.path.basename(p) for p in holdout_paths],
                "device": str(device),
                "results": results,
            }, f, indent=2)
        logger.info("Wrote report to %s", args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
