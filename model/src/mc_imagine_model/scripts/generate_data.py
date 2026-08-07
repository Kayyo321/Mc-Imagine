"""
Regenerates the training shards under `model/data/`.

`model/data/` is ~1.6 GB and gitignored, so a fresh clone has no dataset. Until now the only way to
produce one was to call `ProceduralWorldSource.generate_shards` by hand from a REPL — a method with
no entry point, which is one of the handoff blockers docs/phase2-plan.md §0 lists. This is that
entry point.

Fully deterministic given `--seed`: every macro-region's archetype, parameters and world seed come
from `_region_rng(seed, region_x, region_z)` in `data/world_generator.py`, so the same `--seed` and
`--regions` on any machine produce byte-comparable shards. That determinism is what makes "the
friend's GPU box trains on the same data" a checkable claim rather than a hope.

Usage:
    python -m mc_imagine_model.scripts.generate_data --regions 8000 --out model/data
    python -m mc_imagine_model.scripts.generate_data --regions 64 --out /tmp/smoke --seed 1 --force
"""

import argparse
import concurrent.futures
import glob
import os
import sys
import time
from typing import List

from mc_imagine_model.data.world_generator import ProceduralWorldSource

SHARD_GLOB = "region_*.npz"


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} B"


def existing_shards(out_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(out_dir, SHARD_GLOB)))


def _report_progress(out_dir: str, target: int, future: concurrent.futures.Future) -> None:
    """Polls the output directory while generation runs on a worker thread.

    `generate_shards` writes every shard in one uninterruptible call and takes no progress callback,
    and this script is deliberately not allowed to fork its loop (that would duplicate the region
    layout / RNG contract in two places and let them drift). Counting files on disk gives honest
    progress without touching `world_generator.py` at all.
    """
    t0 = time.time()
    last_done = -1
    while not future.done():
        done = len(existing_shards(out_dir))
        if done != last_done:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 and done > 0 else 0.0
            eta = f"{(target - done) / rate:6.0f}s" if rate > 0 else "     ?"
            print(
                f"\r  {done}/{target} shards  ({100.0 * done / max(1, target):5.1f}%)  "
                f"{rate:6.1f} shards/s  eta {eta}",
                end="", flush=True,
            )
            last_done = done
        time.sleep(0.5)
    print()


def generate(num_regions: int, out_dir: str, seed: int, region_layout_width: int,
             force: bool, quiet: bool, workers: int = 1) -> int:
    os.makedirs(out_dir, exist_ok=True)

    pre_existing = existing_shards(out_dir)
    if pre_existing:
        if not force:
            print(
                f"ERROR: {out_dir} already contains {len(pre_existing)} shard(s) matching "
                f"{SHARD_GLOB}. Re-run with --force to delete them and regenerate, or pick a "
                "different --out. Refusing to mix two datasets in one directory.",
                file=sys.stderr,
            )
            return 1
        print(f"--force: removing {len(pre_existing)} existing shard(s) from {out_dir}")
        for p in pre_existing:
            os.remove(p)

    print(f"Generating {num_regions} region shard(s)")
    print(f"  out                 : {os.path.abspath(out_dir)}")
    print(f"  seed                : {seed}")
    print(f"  region_layout_width : {region_layout_width}")
    print(f"  workers             : {workers}")

    source = ProceduralWorldSource(seed=seed)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(source.generate_shards, num_regions, out_dir, region_layout_width, workers)
        if not quiet:
            _report_progress(out_dir, num_regions, future)
        paths = future.result()  # re-raises anything the generator threw
    elapsed = time.time() - t0

    written = existing_shards(out_dir)
    total_bytes = sum(os.path.getsize(p) for p in written)
    print("--- summary ---")
    print(f"  shards written : {len(paths)} (on disk: {len(written)})")
    print(f"  total size     : {_human(total_bytes)}  (avg {_human(total_bytes / max(1, len(written)))}/shard)")
    print(f"  elapsed        : {elapsed:.1f}s  ({len(written) / max(elapsed, 1e-9):.1f} shards/s)")
    print(f"  output dir     : {os.path.abspath(out_dir)}")

    if len(written) != num_regions:
        print(
            f"ERROR: expected {num_regions} shards but found {len(written)} on disk.",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate procedural training shards for the Mc-Imagine model."
    )
    parser.add_argument("--regions", type=int, default=8000,
                        help="Number of macro-regions (one .npz shard each). Default: 8000")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory (default: model/data)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Dataset seed — same seed gives the same dataset anywhere. Default: 0")
    parser.add_argument("--region-layout-width", type=int, default=64,
                        help="Regions per row in the bookkeeping grid. Default: 64")
    parser.add_argument("--force", action="store_true",
                        help="Delete existing region_*.npz in --out before generating")
    parser.add_argument("--quiet", action="store_true", help="Suppress the progress line")
    parser.add_argument("--workers", type=int, default=1,
                        help="Regions rendered in parallel via ProcessPoolExecutor, one region per "
                             "task (docs/phase4.2-plan.md §2.1). Default 1 (sequential); shards are "
                             "byte-identical regardless of worker count.")
    args = parser.parse_args()

    if args.regions <= 0:
        parser.error("--regions must be positive")
    if args.region_layout_width <= 0:
        parser.error("--region-layout-width must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")

    out_dir = args.out
    if out_dir is None:
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # …/mc_imagine_model
        model_root = os.path.dirname(os.path.dirname(pkg_dir))                # …/model
        out_dir = os.path.join(model_root, "data")

    sys.exit(generate(args.regions, out_dir, args.seed, args.region_layout_width,
                      args.force, args.quiet, args.workers))


if __name__ == "__main__":
    main()
