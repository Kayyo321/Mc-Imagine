"""
Pre-training flight check — run this FIRST on any machine that is about to train.

Purpose: convert "discovered 90 minutes into a metered GPU rental" into "discovered in five
minutes". Every check here corresponds to a numbered issue in docs/remote-training-readiness-plan.md
and exists because that failure mode has either already happened or is computable from the shipped
config.

The critical property of this script is that the *budget* checks (P4, P5, C-class disk math) are
computed from the config's declared `--target-regions` rather than from whatever dataset happens to
be on this machine. That is deliberate: the 2026-08 OOM was invisible at rehearsal scale (400
regions, num_workers=0) and only materialized at 8000 regions with 8 workers. P4 therefore fails on
the development Mac too, which is the whole point — you should not have to rent a machine to find
out the config asks for 155 GB of RAM.

Usage:
    python -m mc_imagine_model.scripts.preflight --config model/src/mc_imagine_model/training/config.cuda.yaml
    python -m mc_imagine_model.scripts.preflight --config ... --quick      # skip the timed runs (P9-P12)
    python -m mc_imagine_model.scripts.preflight --config ... --keep-going # run all checks, don't stop at first failure
    python -m mc_imagine_model.scripts.preflight --config ... --record-hashes  # (re)write the P7 reference fixture

Exit code 0 means every check PASSED or was legitimately SKIPPED for this platform. Any FAIL exits
non-zero. Do not start a long run until this is green.
"""

import argparse
import glob
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------------------------
# Constants measured on 2026-08-05 (see docs/remote-training-readiness-plan.md §6). These are the
# numbers the budget arithmetic depends on; if the shard format changes, change them here and the
# whole harness stays honest.
# ---------------------------------------------------------------------------------------------
SHARD_DECOMPRESSED_BYTES = 18_930_000     # band(520,520,64) bool + heightfield f32 + 2x u8 maps
SHARD_COMPRESSED_BYTES = 915_000          # measured average on disk
CHECKPOINT_BYTES = 33_077_647             # measured torch.save of save_checkpoint()'s dict, post-C2
                                           # (text_encoder.* excluded — was 123.4 MB pre-C2, §2.C2)
PER_SAMPLE_BYTES = 20_116                 # collated tensor bytes/sample; occupancy_band is uint8 (§2.B2)
# Must track train.py's `_cache_size`/`VAL_SHARD_CACHE_SIZE`/`VAL_NUM_WORKERS` (§2.A2/A3): the val
# loader gets its own small fixed cache and a capped worker count, and `shard_cache_size: null` is
# bounded by a byte budget rather than resolving to "cache the whole split".
DEFAULT_SHARD_CACHE_BUDGET_BYTES = 1 * 1024**3
VAL_SHARD_CACHE_SIZE = 4
VAL_NUM_WORKERS = 2

# Thresholds. Deliberately generous-but-real: these are "this machine can finish the run", not
# "this machine is fast".
MIN_FREE_DISK_BYTES = 60 * 1000**3
MIN_TOTAL_RAM_BYTES = 32 * 1000**3
MIN_SHM_BYTES = 1 * 1000**3
MAX_CACHE_BUDGET_FRACTION = 0.25          # per-process caches must fit in a quarter of host RAM
MAX_SMOKE_RSS_BYTES = 12 * 1000**3
MAX_SMOKE_VRAM_FRACTION = 0.80

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../mc_imagine_model
MODEL_ROOT = os.path.dirname(os.path.dirname(_PKG_DIR))                  # .../model
HASH_FIXTURE = os.path.join(MODEL_ROOT, "tests", "fixtures", "shard_hashes_seed0.json")


# =============================================================================================
# result plumbing
# =============================================================================================

PASS, FAIL, SKIP, WARN = "PASS", "FAIL", "SKIP", "WARN"


@dataclass
class Result:
    check: str
    title: str
    status: str
    detail: str = ""
    lines: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in (PASS, SKIP, WARN)


def _gb(n: float) -> str:
    return f"{n / 1000**3:.2f} GB"


def _mb(n: float) -> str:
    return f"{n / 1000**2:.1f} MB"


# =============================================================================================
# platform helpers
# =============================================================================================

def total_ram_bytes() -> Optional[int]:
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        pass
    if platform.system() == "Darwin":
        try:
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
        except Exception:
            return None
    return None


def shm_bytes() -> Optional[int]:
    """Size of /dev/shm. Linux only — macOS has no equivalent and does not need one."""
    if not os.path.isdir("/dev/shm"):
        return None
    st = os.statvfs("/dev/shm")
    return st.f_blocks * st.f_frsize


def process_tree_rss(root_pid: int) -> int:
    """Summed RSS of `root_pid` and every descendant, in bytes.

    Must walk the tree, not just the root: the whole point of the 2026-08 OOM is that the memory
    lives in the DataLoader's *worker* processes, which a naive getrusage on the parent never sees.
    `ps` reports RSS in kilobytes on both macOS and Linux.
    """
    try:
        out = subprocess.check_output(["ps", "-eo", "pid=,ppid=,rss="], text=True)
    except Exception:
        return 0
    children: Dict[int, List[int]] = {}
    rss: Dict[int, int] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            pid, ppid, kb = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)
        rss[pid] = kb * 1024
    total, stack = 0, [root_pid]
    seen = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += rss.get(pid, 0)
        stack.extend(children.get(pid, []))
    return total


def nvidia_smi_used_mib() -> Optional[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
        return max(int(v.strip()) for v in out.splitlines() if v.strip())
    except Exception:
        return None


# =============================================================================================
# checks
# =============================================================================================

def p1_install(cfg: Dict[str, Any], args) -> Result:
    """P1 — clean install: every pinned dependency imports, and the package itself resolves."""
    r = Result("P1", "Install / imports", PASS)
    mods = ["numpy", "scipy", "yaml", "torch", "onnx", "onnxruntime", "tokenizers",
            "safetensors", "matplotlib", "tqdm"]
    missing = []
    for m in mods:
        try:
            mod = __import__(m)
            r.lines.append(f"  {m:<14} {getattr(mod, '__version__', '?')}")
        except Exception as exc:
            missing.append(f"{m} ({type(exc).__name__}: {exc})")
    try:
        import mc_imagine_model  # noqa: F401
        r.lines.append(f"  {'mc_imagine_model':<14} importable")
    except Exception as exc:
        missing.append(f"mc_imagine_model — run `pip install -e model/` ({exc})")
    r.lines.append(f"  python {sys.version.split()[0]} on {platform.system()} {platform.machine()}")
    if missing:
        r.status = FAIL
        r.detail = "unimportable: " + "; ".join(missing)
    return r


def p2_gpu(cfg: Dict[str, Any], args) -> Result:
    """P2 — the GPU is real. FAILS (does not warn) when the config names a device it cannot get.

    docs/remote-training-readiness-plan.md §2.E1: train.py currently downgrades cuda->cpu with a
    warning, which on a rented box means paying for an idle GPU for days.
    """
    import torch
    r = Result("P2", "GPU availability", PASS)
    requested = cfg.get("hardware", {}).get("device", "auto")
    r.lines.append(f"  config requests device: {requested!r}")
    r.lines.append(f"  torch {torch.__version__} (cuda build: {torch.version.cuda})")
    have_cuda = torch.cuda.is_available()
    have_mps = torch.backends.mps.is_available()
    r.lines.append(f"  cuda available: {have_cuda}   mps available: {have_mps}")
    if have_cuda:
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory
        r.lines.append(f"  device 0: {props.name}, {_gb(vram)} VRAM, sm_{props.major}{props.minor}")
        if vram < 16 * 1000**3:
            r.status = WARN
            r.detail = (f"{_gb(vram)} VRAM — plan §2.D1 estimates ~6-8 GB of activations at "
                        f"batch_size 256 fp32. Expect to lower batch_size (and the LR with it).")
    if requested == "cuda" and not have_cuda:
        r.status = FAIL
        r.detail = ("config asks for CUDA and torch cannot see it. This is the CPU-only-wheel "
                    "mistake: training would start, print normal progress, and take weeks. "
                    "Reinstall torch with the --index-url matching `nvidia-smi`'s CUDA version.")
    elif requested == "mps" and not have_mps:
        r.status = FAIL
        r.detail = "config asks for MPS and torch cannot see it."
    elif requested == "auto":
        r.status = WARN
        r.detail = ("device: \"auto\" silently accepts CPU. Name the device explicitly in the "
                    "config you train with, so a missing GPU is an error rather than a slow run.")
    return r


def p3_disk(cfg: Dict[str, Any], args) -> Result:
    """P3 — disk headroom, computed from the run's own checkpoint and dataset footprint."""
    r = Result("P3", "Disk headroom", PASS)
    tcfg = cfg["training"]
    n_epochs = int(tcfg.get("num_epochs", 60))
    val_every = int(tcfg.get("val_every_epochs", 1))
    # train.py rotates epoch_NNN.pt down to keep_last_n_checkpoints (default 3) — §2.C1 — so the
    # steady-state count is bounded by that, not by how many validation epochs there are.
    keep_last_n = int(tcfg.get("keep_last_n_checkpoints", 3))
    n_val_epochs = n_epochs // max(1, val_every)
    n_ckpt = min(n_val_epochs, keep_last_n) + 2            # epoch_XXX.pt (capped) + best.pt + last.pt
    ckpt_total = n_ckpt * CHECKPOINT_BYTES
    data_total = args.target_regions * SHARD_COMPRESSED_BYTES
    need = ckpt_total + data_total
    free = shutil.disk_usage(MODEL_ROOT).free
    r.lines.append(f"  checkpoints: {n_ckpt} x {_mb(CHECKPOINT_BYTES)} = {_gb(ckpt_total)}")
    r.lines.append(f"  dataset:     {args.target_regions} shards x {_mb(SHARD_COMPRESSED_BYTES)} = {_gb(data_total)}")
    r.lines.append(f"  run needs:   {_gb(need)} (excludes ~8-10 GB of CUDA wheels)")
    r.lines.append(f"  free now:    {_gb(free)} on {MODEL_ROOT}")
    if free < need:
        r.status = FAIL
        r.detail = (f"only {_gb(free)} free for a run that writes {_gb(need)}. It will run out at "
                    f"an epoch boundary hours in, and torch.save onto a full disk leaves a "
                    f"truncated last.pt — destroying the resume path with the same event.")
    elif free < MIN_FREE_DISK_BYTES:
        r.status = WARN
        r.detail = f"under the {_gb(MIN_FREE_DISK_BYTES)} recommended floor; no room for a second run."
    return r


def p4_host_ram(cfg: Dict[str, Any], args) -> Result:
    """P4 — host RAM vs. the per-process shard-cache budget. THIS IS THE 2026-08 OOM CHECK.

    Computed at `--target-regions` scale, not at whatever is on disk here, because the failure is
    invisible at rehearsal scale. See plan §1.
    """
    r = Result("P4", "Host RAM vs shard-cache budget", PASS)
    dcfg, hcfg = cfg["data"], cfg.get("hardware", {})
    total = total_ram_bytes()
    n_workers = int(hcfg.get("num_workers", 0))

    val_frac = float(dcfg.get("val_region_fraction", 0.08))
    n_val = max(1, int(args.target_regions * val_frac))
    n_train = args.target_regions - n_val

    cap = dcfg.get("shard_cache_size", None)

    def train_cache_size(n_paths: int) -> int:
        if cap is not None:
            want = max(32, n_paths)
            return min(want, cap)
        bounded = max(1, DEFAULT_SHARD_CACHE_BUDGET_BYTES // SHARD_DECOMPRESSED_BYTES)
        return min(n_paths, int(bounded))

    train_cache = train_cache_size(n_train)
    val_cache = min(VAL_SHARD_CACHE_SIZE, max(1, n_val))
    # train.py gives the val loader its own (small, capped) kwargs — §2.A2's fix.
    holders_train = max(1, n_workers)
    holders_val = min(VAL_NUM_WORKERS, n_workers) if n_workers > 0 else 1
    budget = (train_cache * holders_train + val_cache * holders_val + 32) * SHARD_DECOMPRESSED_BYTES

    r.lines.append(f"  target scale: {args.target_regions} regions -> {n_train} train / {n_val} val")
    r.lines.append(f"  shard_cache_size: {cap!r} -> train cache {train_cache}, val cache {val_cache}")
    r.lines.append(f"  num_workers: {n_workers} train / {holders_val} val (separate loader kwargs)")
    r.lines.append(f"  cache per process: {_gb(train_cache * SHARD_DECOMPRESSED_BYTES)}")
    r.lines.append(f"  TOTAL CACHE BUDGET: {_gb(budget)}")
    r.lines.append(f"  host RAM: {_gb(total) if total else 'unknown'}")

    if total is None:
        r.status = WARN
        r.detail = f"could not read total RAM; budget is {_gb(budget)} — check it by hand."
        return r
    if budget > total * MAX_CACHE_BUDGET_FRACTION:
        r.status = FAIL
        r.detail = (f"shard caches alone want {_gb(budget)} of {_gb(total)} host RAM "
                    f"({budget / total:.0%}, ceiling {MAX_CACHE_BUDGET_FRACTION:.0%}). This is the "
                    f"configuration that OOM-killed the 2026-08 run. Lower data.shard_cache_size "
                    f"(the cache only delivers a ~{100 * min(train_cache, n_train) / max(1, n_train):.0f}% "
                    f"hit rate at this scale anyway).")
    elif total < MIN_TOTAL_RAM_BYTES:
        r.status = WARN
        r.detail = f"{_gb(total)} total RAM is under the {_gb(MIN_TOTAL_RAM_BYTES)} recommended floor."
    return r


def p5_shm(cfg: Dict[str, Any], args) -> Result:
    """P5 — /dev/shm vs the DataLoader's in-flight batch requirement.

    Docker defaults /dev/shm to 64 MB and every GPU rental is containerized. The failure presents as
    `Bus error` / `DataLoader worker killed by signal`, which reads like failing hardware.
    """
    r = Result("P5", "/dev/shm vs DataLoader batches", PASS)
    tcfg, hcfg = cfg["training"], cfg.get("hardware", {})
    bs = int(tcfg.get("batch_size", 64))
    n_workers = int(hcfg.get("num_workers", 0))
    prefetch = int(hcfg.get("prefetch_factor", 2))
    per_batch = bs * PER_SAMPLE_BYTES
    in_flight = per_batch * max(1, n_workers) * prefetch

    r.lines.append(f"  batch {bs} x {PER_SAMPLE_BYTES / 1024:.1f} KiB/sample = {_mb(per_batch)}/batch")
    r.lines.append(f"  in flight: {max(1, n_workers)} workers x prefetch {prefetch} = {_mb(in_flight)}")

    if n_workers == 0:
        r.status = SKIP
        r.detail = "num_workers=0 — batches never cross shared memory."
        return r
    size = shm_bytes()
    if size is None:
        r.status = SKIP
        r.detail = (f"no /dev/shm on {platform.system()} (not applicable). On the Linux box this "
                    f"run needs at least {_mb(in_flight)}; launch with --shm-size=8g or --ipc=host.")
        return r
    r.lines.append(f"  /dev/shm: {_gb(size)}")
    if size < in_flight * 2:
        r.status = FAIL
        r.detail = (f"/dev/shm is {_gb(size)} but the loader needs {_mb(in_flight)} in flight "
                    f"(plus pinned staging). Relaunch the container with --shm-size=8g or "
                    f"--ipc=host. No code change fixes this.")
    elif size < MIN_SHM_BYTES:
        r.status = WARN
        r.detail = f"/dev/shm under {_gb(MIN_SHM_BYTES)}; fine for now but raise it before scaling workers."
    return r


def p6_text_encoder(cfg: Dict[str, Any], args) -> Result:
    """P6 — the MiniLM checkpoint is real, not the silent random fallback.

    text_encoder.py:198-208 falls back to a randomly-initialized frozen encoder with a print and no
    exception. model/checkpoints/ is gitignored, so a fresh clone is one forgotten `bootstrap` away
    from a run that converges and has learned nothing about text.
    """
    from mc_imagine_model.model.text_encoder import PromptEncoder
    r = Result("P6", "Text encoder is pretrained", PASS)
    # allow_random=True so a missing checkpoint reports as a formatted P6 FAIL below rather than
    # an unhandled exception — PromptEncoder itself now raises by default (§2.E2's fix).
    enc = PromptEncoder(checkpoint_dir=cfg.get("model", {}).get("checkpoint_dir"), allow_random=True)
    if enc._loaded_from is None:
        r.status = FAIL
        r.detail = ("PromptEncoder fell back to a RANDOM frozen encoder. Training will converge and "
                    "every prompt will produce the same terrain. Run "
                    "`python -m mc_imagine_model.scripts.bootstrap`.")
    else:
        r.lines.append(f"  loaded from {enc._loaded_from}")
    return r


def _hash_shards(paths: List[str]) -> Dict[str, str]:
    out = {}
    for p in paths:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        out[os.path.basename(p)] = h.hexdigest()
    return out


def p7_determinism(cfg: Dict[str, Any], args) -> Result:
    """P7 — the dataset generator really is machine-independent.

    generate_data.py's docstring and TRAINING.md both rest on "same --seed on any machine produces
    byte-comparable shards". That claim is load-bearing and has never been checked across platforms.

    Generated via the PARALLEL path (docs/phase4.2-plan.md §2.1's `max_workers`), not the sequential
    default: this is what actually exercises "parallel execution is byte-identical to sequential" on
    every preflight run, rather than trusting a one-off manual check that can silently go stale the
    next time `_render_and_save_region` changes.
    """
    from mc_imagine_model.data.world_generator import ProceduralWorldSource
    r = Result("P7", "Dataset determinism", PASS)
    n = 8
    workers = min(4, os.cpu_count() or 1)
    tmp = tempfile.mkdtemp(prefix="mcim_preflight_")
    try:
        t0 = time.time()
        ProceduralWorldSource(seed=0).generate_shards(n, tmp, region_layout_width=64,
                                                      max_workers=workers)
        elapsed = time.time() - t0
        got = _hash_shards(sorted(glob.glob(os.path.join(tmp, "region_*.npz"))))
        r.lines.append(f"  generated {n} regions in {elapsed:.1f}s ({elapsed / n:.2f}s/region)")
        if args.record_hashes or not os.path.isfile(HASH_FIXTURE):
            os.makedirs(os.path.dirname(HASH_FIXTURE), exist_ok=True)
            with open(HASH_FIXTURE, "w", encoding="utf-8") as f:
                json.dump({"seed": 0, "region_layout_width": 64,
                           "recorded_on": f"{platform.system()}-{platform.machine()}",
                           "hashes": got}, f, indent=2, sort_keys=True)
            r.status = WARN
            r.detail = (f"reference fixture written to {HASH_FIXTURE} from THIS machine. Commit it, "
                        f"then this check becomes a real cross-platform comparison.")
            return r
        with open(HASH_FIXTURE, "r", encoding="utf-8") as f:
            ref = json.load(f)
        mismatched = [k for k, v in ref["hashes"].items() if got.get(k) != v]
        r.lines.append(f"  compared against fixture recorded on {ref.get('recorded_on', '?')}")
        if mismatched:
            r.status = FAIL
            r.detail = (f"{len(mismatched)}/{len(ref['hashes'])} shards differ from the reference: "
                        f"{mismatched[:3]}. Either the generator changed (regenerate the dataset "
                        f"and re-record) or this platform's float behaviour differs — in which case "
                        f"'both machines train on the same data' is false and must be resolved "
                        f"before a comparison run.")
        else:
            r.lines.append(f"  all {len(got)} shard hashes identical")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return r


def p8_target_range(cfg: Dict[str, Any], args) -> Result:
    """P8 — surface the existing target-range tripwire before the long job, not 30s into it."""
    from torch.utils.data import DataLoader
    from mc_imagine_model.data.dataset import McImagineDataset
    from mc_imagine_model.training.train import (assert_targets_representable, resolve_path,
                                                 split_shards_by_region)
    r = Result("P8", "Target range tripwire", PASS)
    data_dir = resolve_path(cfg["data"]["data_dir"])
    try:
        train_paths, _ = split_shards_by_region(data_dir, cfg["data"].get("val_region_fraction", 0.08))
    except FileNotFoundError as exc:
        r.status = FAIL
        r.detail = str(exc)
        return r
    r.lines.append(f"  {len(train_paths)} train shards in {data_dir}")
    ds = McImagineDataset(data_dir, shard_paths=train_paths[:16],
                          chunks_per_region=cfg["data"].get("chunks_per_region", 100),
                          max_tokens=cfg["data"].get("max_prompt_length", 128),
                          index_seed=1, shard_cache_size=4)
    loader = DataLoader(ds, batch_size=int(cfg["training"].get("batch_size", 64)),
                        shuffle=False, num_workers=0)
    try:
        assert_targets_representable(loader, num_batches=4)
        r.lines.append("  all height/water targets inside the head's representable band")
    except Exception as exc:
        r.status = FAIL
        r.detail = str(exc).splitlines()[0]
    return r


_STEP_RE = re.compile(r"epoch (\d+) step (\d+)/(\d+) loss=")
_REGIONS_RE = re.compile(r"Regions: (\d+) train, (\d+) val")


def _run_train_subprocess(cfg_path: str, max_steps: int, extra: Optional[List[str]] = None
                          ) -> Tuple[int, List[str], int, Optional[int], List[Tuple[int, float]]]:
    """Runs train.py as a child, sampling process-tree RSS and GPU memory while it goes.

    Returns (returncode, output_lines, peak_rss_bytes, peak_vram_mib, [(step, monotonic_time)]).
    """
    cmd = [sys.executable, "-m", "mc_imagine_model.training.train",
           "--config", cfg_path, "--max-steps", str(max_steps)] + (extra or [])
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env)
    lines: List[str] = []
    steps: List[Tuple[int, float]] = []

    def reader():
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line.rstrip())
            m = _STEP_RE.search(line)
            if m:
                steps.append((int(m.group(2)), time.monotonic()))

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    peak_rss, peak_vram = 0, None
    while proc.poll() is None:
        peak_rss = max(peak_rss, process_tree_rss(proc.pid))
        v = nvidia_smi_used_mib()
        if v is not None:
            peak_vram = max(peak_vram or 0, v)
        time.sleep(0.5)
    t.join(timeout=5)
    return proc.returncode, lines, peak_rss, peak_vram, steps


def _temp_config(cfg: Dict[str, Any], checkpoint_dir: str) -> str:
    import yaml
    cfg = json.loads(json.dumps(cfg))  # deep copy of plain YAML types
    cfg["training"]["checkpoint_dir"] = checkpoint_dir
    cfg["training"]["qualitative_dump_every_epochs"] = 10**6  # not what P9 is measuring
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="mcim_preflight_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return path


def p9_p10_smoke(cfg: Dict[str, Any], args) -> List[Result]:
    """P9 — bounded smoke run at the REAL config, measuring peak process-tree RSS and GPU memory.
    P10 — measured throughput from the same run, projected to the full schedule.

    P9 is the check that would have caught the 2026-08 OOM: it measures the whole process tree, so
    the memory living in DataLoader workers is actually visible.
    """
    import torch
    r9 = Result("P9", f"Bounded smoke run ({args.smoke_steps} steps, real config)", PASS)
    r10 = Result("P10", "Measured throughput and projected wall-clock", PASS)

    ckpt_dir = tempfile.mkdtemp(prefix="mcim_preflight_ckpt_")
    cfg_path = _temp_config(cfg, ckpt_dir)
    try:
        t0 = time.monotonic()
        rc, lines, peak_rss, peak_vram, steps = _run_train_subprocess(cfg_path, args.smoke_steps)
        wall = time.monotonic() - t0

        if rc != 0:
            r9.status = FAIL
            r9.detail = f"train.py exited {rc}. Last output:\n    " + "\n    ".join(lines[-15:])
            r10.status = SKIP
            r10.detail = "no successful run to measure"
            return [r9, r10]

        r9.lines.append(f"  wall clock: {wall:.1f}s")
        r9.lines.append(f"  peak process-tree RSS: {_gb(peak_rss)} (ceiling {_gb(MAX_SMOKE_RSS_BYTES)})")
        if peak_rss > MAX_SMOKE_RSS_BYTES:
            r9.status = FAIL
            r9.detail = (f"peak RSS {_gb(peak_rss)} over a {args.smoke_steps}-step run. At full "
                         f"scale this grows with shard_cache_size. P4's budget arithmetic is the "
                         f"explanation; fix that before running longer.")
        total = total_ram_bytes()
        if total and peak_rss > 0.5 * total:
            r9.status = FAIL if r9.status == PASS else r9.status
            r9.detail = (r9.detail or "") + f" peak RSS is {peak_rss / total:.0%} of host RAM."

        if peak_vram is not None:
            r9.lines.append(f"  peak GPU memory (nvidia-smi): {peak_vram} MiB")
            if torch.cuda.is_available():
                cap = torch.cuda.get_device_properties(0).total_memory / 1024**2
                r9.lines.append(f"  card total: {cap:.0f} MiB ({peak_vram / cap:.0%} used)")
                if peak_vram > MAX_SMOKE_VRAM_FRACTION * cap:
                    r9.status = FAIL if r9.status == PASS else r9.status
                    r9.detail = ((r9.detail or "") +
                                 f" GPU at {peak_vram / cap:.0%} of capacity — lower batch_size "
                                 f"(and scale learning_rate with it).")
        else:
            r9.lines.append("  GPU memory: nvidia-smi unavailable (not a CUDA box)")

        # ---- P10 -------------------------------------------------------------------------
        if len(steps) < 2:
            r10.status = WARN
            r10.detail = (f"only {len(steps)} step log line(s) — raise --smoke-steps above "
                          f"training.log_every ({cfg['training'].get('log_every', 50)}) to measure a rate.")
            return [r9, r10]
        (s0, t_0), (s1, t_1) = steps[0], steps[-1]
        rate = (s1 - s0) / max(1e-9, t_1 - t_0)
        r10.lines.append(f"  measured: {rate:.2f} steps/s "
                         f"({rate * cfg['training'].get('batch_size', 64):.0f} samples/s)")

        n_train = None
        for line in lines:
            m = _REGIONS_RE.search(line)
            if m:
                n_train = int(m.group(1))
        bs = int(cfg["training"].get("batch_size", 64))
        cpr = int(cfg["data"].get("chunks_per_region", 100))
        n_epochs = int(cfg["training"].get("num_epochs", 60))
        if n_train:
            r10.lines.append(f"  this dataset: {n_train} train regions -> {n_train * cpr // bs} steps/epoch")
        val_frac = float(cfg["data"].get("val_region_fraction", 0.08))
        tgt_train = args.target_regions - max(1, int(args.target_regions * val_frac))
        tgt_steps = (tgt_train * cpr // bs) * n_epochs
        hours = tgt_steps / rate / 3600
        r10.lines.append(f"  at --target-regions {args.target_regions}: {tgt_steps} steps over "
                         f"{n_epochs} epochs")
        r10.lines.append(f"  PROJECTED WALL-CLOCK: {hours:.1f} h  <-- price the rental against this")
        r10.detail = ("Projection assumes this machine's throughput. If you measured on the Mac and "
                      "will rent a CUDA box, treat it as an upper bound on GPU time and a LOWER "
                      "bound on data-path time (the data path is the bottleneck — plan §2.F1).")
        if hours > args.budget_hours:
            r10.status = FAIL
            r10.detail = (f"projected {hours:.1f} h exceeds --budget-hours {args.budget_hours}. "
                          f"Fix the data path (plan §2.F1/F2/F3) before renting, not during.")
    finally:
        os.unlink(cfg_path)
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    return [r9, r10]


def p11_checkpoint_roundtrip(cfg: Dict[str, Any], args) -> Result:
    """P11 — a checkpoint written by this machine can actually be resumed by it."""
    r = Result("P11", "Checkpoint write + resume round-trip", PASS)
    ckpt_dir = tempfile.mkdtemp(prefix="mcim_preflight_rt_")
    cfg_path = _temp_config(cfg, ckpt_dir)
    n = max(2, int(cfg["training"].get("log_every", 50)))
    try:
        rc, lines, _, _, _ = _run_train_subprocess(cfg_path, n)
        if rc != 0:
            r.status = FAIL
            r.detail = f"initial run exited {rc}: " + " | ".join(lines[-5:])
            return r
        last = os.path.join(ckpt_dir, "last.pt")
        if not os.path.isfile(last):
            r.status = FAIL
            r.detail = (f"no last.pt after {n} steps. The --max-steps early-exit path must write "
                        f"checkpoints (docs/phase4.1-plan.md §1.1 fixed this once already).")
            return r
        size = os.path.getsize(last)
        r.lines.append(f"  last.pt written: {_mb(size)}")
        rc2, lines2, _, _, _ = _run_train_subprocess(cfg_path, n * 2, extra=["--resume", last])
        if rc2 != 0:
            r.status = FAIL
            r.detail = f"resume exited {rc2}: " + " | ".join(lines2[-5:])
            return r
        resumed = [ln for ln in lines2 if "Resumed from" in ln]
        if not resumed:
            r.status = FAIL
            r.detail = "resume produced no 'Resumed from' line — it started fresh."
        else:
            r.lines.append(f"  {resumed[0].split('INFO')[-1].strip()}")
        # train.py rotates epoch_NNN.pt down to keep_last_n_checkpoints (default 3) — §2.C1 — so
        # the steady-state total is bounded by that, not by num_epochs.
        keep_last_n = int(cfg["training"].get("keep_last_n_checkpoints", 3))
        n_epochs = int(cfg["training"].get("num_epochs", 60))
        capped_epoch_ckpts = min(n_epochs, keep_last_n)
        projected = (capped_epoch_ckpts + 2) * size  # + best.pt + last.pt
        r.lines.append(f"  keep_last_n_checkpoints: {keep_last_n}")
        r.lines.append(f"  projected checkpoint total (steady state): {_gb(projected)}")
    finally:
        os.unlink(cfg_path)
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    return r


def p12_tests(cfg: Dict[str, Any], args) -> Result:
    """P12 — the unit suite is green on THIS machine, before anything expensive runs."""
    r = Result("P12", "Unit test suite", PASS)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(MODEL_ROOT, "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                          cwd=MODEL_ROOT, env=env, capture_output=True, text=True)
    tail = (proc.stdout or proc.stderr).strip().splitlines()[-3:]
    r.lines.extend(f"  {ln}" for ln in tail)
    if proc.returncode != 0:
        r.status = FAIL
        r.detail = "pytest failed on this machine — resolve before spending GPU hours."
    return r


def p13_dataset_density(cfg: Dict[str, Any], args) -> Result:
    """P13 — dataset structure density: ground-truth multi-run column % vs a floor.

    docs/phase4.1-plan.md §1.3 / docs/phase4.2-plan.md §1.3: Gate 2 condition 1 requires the
    model's multi-run column rate to clear the marginal-probability baseline (0.0000% on this
    generator) by >= 5.0 points. A model that reproduced today's ground truth EXACTLY would not
    clear that bar, so tuning the loss toward Gate 2 on under-dense data has no achievable target.
    That finding lived only in a document for a full phase (docs/phase4.2-plan.md §3's rationale
    for why this is a preflight check and not a note) — this makes it a hard stop instead.
    """
    r = Result("P13", "Dataset structure density (multi-run columns)", PASS)
    from mc_imagine_model.scripts.diagnose_overhangs import (
        aggregate_overhang_metrics, compute_overhang_metrics, load_band, resolve_band_paths,
    )
    from mc_imagine_model.training.train import resolve_path
    data_dir = resolve_path(cfg["data"]["data_dir"])
    try:
        paths = resolve_band_paths(data_dir, args.p13_max_regions)
    except (FileNotFoundError, OSError) as exc:
        r.status = FAIL
        r.detail = str(exc)
        return r
    metrics = []
    for p in paths:
        loaded = load_band(p)
        metrics.append(compute_overhang_metrics(loaded.band, heightfield=loaded.heightfield))
    agg = aggregate_overhang_metrics(metrics)
    pct = float(agg["multi_run_column_pct"])
    r.lines.append(f"  sampled {len(paths)} region(s) from {data_dir}")
    r.lines.append(f"  ground-truth multi-run column rate: {pct:.4f}%")
    r.lines.append(f"  floor: --min-multi-run-pct {args.min_multi_run_pct}")
    if pct < args.min_multi_run_pct:
        r.status = FAIL
        r.detail = (
            f"{pct:.4f}% is below the {args.min_multi_run_pct:.1f}% floor. docs/phase4.1-plan.md "
            f"§1.3 / docs/phase4.2-plan.md §1.3: Gate 2 condition 1 needs +5.0 points over a "
            f"0.0000% baseline, so a model that reproduced this ground truth exactly would still "
            f"fail condition 1. Raise density first (docs/phase4.2-plan.md §1.1) — do not tune the "
            f"loss toward this gate on data that cannot clear it."
        )
    return r


# =============================================================================================
# driver
# =============================================================================================

CHECKS = [
    ("P1", p1_install), ("P2", p2_gpu), ("P3", p3_disk), ("P4", p4_host_ram),
    ("P5", p5_shm), ("P6", p6_text_encoder), ("P7", p7_determinism), ("P8", p8_target_range),
    ("P9", p9_p10_smoke), ("P11", p11_checkpoint_roundtrip), ("P12", p12_tests),
    ("P13", p13_dataset_density),
]
SLOW = {"P9", "P11", "P12"}


def main() -> int:
    import yaml
    parser = argparse.ArgumentParser(
        description="Pre-training flight check — run before any long or metered training run.")
    parser.add_argument("--config", required=True, help="The training config you intend to use")
    parser.add_argument("--target-regions", type=int, default=8000,
                        help="Dataset scale the budget checks are computed against (default 8000). "
                             "Deliberately independent of what is on disk.")
    parser.add_argument("--smoke-steps", type=int, default=200, help="Steps for P9/P10 (default 200)")
    parser.add_argument("--budget-hours", type=float, default=48.0,
                        help="P10 fails if the projected full run exceeds this (default 48)")
    parser.add_argument("--quick", action="store_true", help="Skip the timed runs (P9/P10/P11/P12)")
    parser.add_argument("--keep-going", action="store_true", help="Run all checks despite failures")
    parser.add_argument("--skip", default="", help="Comma-separated check ids to skip, e.g. P7,P12")
    parser.add_argument("--record-hashes", action="store_true",
                        help="(Re)write P7's reference shard-hash fixture from this machine")
    parser.add_argument("--min-multi-run-pct", type=float, default=8.0,
                        help="P13 FAILs below this ground-truth multi-run column %% (default 8.0, "
                             "docs/phase4.2-plan.md §1.1)")
    parser.add_argument("--p13-max-regions", type=int, default=200,
                        help="Regions P13 samples from data.data_dir (default 200 — bounds runtime "
                             "independent of full dataset size, like P7's fixed sample)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    skip = {s.strip().upper() for s in args.skip.split(",") if s.strip()}
    if args.quick:
        skip |= SLOW

    print("=" * 92)
    print("Mc-Imagine preflight — docs/remote-training-readiness-plan.md §3")
    print(f"  config          : {args.config}")
    print(f"  target regions  : {args.target_regions}")
    print(f"  machine         : {platform.system()} {platform.machine()}, "
          f"{os.cpu_count()} cpus, {_gb(total_ram_bytes() or 0)} RAM")
    print("=" * 92)

    results: List[Result] = []
    failed = False
    for cid, fn in CHECKS:
        if cid in skip:
            print(f"[SKIP] {cid}  (skipped by request)")
            continue
        try:
            out = fn(cfg, args)
        except Exception as exc:  # a check that cannot run is a failed check, not a crash
            out = Result(cid, fn.__doc__.splitlines()[0] if fn.__doc__ else cid, FAIL,
                         f"check raised {type(exc).__name__}: {exc}")
        for res in (out if isinstance(out, list) else [out]):
            results.append(res)
            marker = {PASS: "PASS", FAIL: "FAIL", SKIP: "SKIP", WARN: "WARN"}[res.status]
            print(f"[{marker}] {res.check}  {res.title}")
            for ln in res.lines:
                print(ln)
            if res.detail:
                prefix = "  -> " if res.status != FAIL else "  !! "
                print(prefix + res.detail.replace("\n", "\n     "))
            print()
            if not res.ok:
                failed = True
        if failed and not args.keep_going:
            print("Stopping at first failure (--keep-going to run everything).")
            break

    print("=" * 92)
    n_fail = sum(1 for r in results if r.status == FAIL)
    n_warn = sum(1 for r in results if r.status == WARN)
    n_pass = sum(1 for r in results if r.status == PASS)
    n_skip = sum(1 for r in results if r.status == SKIP)
    print(f"  {n_pass} passed, {n_warn} warned, {n_skip} skipped, {n_fail} FAILED")
    if n_fail:
        print("  DO NOT START A LONG RUN. Each failure above names the plan section explaining it.")
    else:
        print("  Clear to train. Re-run without --quick on the target machine if you used it here.")
    print("=" * 92)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
