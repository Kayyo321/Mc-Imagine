"""
Training script for the Mc-Imagine model (imaginator-low_intensity-no_structures, docs/poc-plan.md
Phase 5; portability work per docs/phase2-plan.md Phase 4).

This script is expected to run unchanged on three devices — MPS (this Mac, smoke tests), CUDA (the
GPU box that does the real runs) and CPU (fallback/CI) — selected purely by the config file. The
device-specific machinery is confined to four places, all guarded so the non-CUDA paths are exact
no-ops: AMP autocast, the gradient scaler, `pin_memory`, and CUDA RNG seeding. AMP is deliberately
*not* enabled on MPS: autocast there is not reliable for this workload, and a silently-wrong dtype
is worse than a slower run.
"""

import argparse
import glob
import logging
import math
import os
import platform
import random
import shutil
import signal
import time
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from mc_imagine_model.data.dataset import McImagineDataset
from mc_imagine_model.inference_utils import render_area
from mc_imagine_model.model.imagine_net import ImagineNet
from mc_imagine_model.scripts.diagnose_overhangs import compute_overhang_metrics
from mc_imagine_model.spec_constants import (
    GATE_PROMPTS,
    GATE_SEED,
    HEIGHT_MAX,
    HEIGHT_MIN,
)
from mc_imagine_model.training.losses import CombinedLoss
from mc_imagine_model.viz import save_prompt_grid

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# basicConfig() is a no-op if the root logger already has handlers — e.g. under pytest, whose
# logging plugin configures the root logger before this module is ever imported. Set this logger's
# own level explicitly so `logger.info(...)` isn't silently filtered by whatever the root ends up
# at; §2.G2's file handler (added per-run in `train()`) depends on this actually firing.
logger.setLevel(logging.INFO)

_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../mc_imagine_model
MODEL_ROOT = os.path.dirname(os.path.dirname(_PACKAGE_DIR))                 # .../model

# Measured decompressed shard size (docs/remote-training-readiness-plan.md §1.1/§6): band
# (520,520,64) bool + heightfield f32 + 2x u8 maps. Used only to bound the shard cache when
# `data.shard_cache_size` is unset (see `_cache_size` in `train`) — a config change there does not
# require touching this estimate.
SHARD_DECOMPRESSED_BYTES_ESTIMATE = 18_930_000
DEFAULT_SHARD_CACHE_BUDGET_BYTES = 1 * 1024**3  # ~1 GB/process when shard_cache_size is unset
VAL_SHARD_CACHE_SIZE = 4    # val loader runs ~90s once per epoch — no benefit to a large cache
VAL_NUM_WORKERS = 2         # §2.A2 — 8 persistent workers for a once-per-epoch loop is pure waste
# Measured (docs/remote-training-readiness-plan.md §6). Mirrored in scripts/preflight.py's own
# PER_SAMPLE_BYTES/CHECKPOINT_BYTES — kept here too so the startup banner (§2.G2/3, §5 item 1.6)
# doesn't need to import the scripts package.
PER_SAMPLE_BYTES_ESTIMATE = 20_116          # collated tensor bytes/sample, post-B2 (uint8 band)
CHECKPOINT_BYTES_ESTIMATE = 33_077_647      # torch.save size, post-C2 (text_encoder.* excluded)
# docs/phase4.2-plan.md §4.2: if the structure tripwire is still 0.0000% by this global_step, the
# run is a heightfield and will stay one. A module-level constant (not inlined at the call site) so
# a test can monkeypatch it small and observe the WARN without an actual 1000-step run.
STRUCTURE_TRIPWIRE_WARN_STEP = 1000


def resolve_path(path: str) -> str:
    """Resolves a config path against cwd first, then against `model/`.

    Config files are referenced from the repo root (`python -m mc_imagine_model.training.train
    --config model/src/.../config.cuda.yaml`) but their `data_dir`/`checkpoint_dir` values are
    naturally written relative to `model/`. Trying both means the same config works from either
    working directory instead of silently pointing at an empty dataset.
    """
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    candidate = os.path.join(MODEL_ROOT, path)
    if os.path.exists(candidate):
        return candidate
    # Nothing exists yet (e.g. a fresh checkpoint_dir): anchor it under model/ rather than cwd.
    return candidate


def resolve_device(requested: str) -> torch.device:
    """Resolves the configured device, raising rather than downgrading when it is unambiguous.

    docs/remote-training-readiness-plan.md §2.E1: a config that explicitly names "cuda"/"mps" and
    silently downgrades to CPU on failure produces a run that starts, logs normal-looking progress,
    and takes weeks on a rented machine you are paying for by the hour. Only "auto" is allowed to
    fall back quietly — it says "pick whatever's here" by definition.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError(
            "hardware.device: 'mps' but torch.backends.mps.is_available() is False on this "
            "machine. This is a fatal config/hardware mismatch, not a warning — "
            "docs/remote-training-readiness-plan.md §2.E1."
        )
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError(
            "hardware.device: 'cuda' but torch.cuda.is_available() is False on this machine "
            "(cuda build: " + str(torch.version.cuda) + "). This is almost always a CPU-only "
            "torch wheel — training would start, print normal progress, and take weeks on an "
            "idle GPU you are paying for. Reinstall torch with the --index-url matching "
            "`nvidia-smi`'s CUDA version. docs/remote-training-readiness-plan.md §2.E1."
        )
    if requested == "cpu":
        return torch.device("cpu")
    raise RuntimeError(
        f"hardware.device: {requested!r} is not one of 'auto', 'cuda', 'mps', 'cpu'."
    )


def seed_everything(seed: int) -> torch.Generator:
    """Seeds python/numpy/torch identically on every device.

    `torch.manual_seed` alone leaves numpy and the DataLoader's shuffling generator unseeded on some
    paths, so two machines drew different sample orders and their loss curves were not comparable.
    The returned generator is handed to the DataLoader so shuffling is reproducible too, and CUDA's
    RNG is seeded only when CUDA exists (a no-op guard, not a device-specific behaviour change).
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _worker_init_fn(worker_id: int) -> None:
    """Gives every DataLoader worker a distinct-but-deterministic seed (num_workers>0 on CUDA)."""
    base = torch.initial_seed() % (2**32)
    np.random.seed((base + worker_id) % (2**32))
    random.seed(base + worker_id)


def make_lr_lambda(warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def format_eta(seconds: float) -> str:
    """Formats a duration in seconds as e.g. '2h05m' or '47m12s'. 'n/a' if not finite/positive."""
    if not math.isfinite(seconds) or seconds < 0:
        return "n/a"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s"


def _total_ram_bytes() -> Optional[int]:
    """Mirrors scripts/preflight.py's total_ram_bytes() — kept separate to avoid train.py
    importing the scripts package (see PER_SAMPLE_BYTES_ESTIMATE's comment)."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        pass
    if platform.system() == "Darwin":
        try:
            import subprocess
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
        except Exception:
            return None
    return None


def _shm_bytes() -> Optional[int]:
    """Mirrors scripts/preflight.py's shm_bytes(). Linux only — macOS has no /dev/shm."""
    if not os.path.isdir("/dev/shm"):
        return None
    st = os.statvfs("/dev/shm")
    return st.f_blocks * st.f_frsize


def print_resource_banner(
    device: torch.device,
    train_cache_size: int,
    val_cache_size: int,
    num_workers: int,
    val_num_workers: int,
    batch_size: int,
    prefetch_factor: int,
    checkpoint_dir: str,
    num_epochs: int,
    keep_last_n_checkpoints: int,
) -> None:
    """Startup resource banner (docs/remote-training-readiness-plan.md §3/§5 item 1.6).

    Prints the same numbers `scripts/preflight.py`'s P2-P5/P11 compute — most of §2's issues are
    visible in this banner, and printing it here means they are visible on the machine that is
    actually about to pay for a mistake, not only on the Mac ahead of time. Must run before the
    target-range check so it is the first thing on screen, not something scrolled past.
    """
    lines = ["=" * 92, "Mc-Imagine startup resource banner (docs/remote-training-readiness-plan.md §3)"]

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        lines.append(f"  device: {device} -- {props.name}, {props.total_memory / 1000**3:.2f} GB VRAM")
    else:
        lines.append(f"  device: {device}")

    total_ram = _total_ram_bytes()
    lines.append(f"  host RAM: {total_ram / 1000**3:.2f} GB" if total_ram else "  host RAM: unknown")

    shm = _shm_bytes()
    if shm is not None:
        lines.append(f"  /dev/shm: {shm / 1000**3:.2f} GB")
    else:
        lines.append("  /dev/shm: n/a (no /dev/shm on this platform)")

    holders_train = max(1, num_workers)
    holders_val = max(1, val_num_workers)
    cache_budget = (
        (train_cache_size * holders_train + val_cache_size * holders_val + 32)
        * SHARD_DECOMPRESSED_BYTES_ESTIMATE
    )
    pct = f" ({cache_budget / total_ram:.0%} of host RAM)" if total_ram else ""
    lines.append(
        f"  shard cache budget: train {train_cache_size}x{holders_train} + val "
        f"{val_cache_size}x{holders_val} = {cache_budget / 1000**3:.2f} GB{pct}"
    )

    in_flight = batch_size * PER_SAMPLE_BYTES_ESTIMATE * max(1, num_workers) * prefetch_factor
    lines.append(f"  DataLoader in-flight shm requirement: {in_flight / 1000**2:.1f} MB")

    free_disk = shutil.disk_usage(checkpoint_dir).free
    lines.append(f"  free disk at checkpoint_dir: {free_disk / 1000**3:.2f} GB")

    n_ckpt = min(num_epochs, keep_last_n_checkpoints) + 2  # capped epoch_NNN.pt's + best.pt + last.pt
    projected_ckpt = n_ckpt * CHECKPOINT_BYTES_ESTIMATE
    lines.append(
        f"  projected checkpoint total (steady state): {n_ckpt} x "
        f"~{CHECKPOINT_BYTES_ESTIMATE / 1000**2:.1f} MB = {projected_ckpt / 1000**3:.2f} GB"
    )

    lines.append("=" * 92)
    for ln in lines:
        logger.info(ln)


def split_shards_by_region(data_dir: str, val_fraction: float, seed: int = 0):
    shard_paths = sorted(glob.glob(os.path.join(data_dir, "region_*.npz")))
    if not shard_paths:
        raise FileNotFoundError(
            f"No region_*.npz shards found in {data_dir}. Generate the dataset first:\n"
            "  python -m mc_imagine_model.scripts.generate_data --regions 8000 --out model/data"
        )
    rng = random.Random(seed)
    shuffled = shard_paths[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val_paths = sorted(shuffled[:n_val])
    train_paths = sorted(shuffled[n_val:])
    return train_paths, val_paths


def move_batch(batch: Dict[str, Any], device: torch.device, non_blocking: bool = False) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=non_blocking)
        else:
            out[k] = v
    return out


def build_targets(batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    targets = {
        "heightmap": batch["heightmap"],
        "profile_id": batch["profile_id"],
        "water_level": batch["water_level"],
        "biome_grid": batch["biome_grid"],
    }
    # Absent for pre-Phase-4 shard directories; CombinedLoss falls back to a zero occupancy/overhang
    # loss when this is missing, which silently makes occupancy_weight/overhang_weight no-ops — do
    # not drop this key, or a volumetric sweep trains on the pre-Phase-4 objective without warning.
    if "occupancy_band" in batch:
        targets["occupancy_band"] = batch["occupancy_band"]
    return targets


def assert_targets_representable(loader: DataLoader, num_batches: int = 4) -> None:
    """Fails loudly if any height/water target falls outside what `TerrainHead` can output.

    Day 1 trained against a generator that produced terrain up to y=235 while the head was
    `63 + 96*tanh(x)` — a hard ceiling of 159. 14% of regions were unrepresentable, tanh saturated,
    the gradient went to zero and *all* local relief died: the "towering snow-capped peaks" prompt
    came out with height std 0.03. Nothing crashed and no loss looked obviously wrong; it just
    quietly trained a flat world. This check is the tripwire for that entire class of bug, so it
    runs before a single optimizer step and raises rather than warns.
    """
    lo, hi = float(HEIGHT_MIN), float(HEIGHT_MAX)
    seen = 0
    obs_min, obs_max = float("inf"), float("-inf")
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        for key in ("heightmap", "water_level"):
            t = batch[key]
            if not isinstance(t, torch.Tensor):
                continue
            t = t.to(torch.float32)
            tmin, tmax = float(t.min()), float(t.max())
            obs_min, obs_max = min(obs_min, tmin), max(obs_max, tmax)
            if tmin < lo or tmax > hi:
                bad = int(((t < lo) | (t > hi)).sum())
                raise ValueError(
                    f"TARGET RANGE CHECK FAILED: '{key}' in batch {i} has {bad} value(s) outside "
                    f"the model head's representable band [{lo}, {hi}] "
                    f"(observed min={tmin:.2f}, max={tmax:.2f}).\n"
                    "The head squashes height as HEIGHT_CENTER + HEIGHT_AMPLITUDE*tanh(x), so any "
                    "target beyond that band can never be reached: tanh saturates, gradients "
                    "vanish, and the model trains to a flat world without any error being raised "
                    "(this is exactly the Day-1 bug, docs/phase2-plan.md §0).\n"
                    "Fix the data range (data/world_generator.py's TERRAIN_CLIP_MIN/MAX) or widen "
                    "HEIGHT_CENTER/HEIGHT_AMPLITUDE in spec_constants.py — do not disable this check."
                )
        seen += 1
    if seen == 0:
        raise ValueError("Target range check could not read a single batch — is the dataset empty?")
    logger.info(
        "Target range check passed over %d batch(es): targets in [%.2f, %.2f], head band [%.1f, %.1f].",
        seen, obs_min, obs_max, lo, hi,
    )


def assert_text_encoder_loaded(model: nn.Module, allow_random: bool) -> None:
    """Fails loudly if the text encoder is the random fallback, the way `assert_targets_representable`
    fails loudly on out-of-range targets — a second tripwire alongside `PromptEncoder.__init__`'s
    own raise, so the failure is visible in the training log at the same point as the other startup
    checks rather than only in a constructor traceback (docs/remote-training-readiness-plan.md §2.E2).
    """
    loaded_from = getattr(model.text_encoder, "_loaded_from", None)
    if loaded_from is not None:
        logger.info("Text encoder loaded from %s", loaded_from)
        return
    if not allow_random:
        # PromptEncoder.__init__ already raises before this point when allow_random is False —
        # this branch is an extra safety net, not the primary enforcement.
        raise RuntimeError(
            "Text encoder is not loaded from a pretrained checkpoint and "
            "--allow-random-text-encoder was not passed. Refusing to start."
        )
    logger.warning(
        "Text encoder is a RANDOM, non-semantic fallback (--allow-random-text-encoder was passed). "
        "Every prompt will produce similar terrain. This is a deliberately degraded run."
    )


def qualitative_dump(model: nn.Module, device: torch.device, out_path: str, chunk_span: int = 8) -> None:
    was_training = model.training
    model.eval()
    entries = []
    for prompt in GATE_PROMPTS:
        h, p, water = render_area(model, prompt, GATE_SEED, chunk_span=chunk_span, device=str(device))
        entries.append({"caption": prompt, "heightfield": h, "profile_map": p, "water_level": water})
    save_prompt_grid(entries, out_path, title=f"Held-out prompts @ seed {GATE_SEED}")
    if was_training:
        model.train()


# Prefix of state_dict keys excluded from saved checkpoints (§2.C2) and tolerated as "missing" on
# load (§2.J1's same allowlist, mirrored here so save/resume/export all agree on the one gap that
# is safe). LOCKED per docs/remote-training-readiness-plan.md §5 item 1.2: this does not land
# without export_onnx.py's load_imagine_net already raising on any *other* missing/unexpected key
# (§2.J1) and train.py's text encoder fallback already being fatal by default (§2.E2, item 0.6) —
# both are true as of this change, which is the only reason dropping this is safe.
CHECKPOINT_EXCLUDED_PREFIXES = ("text_encoder.",)


def save_checkpoint(path: str, model: nn.Module, optimizer, scheduler, scaler,
                    config: Dict[str, Any], epoch: int, global_step: int,
                    val_loss: float, best_val_loss: float, step_in_epoch: int = 0,
                    loader_generator_state: Optional[torch.Tensor] = None,
                    total_steps: Optional[int] = None) -> None:
    """Saves *everything* needed to resume, not just the weights.

    A multi-hour GPU run that dies at hour 3 must not have to start over, and `export_onnx.py` only
    ever reads `model_state_dict`/`config`, so the extra keys are additive and backwards compatible.

    `step_in_epoch` disambiguates a checkpoint written at a natural epoch boundary (`0` — `epoch` is
    fully complete, resume at `epoch + 1`) from one written mid-epoch (`> 0` — `epoch` is still in
    progress, this many batches of it have already been consumed). Without this, `load_checkpoint`
    could not tell the two apart and always resumed at `epoch + 1`, silently dropping the rest of
    `epoch` on any mid-epoch checkpoint (docs/remote-training-readiness-plan.md §2.G6 — observed
    directly: a run stopped at step 50 of a 575-step epoch resumed at epoch+1, step 50, dropping the
    other 525 steps of that epoch's data with a log line that looked entirely normal).

    `loader_generator_state` (§2.E4) is the shuffling `torch.Generator`'s state as it was BEFORE the
    permutation this checkpoint's `epoch`/`step_in_epoch` needs to reproduce was drawn — the caller
    (`train()`) is responsible for capturing the right snapshot (see its own comments), because
    which one is "right" depends on whether this is a mid-epoch or an epoch-boundary checkpoint.
    Restoring it on resume makes the DataLoader redraw the SAME shuffle order instead of replaying
    whatever a freshly re-seeded generator produces first, which — without this — was always the
    original run's epoch-0 order, regardless of which epoch is actually being resumed into.

    `total_steps` (§2.G4) is the cosine schedule's total step count at save time. `load_checkpoint`
    warns (does not raise) if a resume recomputes a different value — e.g. the dataset size or
    `num_epochs` changed — since that silently shifts the LR curve without anything else looking
    wrong.

    Excludes the frozen MiniLM text encoder's weights (`CHECKPOINT_EXCLUDED_PREFIXES`): it is ~90 MB
    of byte-identical parameters re-saved on every one of ~62 checkpoints in a full run — 5.6 GB of
    pure duplication (§2.C2) — and both `ImagineNet.__init__` (on resume, via `PromptEncoder`) and
    `export_onnx.load_imagine_net` (on export) already reload it from `model/checkpoints/` and raise
    if that reload didn't happen, so nothing reads this copy anyway.

    Writes to a temp file in the same directory, then `os.replace`s it onto `path`: `os.replace` is
    atomic on both POSIX and Windows, so a process killed (or a disk that fills up) mid-`torch.save`
    leaves `path` exactly as it was — never a truncated, unloadable file where a good `last.pt` used
    to be (docs/remote-training-readiness-plan.md §2.C1).
    """
    model_state = {
        k: v for k, v in model.state_dict().items()
        if not k.startswith(CHECKPOINT_EXCLUDED_PREFIXES)
    }
    tmp_path = path + ".tmp"
    torch.save(
        {
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "config": config,
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "global_step": global_step,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "loader_generator_state": loader_generator_state,
            "total_steps": total_steps,
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


def prune_old_epoch_checkpoints(checkpoint_dir: str, keep_last_n: int) -> None:
    """Deletes `epoch_NNN.pt` files beyond the most recent `keep_last_n` (docs/remote-training-
    readiness-plan.md §2.C1: 60 epochs x 123 MB is 7.6 GB against a 20-50 GB rented disk with no
    other margin). `best.pt`/`last.pt` are untouched — they are overwritten in place, not part of
    this rotation.
    """
    if keep_last_n < 0:
        return
    paths = sorted(glob.glob(os.path.join(checkpoint_dir, "epoch_*.pt")))
    excess = len(paths) - keep_last_n
    for old_path in paths[:max(0, excess)]:
        os.remove(old_path)


def load_checkpoint(path: str, model: nn.Module, optimizer, scheduler, scaler,
                    device: torch.device, loader_generator: Optional[torch.Generator] = None,
                    total_steps: Optional[int] = None) -> Tuple[int, int, float, int]:
    """Restores model/optimizer/scheduler/scaler state.

    Returns (start_epoch, global_step, best_val_loss, resume_step_in_epoch). `resume_step_in_epoch`
    is 0 for a checkpoint written at a natural epoch boundary (nothing to skip — `start_epoch` is
    already the next, unstarted epoch) or > 0 for one written mid-epoch, in which case `start_epoch`
    is the SAME (in-progress) epoch the checkpoint was taken in, and the caller must skip that many
    already-consumed batches of that epoch's loader before resuming training (docs/remote-training-
    readiness-plan.md §2.G6 — see `save_checkpoint`'s docstring for the failure this fixes).

    If `loader_generator` is given and the checkpoint carries a saved RNG state (§2.E4), that
    generator's state is restored in place so the resumed run's DataLoader shuffling continues the
    original sequence rather than a freshly re-seeded one (which always reproduces epoch 0's order,
    regardless of which epoch is actually being resumed into — the bug this fixes).

    If `total_steps` is given and differs from what the checkpoint recorded (§2.G4), warns — a
    changed dataset size or `num_epochs` silently shifts the cosine LR curve without anything else
    looking wrong.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"--resume checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # strict=False + explicit inspection, not strict=True: checkpoints written under C2 (§2.C2)
    # omit text_encoder.* on purpose (`model` here was already constructed with it correctly
    # loaded from model/checkpoints/, raising if that failed — §2.E2). Anything else missing or
    # unexpected means this checkpoint does not match the current architecture and must not be
    # silently partially loaded (the same reasoning as export_onnx.load_imagine_net, §2.J1).
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    missing = [k for k in missing if not k.startswith(CHECKPOINT_EXCLUDED_PREFIXES)]
    unexpected = [k for k in unexpected if not k.startswith(CHECKPOINT_EXCLUDED_PREFIXES)]
    if missing or unexpected:
        raise RuntimeError(
            f"--resume checkpoint {path!r} does not match the current model architecture — "
            f"missing={missing}, unexpected={unexpected}. Refusing to partially load it."
        )
    if ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    else:
        logger.warning("Checkpoint has no optimizer state — resuming with a fresh optimizer.")
    if ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    else:
        logger.warning("Checkpoint has no scheduler state — LR schedule restarts from step 0.")
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if loader_generator is not None:
        gen_state = ckpt.get("loader_generator_state")
        if gen_state is not None:
            # `torch.load(..., map_location=device)` moves every tensor in the checkpoint to
            # `device`, including this one — but Generator.set_state() requires a CPU ByteTensor
            # regardless of what device training runs on (the loader's shuffle generator is always
            # CPU-side; only the model/batch tensors move to the accelerator).
            loader_generator.set_state(gen_state.cpu())
            logger.info("Restored DataLoader shuffle RNG state — resumed run continues the original sample order.")
        else:
            logger.warning(
                "Checkpoint has no loader_generator_state — the resumed run will NOT reproduce the "
                "original sample order (docs/remote-training-readiness-plan.md §2.E4)."
            )
    ckpt_total_steps = ckpt.get("total_steps")
    if total_steps is not None and ckpt_total_steps is not None and int(ckpt_total_steps) != int(total_steps):
        logger.warning(
            "total_steps mismatch on resume: checkpoint was trained against %d, this run computes "
            "%d (dataset size or num_epochs changed) — the cosine LR schedule will not match the "
            "original curve (docs/remote-training-readiness-plan.md §2.G4).",
            int(ckpt_total_steps), int(total_steps),
        )
    step_in_epoch = int(ckpt.get("step_in_epoch", 0))
    if step_in_epoch > 0:
        # Mid-epoch checkpoint: resume into the SAME epoch, not the next one.
        start_epoch = int(ckpt.get("epoch", 0))
    else:
        start_epoch = int(ckpt.get("epoch", -1)) + 1
    global_step = int(ckpt.get("global_step", 0))
    best_val_loss = float(ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf"))))
    logger.info(
        "Resumed from %s — starting at epoch %d (skipping %d already-consumed batches), "
        "global_step %d, best_val_loss %.4f",
        path, start_epoch, step_in_epoch, global_step, best_val_loss,
    )
    return start_epoch, global_step, best_val_loss, step_in_epoch


def evaluate_validation(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: CombinedLoss,
    device: torch.device,
    pin_memory: bool,
    use_amp: bool,
) -> Tuple[float, Dict[str, float]]:
    """Runs evaluation over val_loader and returns (val_loss, val_components)."""
    model.eval()
    val_loss_total = 0.0
    val_component_totals: Dict[str, float] = {}
    val_count = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = move_batch(batch, device, non_blocking=pin_memory)
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
                if use_amp else nullcontext()
            )
            with autocast_ctx:
                preds = model(batch["prompt_tokens"], batch["chunk_x"], batch["chunk_z"], batch["seed"])
                targets = build_targets(batch)
                loss, loss_dict = criterion(preds, targets)
            val_loss_total += loss.item()
            for name, value in loss_dict.items():
                val_component_totals[name] = val_component_totals.get(name, 0.0) + value.item()
            val_count += 1
    val_loss = val_loss_total / max(1, val_count)
    val_components = {
        name: total / max(1, val_count) for name, total in val_component_totals.items()
    }
    return val_loss, val_components


def run_validation_and_checkpoint(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: CombinedLoss,
    device: torch.device,
    pin_memory: bool,
    use_amp: bool,
    checkpoint_dir: str,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    config: Dict[str, Any],
    save_epoch_ckpt: bool = True,
    step_in_epoch: int = 0,
    loader_generator_state: Optional[torch.Tensor] = None,
    total_steps: Optional[int] = None,
) -> float:
    """Evaluates validation loss and saves best.pt, last.pt, and optionally epoch_XXX.pt.

    `step_in_epoch`: 0 if `epoch` is fully complete (the normal end-of-epoch call), or the number
    of batches already consumed in `epoch` if this is a mid-epoch checkpoint (the `--max-steps`
    early-exit path, or future periodic mid-epoch saves) — see `save_checkpoint`'s docstring (§2.G6).

    `loader_generator_state`/`total_steps`: see `save_checkpoint`'s docstring (§2.E4/§2.G4) — passed
    straight through to every checkpoint this call writes.

    Returns the updated best_val_loss.
    """
    val_loss, val_components = evaluate_validation(
        model, val_loader, criterion, device, pin_memory, use_amp
    )
    val_components_str = ", ".join(f"{name}={value:.4f}" for name, value in val_components.items())
    logger.info("=== epoch %d val_loss=%.4f (%s) ===", epoch, val_loss, val_components_str)

    if save_epoch_ckpt:
        ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt")
        save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, config,
                        epoch, global_step, val_loss, best_val_loss, step_in_epoch=step_in_epoch,
                        loader_generator_state=loader_generator_state, total_steps=total_steps)
        keep_last_n = int(config.get("training", {}).get("keep_last_n_checkpoints", 3))
        prune_old_epoch_checkpoints(checkpoint_dir, keep_last_n)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(os.path.join(checkpoint_dir, "best.pt"), model, optimizer,
                        scheduler, scaler, config, epoch, global_step, val_loss, best_val_loss,
                        step_in_epoch=step_in_epoch, loader_generator_state=loader_generator_state,
                        total_steps=total_steps)
        logger.info("New best checkpoint (val_loss=%.4f) saved.", val_loss)

    # `last.pt` is what --resume wants after a crash, independent of which epoch was best.
    save_checkpoint(os.path.join(checkpoint_dir, "last.pt"), model, optimizer, scheduler,
                    scaler, config, epoch, global_step, val_loss, best_val_loss,
                    step_in_epoch=step_in_epoch, loader_generator_state=loader_generator_state,
                    total_steps=total_steps)

    return best_val_loss



def train(config: Dict[str, Any], max_steps: Optional[int] = None, resume: Optional[str] = None) -> None:
    train_cfg = config["training"]
    data_cfg = config["data"]
    hw_cfg = config.get("hardware", {})

    seed = train_cfg.get("seed", 1234)
    loader_generator = seed_everything(seed)

    device = resolve_device(hw_cfg.get("device", "auto"))
    is_cuda = device.type == "cuda"
    # AMP is CUDA-only by policy: MPS autocast is unreliable for this workload and CPU autocast buys
    # nothing here. `use_amp` False makes autocast/GradScaler exact no-ops (see below).
    use_amp = bool(hw_cfg.get("mixed_precision", False)) and is_cuda
    if hw_cfg.get("mixed_precision", False) and not is_cuda:
        logger.warning(
            "mixed_precision requested but device is %s — AMP is CUDA-only here, running in fp32.",
            device.type,
        )
    logger.info("Using device: %s (amp=%s)", device, use_amp)
    if is_cuda:
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))

    # Resolved early (not just before the epoch loop) so the file log (§2.G2) captures the whole
    # run from the top, and so the resource banner below (§2.G2/G3, §5 item 1.6) has a real
    # checkpoint_dir to measure free disk against.
    checkpoint_dir = resolve_path(train_cfg.get("checkpoint_dir", "./checkpoints"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info("Checkpoint dir: %s", checkpoint_dir)

    log_file_path = os.path.join(checkpoint_dir, "train.log")
    file_log_handler = logging.FileHandler(log_file_path)
    file_log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(file_log_handler)
    logger.info("Also logging to %s", log_file_path)

    data_dir = resolve_path(data_cfg["data_dir"])
    logger.info("Data dir: %s", data_dir)
    train_paths, val_paths = split_shards_by_region(
        data_dir, data_cfg.get("val_region_fraction", 0.08), seed=seed,
    )
    logger.info("Regions: %d train, %d val", len(train_paths), len(val_paths))

    # shard_cache_size covers every unique shard in the split: with DataLoader shuffle=True drawing
    # samples uniformly across the whole epoch, a shard's ~100 samples are scattered throughout, so
    # a cache smaller than the shard count causes near-100%-miss thrashing (every batch re-decompresses
    # ~64 distinct gzip'd .npz files). Sizing it to the full split means each shard is decompressed
    # exactly once (ever, not once per epoch) and held resident — a few GB, safely within RAM budget.
    # Note: with num_workers>0 each worker holds its own copy of that cache, so shard_cache_size is
    # capped by config to keep total RSS bounded on the GPU box.
    cache_cap = data_cfg.get("shard_cache_size", None)

    def _cache_size(n_paths: int) -> int:
        # `null` no longer means "cache the whole split" — at 8000 regions that resolved to 139 GB
        # (docs/remote-training-readiness-plan.md §2.A3). Bound it by an absolute byte budget
        # instead of a raw shard count so an unset config is safe at any dataset scale.
        if cache_cap is not None:
            want = max(32, n_paths)
            return min(want, cache_cap)
        bounded = max(1, DEFAULT_SHARD_CACHE_BUDGET_BYTES // SHARD_DECOMPRESSED_BYTES_ESTIMATE)
        return min(n_paths, int(bounded))

    train_cache_size = _cache_size(len(train_paths))
    val_cache_size = min(VAL_SHARD_CACHE_SIZE, max(1, len(val_paths)))
    train_ds = McImagineDataset(
        data_dir, shard_paths=train_paths,
        chunks_per_region=data_cfg.get("chunks_per_region", 100),
        max_tokens=data_cfg.get("max_prompt_length", 128), index_seed=1,
        shard_cache_size=train_cache_size,
    )
    # The val loader gets its own (much smaller) cache: it runs once per epoch for ~90s, not
    # continuously, so there is no benefit to sizing it like the train split.
    val_ds = McImagineDataset(
        data_dir, shard_paths=val_paths,
        chunks_per_region=data_cfg.get("chunks_per_region", 100),
        max_tokens=data_cfg.get("max_prompt_length", 128), index_seed=2,
        shard_cache_size=val_cache_size,
    )
    logger.info("Samples: %d train, %d val", len(train_ds), len(val_ds))

    batch_size = train_cfg.get("batch_size", 64)
    num_workers = int(hw_cfg.get("num_workers", 0))
    # pin_memory only means anything for CUDA host->device copies; on MPS/CPU it is wasted work
    # (and on some MPS builds it warns), so it is ANDed with the device rather than passed through.
    pin_memory = bool(hw_cfg.get("pin_memory", False)) and is_cuda
    loader_kwargs: Dict[str, Any] = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(hw_cfg.get("prefetch_factor", 2))
    logger.info(
        "DataLoader: batch_size=%d num_workers=%d pin_memory=%s", batch_size, num_workers, pin_memory
    )

    # Separate kwargs for val: docs/remote-training-readiness-plan.md §2.A2 — the train loader's
    # 8 persistent workers each holding a second full-size cache, to serve 250 batches once per
    # epoch, is pure waste and doubles the OOM exposure for no benefit.
    val_num_workers = min(VAL_NUM_WORKERS, num_workers) if num_workers > 0 else 0
    val_loader_kwargs: Dict[str, Any] = dict(
        num_workers=val_num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        worker_init_fn=_worker_init_fn if val_num_workers > 0 else None,
    )
    if val_num_workers > 0:
        val_loader_kwargs["prefetch_factor"] = int(hw_cfg.get("prefetch_factor", 2))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        generator=loader_generator, **loader_kwargs,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **val_loader_kwargs)

    # --- Structure tripwire setup (docs/phase4.2-plan.md §4.1/§4.2) ---------------------------
    # One fixed validation batch, cached ONCE here and never redrawn, so the tripwire below adds
    # NO new data-loading path (§4.1's explicit constraint) and every log_every reading through
    # the whole run is measuring the exact same chunks. The target side's multi-run % is computed
    # once here too, since it never changes for a fixed batch.
    #
    # NOT simply the first batch: val_loader is shuffle=False, and McImagineDataset orders samples
    # region-by-region, so consecutive batches sit inside ONE region — a region drawn from a flat
    # archetype (desert_dunes, rolling_grassland, ... — carve_strength pinned to 0) has EXACTLY 0%
    # target density by construction. Measured on this dataset: batches 0-3 were 0.0000%, batch 5
    # was 18.87%. A tripwire whose fixed TARGET is 0% can never demonstrate a nonzero PREDICTED
    # reading regardless of model quality, so scan forward (still only consuming val_loader's own
    # existing iterator — no second loader, no extra data path) for the first batch with nonzero
    # target density.
    TRIPWIRE_SCAN_LIMIT = 20
    tripwire_batch: Optional[Dict[str, Any]] = None
    tripwire_target_multirun_pct: Optional[float] = None
    try:
        val_iter = iter(val_loader)
        for scan_i in range(TRIPWIRE_SCAN_LIMIT):
            candidate = next(val_iter)
            if "occupancy_band" not in candidate:
                logger.warning(
                    "Structure tripwire (docs/phase4.2-plan.md §4.1): val batch has no "
                    "'occupancy_band' key (pre-Phase-4 shards?) — disabled for this run."
                )
                break
            cand_np = candidate["occupancy_band"].numpy()
            cand_pct = compute_overhang_metrics(
                cand_np, chunk_grid=(cand_np.shape[0], 1)
            )["multi_run_column_pct"]
            if tripwire_batch is None:
                tripwire_batch, tripwire_target_multirun_pct = candidate, cand_pct  # fallback
            if cand_pct > 0.0:
                tripwire_batch, tripwire_target_multirun_pct = candidate, cand_pct
                logger.info(
                    "Structure tripwire: using val batch %d (target multi-run columns %.4f%%).",
                    scan_i, cand_pct,
                )
                break
        else:
            logger.warning(
                "Structure tripwire (docs/phase4.2-plan.md §4.1): none of the first %d val "
                "batches had nonzero target density — falling back to batch 0 (%.4f%%). The "
                "tripwire will be uninformative until this dataset is denser or the scan limit "
                "is raised.",
                TRIPWIRE_SCAN_LIMIT, tripwire_target_multirun_pct or 0.0,
            )
    except StopIteration:
        logger.warning(
            "Structure tripwire (docs/phase4.2-plan.md §4.1): val_loader produced no batches — "
            "disabled for this run."
        )
    if tripwire_batch is not None:
        # chunk_grid=(B, 1) is used identically at read time below — multi_run_column_pct is a
        # pure per-column statistic (no connectivity across chunks involved), so how the batch is
        # tiled cannot affect it, and forcing a 1-column strip sidesteps compute_overhang_metrics'
        # "batch size must be a perfect square" requirement, which an arbitrary batch_size is not
        # guaranteed to satisfy. `tripwire_target_multirun_pct` was already computed by the scan
        # above; only the device transfer is left to do here.
        tripwire_batch = move_batch(tripwire_batch, device, non_blocking=pin_memory)

    tripwire_warned_step1000 = False

    print_resource_banner(
        device=device,
        train_cache_size=train_cache_size,
        val_cache_size=val_cache_size,
        num_workers=num_workers,
        val_num_workers=val_num_workers,
        batch_size=batch_size,
        prefetch_factor=int(hw_cfg.get("prefetch_factor", 2)),
        checkpoint_dir=checkpoint_dir,
        num_epochs=int(train_cfg.get("num_epochs", 10)),
        keep_last_n_checkpoints=int(train_cfg.get("keep_last_n_checkpoints", 3)),
    )

    # Startup tripwire — must run before any optimizer step (see the function's docstring).
    check_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    assert_targets_representable(
        check_loader, num_batches=int(train_cfg.get("target_range_check_batches", 4))
    )

    model = ImagineNet(config["model"]).to(device)
    assert_text_encoder_loaded(model, allow_random=bool(config["model"].get("allow_random_text_encoder", False)))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info("Model params: %d total, %d trainable", n_total, n_trainable)

    optimizer = torch.optim.AdamW(
        trainable_params, lr=train_cfg.get("learning_rate", 3e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    criterion = CombinedLoss(
        terrain_weight=train_cfg.get("terrain_weight", 1.0),
        biome_weight=train_cfg.get("biome_weight", 1.0),
        slope_weight=train_cfg.get("slope_weight", 1.0),
        relief_weight=train_cfg.get("relief_weight", 0.0),
        occupancy_weight=train_cfg.get("occupancy_weight", 0.0),
        overhang_weight=train_cfg.get("overhang_weight", 0.0),
        consistency_weight=train_cfg.get("consistency_weight", 0.0),
        structure_weight=0.0,
        structure_graph_weight=0.0,
    )

    num_epochs = train_cfg.get("num_epochs", 10)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = train_cfg.get("warmup_steps", 200)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(warmup_steps, total_steps)
    )
    # enabled=False makes this a documented pass-through: scale()/step()/update() become plain
    # forwarding calls, so the identical loop body runs on MPS and CPU with no branching.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_clip = train_cfg.get("grad_clip_norm", 0.0)

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    resume_step_in_epoch = 0
    resume_path = resume or train_cfg.get("resume") or None
    if resume_path:
        start_epoch, global_step, best_val_loss, resume_step_in_epoch = load_checkpoint(
            resolve_path(resume_path), model, optimizer, scheduler, scaler, device,
            loader_generator=loader_generator, total_steps=total_steps,
        )

    # §2.G1: cheap spot/interruptible instances (50-70% of on-demand) can be reclaimed with ~30
    # seconds' notice, and `last.pt` was previously written only at epoch end — an hour or more of
    # lost work at F1's un-fixed data-path rate. `save_every_steps` bounds that loss window; the
    # SIGTERM/SIGINT handler makes a preemption's own notice actually usable. Both write ONLY
    # last.pt via a direct save_checkpoint call — no validation pass, which is exactly the cost
    # `run_validation_and_checkpoint` is not safe to run inside a  ~30-second eviction window.
    save_every_steps = int(train_cfg.get("save_every_steps", 0))
    # Reassigned once per epoch, right before that epoch's DataLoader iteration begins (see the
    # loop below) — §2.E4's snapshot of loader_generator's state as it was BEFORE the CURRENT
    # epoch's shuffle permutation was drawn, so a mid-epoch checkpoint can restore it and have the
    # resumed run redraw the SAME permutation (then skip forward, §2.G6), rather than whatever a
    # freshly re-seeded generator produces.
    epoch_start_generator_state: Optional[torch.Tensor] = None

    def _flush_last_checkpoint(epoch: int, step_in_epoch: int, reason: str) -> None:
        logger.info(
            "%s — writing %s/last.pt at epoch %d, global_step %d, step_in_epoch %d.",
            reason, checkpoint_dir, epoch, global_step, step_in_epoch,
        )
        save_checkpoint(
            os.path.join(checkpoint_dir, "last.pt"), model, optimizer, scheduler, scaler,
            config, epoch, global_step, val_loss=best_val_loss, best_val_loss=best_val_loss,
            step_in_epoch=step_in_epoch, loader_generator_state=epoch_start_generator_state,
            total_steps=total_steps,
        )

    shutdown_requested = {"flag": False}

    def _handle_shutdown_signal(signum, frame) -> None:
        shutdown_requested["flag"] = True
        logger.warning(
            "Received signal %s — will checkpoint and exit after the current step "
            "(docs/remote-training-readiness-plan.md §2.G1).",
            signal.Signals(signum).name,
        )

    prev_sigterm_handler = signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    prev_sigint_handler = signal.signal(signal.SIGINT, _handle_shutdown_signal)

    t_start = time.time()
    # §2.G3: no throughput/ETA logging meant "did the loss look sane" was the only signal for the
    # first hour or two of a metered run — not "are we on track for 20 hours or 200". Rolling
    # (since-last-log-line), not cumulative-average, so it reflects current speed, not a startup
    # hiccup diluted over the whole run.
    rate_t_prev = t_start
    rate_step_prev = global_step

    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            epoch_t0 = time.time()
            running_loss = 0.0
            running_count = 0
            # §2.G6: a mid-epoch checkpoint's resume skips the batches this epoch already consumed,
            # instead of always starting the loader from its beginning — which is what silently
            # dropped the rest of the epoch's data on every mid-epoch resume before this fix. Only
            # applies to the FIRST epoch after a resume; every later epoch in this run starts at 0.
            skip_n = resume_step_in_epoch if epoch == start_epoch else 0
            # §2.E4 — captured BEFORE train_loader's iter() call below draws this epoch's
            # permutation, so a mid-epoch checkpoint written during this epoch can restore this
            # exact snapshot and redraw the identical permutation on resume.
            epoch_start_generator_state = loader_generator.get_state()
            train_epoch_iter = enumerate(train_loader)
            if skip_n > 0:
                for _ in range(skip_n):
                    next(train_epoch_iter, None)
                logger.info(
                    "Resuming mid-epoch %d: skipping %d of %d already-consumed batches.",
                    epoch, skip_n, steps_per_epoch,
                )
            for step, batch in train_epoch_iter:
                batch = move_batch(batch, device, non_blocking=pin_memory)
                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
                    if use_amp else nullcontext()
                )
                with autocast_ctx:
                    preds = model(batch["prompt_tokens"], batch["chunk_x"], batch["chunk_z"], batch["seed"])
                    targets = build_targets(batch)
                    loss, loss_dict = criterion(preds, targets)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                running_loss += loss.item()
                running_count += 1
                global_step += 1

                if step % train_cfg.get("log_every", 50) == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    components_str = ", ".join(
                        f"{name}={value.item():.4f}" for name, value in loss_dict.items()
                    )
                    now = time.time()
                    dt = now - rate_t_prev
                    d_steps = global_step - rate_step_prev
                    if dt > 0 and d_steps > 0:
                        steps_per_sec = d_steps / dt
                        samples_per_sec = steps_per_sec * batch_size
                        eta = format_eta((total_steps - global_step) / steps_per_sec)
                        rate_str = f"{steps_per_sec:.2f} steps/s ({samples_per_sec:.0f} samples/s) ETA {eta}"
                    else:
                        rate_str = "rate n/a (warming up)"
                    rate_t_prev, rate_step_prev = now, global_step
                    logger.info(
                        "epoch %d step %d/%d loss=%.4f (%s) lr=%.6g | %s",
                        epoch, step, steps_per_epoch,
                        loss.item(),
                        components_str,
                        lr,
                        rate_str,
                    )

                    # --- Structure tripwire (docs/phase4.2-plan.md §4.1/§4.2) ------------------
                    # "See failure in 2 minutes instead of 35": without this, a run that collapsed
                    # to the heightfield optimum (§1.1) looks identical to a healthy one in the
                    # loss log above until a full diagnose_overhangs.py pass is run against a
                    # checkpoint. Same fixed batch every time (cached above, never redrawn).
                    if tripwire_batch is not None:
                        model.eval()
                        with torch.no_grad():
                            tw_preds = model(
                                tripwire_batch["prompt_tokens"], tripwire_batch["chunk_x"],
                                tripwire_batch["chunk_z"], tripwire_batch["seed"],
                            )
                        model.train()
                        # Threshold at 0.5 probability, per §4.1 — compute_overhang_metrics does
                        # this itself when handed a float array, so sigmoid first and let it.
                        tw_pred_probs = torch.sigmoid(tw_preds["occupancy_logits"]).cpu().numpy()
                        tw_pred_metrics = compute_overhang_metrics(
                            tw_pred_probs, chunk_grid=(tw_pred_probs.shape[0], 1)
                        )
                        tw_pred_pct = tw_pred_metrics["multi_run_column_pct"]
                        logger.info(
                            "  structure tripwire: predicted multi-run columns %.4f%% "
                            "(target on this fixed batch: %.4f%%)",
                            tw_pred_pct, tripwire_target_multirun_pct,
                        )
                        if global_step >= STRUCTURE_TRIPWIRE_WARN_STEP and not tripwire_warned_step1000:
                            tripwire_warned_step1000 = True
                            if tw_pred_pct == 0.0:
                                logger.warning(
                                    "Structure tripwire (docs/phase4.2-plan.md §4.2): predicted "
                                    "multi-run columns is still 0.0000%% at step %d. This run is "
                                    "a heightfield and will stay one — see §4.2 before letting it "
                                    "continue.",
                                    global_step,
                                )

                if max_steps is not None and global_step >= max_steps:
                    logger.info("Reached max_steps=%d, running validation and saving checkpoints before exit.", max_steps)
                    best_val_loss = run_validation_and_checkpoint(
                        model, val_loader, criterion, device, pin_memory, use_amp,
                        checkpoint_dir, epoch, global_step, best_val_loss,
                        optimizer, scheduler, scaler, config, save_epoch_ckpt=False,
                        step_in_epoch=step + 1,
                        loader_generator_state=epoch_start_generator_state, total_steps=total_steps,
                    )
                    return

                if shutdown_requested["flag"]:
                    _flush_last_checkpoint(epoch, step + 1, "Shutdown signal received")
                    return

                if save_every_steps > 0 and global_step % save_every_steps == 0:
                    _flush_last_checkpoint(epoch, step + 1, f"save_every_steps={save_every_steps} reached")

            train_loss = running_loss / max(1, running_count)
            epoch_time = time.time() - epoch_t0
            logger.info("=== epoch %d done: train_loss=%.4f, time=%.1fs ===", epoch, train_loss, epoch_time)

            if (epoch + 1) % train_cfg.get("val_every_epochs", 1) == 0:
                # Natural epoch boundary: the generator has, by now, been advanced by exactly this
                # epoch's own draw and nothing else — its CURRENT (live) state is precisely what
                # the next epoch's iter() call needs to draw ITS permutation correctly (§2.E4).
                best_val_loss = run_validation_and_checkpoint(
                    model, val_loader, criterion, device, pin_memory, use_amp,
                    checkpoint_dir, epoch, global_step, best_val_loss,
                    optimizer, scheduler, scaler, config, save_epoch_ckpt=True,
                    loader_generator_state=loader_generator.get_state(), total_steps=total_steps,
                )

            if (epoch + 1) % train_cfg.get("qualitative_dump_every_epochs", 1) == 0:
                dump_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}_qualitative.png")
                try:
                    qualitative_dump(model, device, dump_path)
                    logger.info("Wrote qualitative dump: %s", dump_path)
                except Exception:
                    logger.exception("Qualitative dump failed (non-fatal), continuing training.")

        logger.info("Training complete in %.1f min.", (time.time() - t_start) / 60.0)
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm_handler)
        signal.signal(signal.SIGINT, prev_sigint_handler)
        logging.getLogger().removeHandler(file_log_handler)
        file_log_handler.close()


def main() -> None:
    """
    Main training function.
    Parses configuration, sets up dataset, model, optimizer, and runs the training loop.
    """
    parser = argparse.ArgumentParser(description="Train the Mc-Imagine model")
    # Required on purpose: there is no longer a single config.yaml, and defaulting to either
    # config.mps.yaml or config.cuda.yaml would mean one of the two machines silently trains with
    # the wrong device/AMP/worker settings.
    parser.add_argument("--config", type=str, required=True,
                        help="Path to training config (training/config.mps.yaml or config.cuda.yaml)")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after N steps (smoke test)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint to resume from (model+optimizer+scheduler+epoch/step)")
    parser.add_argument("--relief-weight", type=float, default=None,
                        help="Override relief_weight specified in config")
    parser.add_argument("--occupancy-weight", type=float, default=None,
                        help="Override occupancy_weight specified in config")
    parser.add_argument("--overhang-weight", type=float, default=None,
                        help="Override overhang_weight specified in config")
    parser.add_argument("--consistency-weight", type=float, default=None,
                        help="Override consistency_weight specified in config")
    parser.add_argument("--allow-random-text-encoder", action="store_true",
                        help="Accept a randomly-initialized, non-semantic text encoder when the "
                             "pretrained MiniLM checkpoint is missing, instead of refusing to "
                             "start (docs/remote-training-readiness-plan.md §2.E2). Off by default.")
    args = parser.parse_args()

    print(f"Starting training with config: {args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.relief_weight is not None:
        config["training"]["relief_weight"] = args.relief_weight
    if args.occupancy_weight is not None:
        config["training"]["occupancy_weight"] = args.occupancy_weight
    if args.overhang_weight is not None:
        config["training"]["overhang_weight"] = args.overhang_weight
    if args.consistency_weight is not None:
        config["training"]["consistency_weight"] = args.consistency_weight
    if args.allow_random_text_encoder:
        config.setdefault("model", {})["allow_random_text_encoder"] = True

    train(config, max_steps=args.max_steps, resume=args.resume)


if __name__ == "__main__":
    main()
