"""
Training script for the Mc-Imagine model (imaginator-low_intensity-no_structures, docs/poc-plan.md
Phase 5).
"""

import argparse
import logging
import math
import os
import time
from typing import Any, Dict, List

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from mc_imagine_model.data.dataset import McImagineDataset
from mc_imagine_model.inference_utils import render_area
from mc_imagine_model.model.imagine_net import ImagineNet
from mc_imagine_model.spec_constants import GATE_PROMPTS, GATE_SEED
from mc_imagine_model.training.losses import CombinedLoss
from mc_imagine_model.viz import save_prompt_grid

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def resolve_device(requested: str) -> torch.device:
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested not in ("cpu", "auto") and requested not in ("mps", "cuda"):
        logger.warning("Unknown device %r, falling back to cpu", requested)
    if requested in ("mps", "cuda"):
        logger.warning("%s requested but not available, falling back to cpu", requested)
    return torch.device("cpu")


def make_lr_lambda(warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def split_shards_by_region(data_dir: str, val_fraction: float, seed: int = 0):
    import glob
    import random

    shard_paths = sorted(glob.glob(os.path.join(data_dir, "region_*.npz")))
    rng = random.Random(seed)
    shuffled = shard_paths[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val_paths = sorted(shuffled[:n_val])
    train_paths = sorted(shuffled[n_val:])
    return train_paths, val_paths


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def build_targets(batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    return {
        "heightmap": batch["heightmap"],
        "profile_id": batch["profile_id"],
        "water_level": batch["water_level"],
        "biome_grid": batch["biome_grid"],
    }


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


def train(config: Dict[str, Any], max_steps: int = None) -> None:
    torch.manual_seed(config["training"].get("seed", 1234))

    device = resolve_device(config["hardware"].get("device", "auto"))
    logger.info("Using device: %s", device)

    data_cfg = config["data"]
    train_paths, val_paths = split_shards_by_region(
        data_cfg["data_dir"], data_cfg.get("val_region_fraction", 0.08),
        seed=config["training"].get("seed", 1234),
    )
    logger.info("Regions: %d train, %d val", len(train_paths), len(val_paths))

    # shard_cache_size covers every unique shard in the split: with DataLoader shuffle=True drawing
    # samples uniformly across the whole epoch, a shard's ~100 samples are scattered throughout, so
    # a cache smaller than the shard count causes near-100%-miss thrashing (every batch re-decompresses
    # ~64 distinct gzip'd .npz files). Sizing it to the full split means each shard is decompressed
    # exactly once (ever, not once per epoch) and held resident — a few GB, safely within RAM budget.
    train_ds = McImagineDataset(
        data_cfg["data_dir"], shard_paths=train_paths,
        chunks_per_region=data_cfg.get("chunks_per_region", 100),
        max_tokens=data_cfg.get("max_prompt_length", 128), index_seed=1,
        shard_cache_size=max(32, len(train_paths)),
    )
    val_ds = McImagineDataset(
        data_cfg["data_dir"], shard_paths=val_paths,
        chunks_per_region=data_cfg.get("chunks_per_region", 100),
        max_tokens=data_cfg.get("max_prompt_length", 128), index_seed=2,
        shard_cache_size=max(32, len(val_paths)),
    )
    logger.info("Samples: %d train, %d val", len(train_ds), len(val_ds))

    train_cfg = config["training"]
    batch_size = train_cfg.get("batch_size", 64)
    num_workers = config["hardware"].get("num_workers", 0)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        drop_last=True, persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )

    model = ImagineNet(config["model"]).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info("Model params: %d total, %d trainable", n_total, n_trainable)

    optimizer = torch.optim.AdamW(
        trainable_params, lr=train_cfg.get("learning_rate", 3e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    criterion = CombinedLoss(structure_weight=0.0, structure_graph_weight=0.0)

    num_epochs = train_cfg.get("num_epochs", 10)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = train_cfg.get("warmup_steps", 200)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(warmup_steps, total_steps)
    )

    checkpoint_dir = train_cfg.get("checkpoint_dir", "./checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    global_step = 0
    best_val_loss = float("inf")
    t_start = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_t0 = time.time()
        running_loss = 0.0
        running_count = 0
        for step, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            preds = model(batch["prompt_tokens"], batch["chunk_x"], batch["chunk_z"], batch["seed"])
            targets = build_targets(batch)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            running_count += 1
            global_step += 1

            if step % train_cfg.get("log_every", 50) == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    "epoch %d step %d/%d loss=%.4f lr=%.6g", epoch, step, steps_per_epoch,
                    loss.item(), lr,
                )

            if max_steps is not None and global_step >= max_steps:
                logger.info("Reached max_steps=%d, stopping early (smoke test mode).", max_steps)
                return

        train_loss = running_loss / max(1, running_count)
        epoch_time = time.time() - epoch_t0
        logger.info("=== epoch %d done: train_loss=%.4f, time=%.1fs ===", epoch, train_loss, epoch_time)

        if (epoch + 1) % train_cfg.get("val_every_epochs", 1) == 0:
            model.eval()
            val_loss_total = 0.0
            val_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = move_batch(batch, device)
                    preds = model(batch["prompt_tokens"], batch["chunk_x"], batch["chunk_z"], batch["seed"])
                    targets = build_targets(batch)
                    loss = criterion(preds, targets)
                    val_loss_total += loss.item()
                    val_count += 1
            val_loss = val_loss_total / max(1, val_count)
            logger.info("=== epoch %d val_loss=%.4f ===", epoch, val_loss)

            ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt")
            torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch,
                        "val_loss": val_loss}, ckpt_path)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch,
                            "val_loss": val_loss}, os.path.join(checkpoint_dir, "best.pt"))
                logger.info("New best checkpoint (val_loss=%.4f) saved.", val_loss)

        if (epoch + 1) % train_cfg.get("qualitative_dump_every_epochs", 1) == 0:
            dump_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}_qualitative.png")
            try:
                qualitative_dump(model, device, dump_path)
                logger.info("Wrote qualitative dump: %s", dump_path)
            except Exception:
                logger.exception("Qualitative dump failed (non-fatal), continuing training.")

    logger.info("Training complete in %.1f min.", (time.time() - t_start) / 60.0)


def main() -> None:
    """
    Main training function.
    Parses configuration, sets up dataset, model, optimizer, and runs the training loop.
    """
    parser = argparse.ArgumentParser(description="Train the Mc-Imagine model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to training config")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after N steps (smoke test)")
    args = parser.parse_args()

    print(f"Starting training with config: {args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train(config, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
