import os
import pytest
import torch
from mc_imagine_model.training.train import train, load_checkpoint

SMALL_TEST_CONFIG = {
    "model": {
        "conv_channels": 32,
        "fusion_hidden": 64,
        "fusion_out": 32,
        "num_conv_layers": 4,
        "num_conv3d_layers": 3,
        "halo": 7,
    },
    "training": {
        "seed": 42,
        "batch_size": 2,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "num_epochs": 10,
        "warmup_steps": 1,
        "log_every": 1,
        "val_every_epochs": 1,
        "checkpoint_dir": "",  # dynamically set to tmp_path
    },
    "data": {
        "data_dir": "data",
        "val_region_fraction": 0.2,
        "chunks_per_region": 4,
        "max_prompt_length": 128,
        "shard_cache_size": 4,
    },
    "hardware": {
        "device": "cpu",
        "num_workers": 0,
        "mixed_precision": False,
    },
}


def test_train_max_steps_creates_best_and_last_checkpoints(tmp_path) -> None:
    """Verifies that running train() with max_steps exits early, runs validation,
    and produces valid best.pt and last.pt checkpoints."""
    checkpoint_dir = os.path.join(str(tmp_path), "checkpoints")
    config = dict(SMALL_TEST_CONFIG)
    config["training"] = dict(SMALL_TEST_CONFIG["training"])
    config["training"]["checkpoint_dir"] = checkpoint_dir

    max_steps = 2
    train(config, max_steps=max_steps)

    best_pt = os.path.join(checkpoint_dir, "best.pt")
    last_pt = os.path.join(checkpoint_dir, "last.pt")

    assert os.path.isfile(best_pt), "best.pt was not created when training exited via max_steps"
    assert os.path.isfile(last_pt), "last.pt was not created when training exited via max_steps"

    # Verify best.pt structure and contents
    ckpt_best = torch.load(best_pt, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt_best
    assert "optimizer_state_dict" in ckpt_best
    assert "scheduler_state_dict" in ckpt_best
    assert "global_step" in ckpt_best
    assert ckpt_best["global_step"] == max_steps
    assert "val_loss" in ckpt_best
    assert isinstance(ckpt_best["val_loss"], float)
    assert not torch.isnan(torch.tensor(ckpt_best["val_loss"]))

    # Verify last.pt structure and contents
    ckpt_last = torch.load(last_pt, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt_last
    assert "optimizer_state_dict" in ckpt_last
    assert "scheduler_state_dict" in ckpt_last
    assert ckpt_last["global_step"] == max_steps

    # Verify load_checkpoint functionality
    from mc_imagine_model.model.imagine_net import ImagineNet
    model = ImagineNet(config["model"])
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    scaler = None
    start_epoch, global_step, best_val = load_checkpoint(
        last_pt, model, optimizer, scheduler, scaler, torch.device("cpu")
    )
    assert global_step == max_steps
