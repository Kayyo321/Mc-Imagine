import os
import signal

import pytest
import torch
import mc_imagine_model.training.train as train_module
from mc_imagine_model.training.train import (
    load_checkpoint,
    prune_old_epoch_checkpoints,
    save_checkpoint,
    train,
)

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
    start_epoch, global_step, best_val, step_in_epoch = load_checkpoint(
        last_pt, model, optimizer, scheduler, scaler, torch.device("cpu")
    )
    assert global_step == max_steps


def test_resume_mid_epoch_stays_in_same_epoch_not_epoch_plus_one(tmp_path) -> None:
    """docs/remote-training-readiness-plan.md §2.G6 / §5 item 1.3.

    Before this fix, `load_checkpoint` always resumed at `epoch + 1` regardless of whether the
    checkpoint was written at a natural epoch boundary or mid-epoch — silently dropping the rest of
    the epoch's data on every mid-epoch resume (observed directly via preflight P11: a run stopped
    at step 50 of a 575-step epoch resumed at epoch 1, step 50). `SMALL_TEST_CONFIG`'s dataset has
    hundreds of steps per epoch, so `--max-steps 5` then `--max-steps 10` both land well inside
    epoch 0 — resuming must stay in epoch 0, not jump to epoch 1.
    """
    checkpoint_dir = os.path.join(str(tmp_path), "checkpoints")
    config = dict(SMALL_TEST_CONFIG)
    config["training"] = dict(SMALL_TEST_CONFIG["training"])
    config["training"]["checkpoint_dir"] = checkpoint_dir

    train(config, max_steps=5)
    last_pt = os.path.join(checkpoint_dir, "last.pt")
    ckpt = torch.load(last_pt, map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 0
    assert ckpt["global_step"] == 5
    assert ckpt["step_in_epoch"] == 5

    train(config, max_steps=10, resume=last_pt)
    ckpt2 = torch.load(last_pt, map_location="cpu", weights_only=False)
    assert ckpt2["epoch"] == 0, "mid-epoch resume must stay in the SAME epoch, not jump to epoch+1"
    assert ckpt2["global_step"] == 10
    assert ckpt2["step_in_epoch"] == 10


def test_resume_does_not_replay_the_same_sample_order(tmp_path, monkeypatch) -> None:
    """docs/remote-training-readiness-plan.md §2.E4 / §5 item 1.7 — the literal acceptance test.

    Before this fix, `train()` always built a fresh `torch.Generator` from the config's seed and
    never restored its advanced state on resume — so whichever epoch a run was resumed into always
    replayed epoch 0's original shuffle order, not its own. This proves the opposite: a run resumed
    at a natural epoch boundary into epoch 1 sees the SAME epoch-1 sample order an uninterrupted run
    would have produced, and NOT epoch 0's order.
    """
    STEPS_PER_EPOCH = 20  # 320 train regions (400 x 0.8) x chunks_per_region=1 / batch_size=16

    def make_config(num_epochs: int):
        cfg = dict(SMALL_TEST_CONFIG)
        cfg["training"] = dict(SMALL_TEST_CONFIG["training"])
        cfg["data"] = dict(SMALL_TEST_CONFIG["data"])
        cfg["data"]["chunks_per_region"] = 1
        cfg["training"]["batch_size"] = 16
        cfg["training"]["num_epochs"] = num_epochs
        cfg["training"]["log_every"] = 1000  # quiet; not what this test checks
        return cfg

    def capture_batch_order(cfg, checkpoint_dir, resume_path=None):
        """Records each TRAINING step's chunk_x tuple, in order. Distinguishes training steps from
        validation steps (which also call move_batch) by hooking optimizer.zero_grad, which only
        the training loop ever calls."""
        cfg = dict(cfg)
        cfg["training"] = dict(cfg["training"])
        cfg["training"]["checkpoint_dir"] = checkpoint_dir

        calls = []
        last_batch_chunk_x = {"v": None}
        real_move_batch = train_module.move_batch
        real_zero_grad = torch.optim.AdamW.zero_grad

        def _spy_move_batch(batch, device, non_blocking=False):
            last_batch_chunk_x["v"] = tuple(batch["chunk_x"].tolist())
            return real_move_batch(batch, device, non_blocking=non_blocking)

        def _spy_zero_grad(self, *args, **kwargs):
            calls.append(last_batch_chunk_x["v"])
            return real_zero_grad(self, *args, **kwargs)

        with monkeypatch.context() as m:
            m.setattr(train_module, "move_batch", _spy_move_batch)
            m.setattr(torch.optim.AdamW, "zero_grad", _spy_zero_grad)
            train_module.train(cfg, resume=resume_path)
        return calls

    # Reference: one uninterrupted 2-epoch run.
    ref_config = make_config(num_epochs=2)
    ref_dir = os.path.join(str(tmp_path), "reference")
    ref_calls = capture_batch_order(ref_config, ref_dir)
    assert len(ref_calls) == 2 * STEPS_PER_EPOCH
    ref_epoch0, ref_epoch1 = ref_calls[:STEPS_PER_EPOCH], ref_calls[STEPS_PER_EPOCH:]
    assert ref_epoch0 != ref_epoch1, "test setup assumption broken: epochs 0 and 1 should shuffle differently"

    # Stop after epoch 0 completes naturally, then resume a SEPARATE process into epoch 1.
    stop_config = make_config(num_epochs=1)
    stop_dir = os.path.join(str(tmp_path), "stopped")
    capture_batch_order(stop_config, stop_dir)
    last_pt = os.path.join(stop_dir, "last.pt")
    ckpt = torch.load(last_pt, map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 0 and ckpt["step_in_epoch"] == 0, "must be a natural-boundary checkpoint"

    resumed_dir = os.path.join(str(tmp_path), "resumed")
    resumed_calls = capture_batch_order(make_config(num_epochs=2), resumed_dir, resume_path=last_pt)
    assert len(resumed_calls) == STEPS_PER_EPOCH  # only epoch 1 runs in this resumed process

    assert resumed_calls == ref_epoch1, (
        "resumed epoch 1 must see the SAME sample order as an uninterrupted run's epoch 1"
    )
    assert resumed_calls != ref_epoch0, (
        "resumed epoch 1 must NOT replay epoch 0's order — the bug this fixes"
    )


def test_resource_banner_appears_before_target_range_check(tmp_path) -> None:
    """docs/remote-training-readiness-plan.md §3/§5 item 1.6.

    The startup resource banner must appear before the target-range tripwire (the first thing a
    human or a `tail -f` sees), and must carry the fields §2 names: device, host RAM, /dev/shm,
    shard cache budget, in-flight shm requirement, free disk, and projected checkpoint total.
    """
    checkpoint_dir = os.path.join(str(tmp_path), "checkpoints")
    config = dict(SMALL_TEST_CONFIG)
    config["training"] = dict(SMALL_TEST_CONFIG["training"])
    config["training"]["checkpoint_dir"] = checkpoint_dir

    train(config, max_steps=1)

    log_path = os.path.join(checkpoint_dir, "train.log")
    with open(log_path) as f:
        log_lines = f.readlines()

    banner_idx = next(i for i, ln in enumerate(log_lines) if "startup resource banner" in ln)
    range_check_idx = next(i for i, ln in enumerate(log_lines) if "Target range check passed" in ln)
    assert banner_idx < range_check_idx, "banner must appear before the target-range check"

    banner_text = "".join(log_lines[banner_idx:range_check_idx])
    for expected in ("device:", "host RAM:", "/dev/shm:", "shard cache budget:",
                    "DataLoader in-flight shm requirement:", "free disk at checkpoint_dir:",
                    "projected checkpoint total"):
        assert expected in banner_text, f"banner missing {expected!r}: {banner_text}"


def test_file_log_and_throughput_eta_on_every_step_line(tmp_path) -> None:
    """docs/remote-training-readiness-plan.md §2.G2/G3 / §5 item 1.5.

    A file log must exist next to the checkpoints after a run (an ssh drop shouldn't lose the only
    copy of the log), and every step-progress line must carry a throughput rate and an ETA — the
    number needed to tell, early, whether a metered run is on track for 20 hours or 200.
    """
    checkpoint_dir = os.path.join(str(tmp_path), "checkpoints")
    config = dict(SMALL_TEST_CONFIG)
    config["training"] = dict(SMALL_TEST_CONFIG["training"])
    config["training"]["checkpoint_dir"] = checkpoint_dir
    config["training"]["log_every"] = 1

    train(config, max_steps=6)

    log_path = os.path.join(checkpoint_dir, "train.log")
    assert os.path.isfile(log_path), "train.log must exist in checkpoint_dir after a run"
    with open(log_path) as f:
        log_text = f.read()
    assert log_text.strip(), "train.log must not be empty"

    step_lines = [ln for ln in log_text.splitlines() if " step " in ln and "loss=" in ln]
    assert len(step_lines) >= 6, f"expected at least 6 step lines, got: {step_lines}"
    for ln in step_lines:
        assert "steps/s" in ln, f"step line missing throughput: {ln!r}"
        assert "ETA" in ln, f"step line missing ETA: {ln!r}"


def test_structure_tripwire_logs_every_log_every_and_warns_at_threshold(tmp_path, monkeypatch) -> None:
    """docs/phase4.2-plan.md §4.1/§4.2.

    §4.1: every `log_every` steps, a fixed validation batch's predicted vs. target multi-run
    column % must be logged. §4.2: a one-line WARN must fire, naming this section, the first time
    `global_step` reaches the threshold with the predicted metric still at 0.0000% — which is what
    a freshly-initialized model always reports (real init produces no multi-run structure), so this
    test monkeypatches the threshold down to 2 rather than actually running 1000 steps.
    """
    monkeypatch.setattr(train_module, "STRUCTURE_TRIPWIRE_WARN_STEP", 2)

    checkpoint_dir = os.path.join(str(tmp_path), "checkpoints")
    config = dict(SMALL_TEST_CONFIG)
    config["training"] = dict(SMALL_TEST_CONFIG["training"])
    config["training"]["checkpoint_dir"] = checkpoint_dir
    config["training"]["log_every"] = 1

    train_module.train(config, max_steps=3)

    log_path = os.path.join(checkpoint_dir, "train.log")
    with open(log_path) as f:
        log_text = f.read()

    tripwire_lines = [ln for ln in log_text.splitlines() if "structure tripwire: predicted" in ln]
    assert len(tripwire_lines) >= 3, f"expected a tripwire line every log_every step, got: {log_text}"
    for ln in tripwire_lines:
        assert "target on this fixed batch:" in ln, f"tripwire line missing target: {ln!r}"

    warn_lines = [ln for ln in log_text.splitlines() if "docs/phase4.2-plan.md §4.2" in ln]
    assert len(warn_lines) == 1, (
        f"expected exactly one §4.2 WARN line once global_step reached the threshold, got: "
        f"{warn_lines}"
    )
    assert "still 0.0000%" in warn_lines[0]


def test_save_every_steps_writes_last_pt_only_no_validation(tmp_path, monkeypatch) -> None:
    """docs/remote-training-readiness-plan.md §2.G1 / §5 item 1.4.

    `save_every_steps` must write `last.pt` periodically, mid-epoch, without running a validation
    pass — that cost is exactly what makes it unsafe to do inside a ~30-second preemption window.
    Spies on the module-level `save_checkpoint` to observe every call `train()` makes.
    """
    checkpoint_dir = os.path.join(str(tmp_path), "checkpoints")
    config = dict(SMALL_TEST_CONFIG)
    config["training"] = dict(SMALL_TEST_CONFIG["training"])
    config["training"]["checkpoint_dir"] = checkpoint_dir
    config["training"]["save_every_steps"] = 3

    calls = []
    real_save_checkpoint = train_module.save_checkpoint

    def _spy(path, model, optimizer, scheduler, scaler, cfg, epoch, global_step, val_loss,
            best_val_loss, step_in_epoch=0, loader_generator_state=None, total_steps=None):
        calls.append((os.path.basename(path), global_step, step_in_epoch, val_loss))
        return real_save_checkpoint(path, model, optimizer, scheduler, scaler, cfg, epoch,
                                    global_step, val_loss, best_val_loss, step_in_epoch=step_in_epoch,
                                    loader_generator_state=loader_generator_state, total_steps=total_steps)

    monkeypatch.setattr(train_module, "save_checkpoint", _spy)
    train_module.train(config, max_steps=10)

    # Periodic saves fire at global_step 3, 6, 9 — all mid-epoch (step_in_epoch > 0, not a natural
    # epoch boundary), all writing only last.pt, all before the first real validation ever ran
    # (val_loss is still the untouched +inf sentinel — no evaluate_validation call happened yet).
    periodic = [c for c in calls if c[1] in (3, 6, 9)]
    assert len(periodic) == 3, f"expected exactly 3 periodic saves at steps 3/6/9: {calls}"
    for name, global_step, step_in_epoch, val_loss in periodic:
        assert name == "last.pt", f"periodic save wrote {name}, not last.pt"
        assert step_in_epoch == global_step, "periodic save must be mid-epoch, not epoch-boundary"
        assert val_loss == float("inf"), "periodic save must not run a validation pass"


def test_sigterm_mid_epoch_produces_resumable_last_pt(tmp_path, monkeypatch) -> None:
    """docs/remote-training-readiness-plan.md §2.G1 / §5 item 1.4 — the literal acceptance test:
    a SIGTERM mid-epoch must flush a resumable `last.pt`, not lose the run. Sends the signal to
    this process from inside a real training step (simulating an operator's `kill -TERM` or a spot
    instance's preemption notice) rather than via a subprocess, so the test is fast and
    deterministic; the code path exercised — `train()`'s own signal handler — is identical either
    way.
    """
    checkpoint_dir = os.path.join(str(tmp_path), "checkpoints")
    config = dict(SMALL_TEST_CONFIG)
    config["training"] = dict(SMALL_TEST_CONFIG["training"])
    config["training"]["checkpoint_dir"] = checkpoint_dir
    config["training"]["save_every_steps"] = 0  # isolate: only the signal handler should act here

    # LambdaLR.__init__ calls self.step() once during construction (to set the initial LR),
    # before the training loop's own per-batch calls begin — so the 4th *training-loop* call is
    # the 5th call overall.
    steps_seen = {"n": 0}
    orig_step = torch.optim.lr_scheduler.LambdaLR.step

    def _step_and_kill_at_step_4(self, *args, **kwargs):
        result = orig_step(self, *args, **kwargs)
        steps_seen["n"] += 1
        if steps_seen["n"] == 5:
            os.kill(os.getpid(), signal.SIGTERM)
        return result

    monkeypatch.setattr(torch.optim.lr_scheduler.LambdaLR, "step", _step_and_kill_at_step_4)
    # No --max-steps: the run must exit via the signal handler, not by hitting a step budget —
    # if the handler didn't fire, this would run the whole (much longer) first epoch instead.
    train_module.train(config, max_steps=None)

    last_pt = os.path.join(checkpoint_dir, "last.pt")
    assert os.path.isfile(last_pt), "SIGTERM must flush a last.pt before train() returns"
    ckpt = torch.load(last_pt, map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 0
    assert ckpt["step_in_epoch"] == 4
    assert ckpt["global_step"] == 4

    # Resumable: a fresh train() call picks it back up in the SAME epoch at the right step.
    train_module.train(config, max_steps=6, resume=last_pt)
    ckpt2 = torch.load(last_pt, map_location="cpu", weights_only=False)
    assert ckpt2["epoch"] == 0
    assert ckpt2["global_step"] == 6


def _tiny_trainable_bits():
    """A small model/optimizer/scheduler triple, just enough to call save_checkpoint directly."""
    from mc_imagine_model.model.imagine_net import ImagineNet

    model = ImagineNet(SMALL_TEST_CONFIG["model"])
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    return model, optimizer, scheduler


def test_save_checkpoint_atomic_write_preserves_last_pt_on_crash(tmp_path, monkeypatch) -> None:
    """docs/remote-training-readiness-plan.md §2.C1 / §5 item 1.1.

    `torch.save` onto a full disk (or a process killed mid-write) used to leave a truncated,
    unloadable file exactly where a good `last.pt` used to be — destroying the resume path with the
    same event that made resuming necessary. `save_checkpoint` now writes to `path + ".tmp"` and
    `os.replace`s it onto `path`, which is atomic. This test simulates the crash directly: a first,
    good checkpoint is written, then a second `save_checkpoint` call is made to fail partway through
    `torch.save` (as a mid-write kill or ENOSPC would), and the original file must be untouched and
    still loadable.
    """
    model, optimizer, scheduler = _tiny_trainable_bits()
    path = os.path.join(str(tmp_path), "last.pt")

    save_checkpoint(path, model, optimizer, scheduler, None, {}, epoch=0, global_step=10,
                    val_loss=1.0, best_val_loss=1.0)
    assert os.path.isfile(path)
    good_bytes = open(path, "rb").read()

    real_torch_save = torch.save

    def _crash_partway(obj, f, *args, **kwargs):
        # Simulate a kill/ENOSPC mid-write: some bytes land on disk, then it dies.
        with open(f, "wb") as fh:
            fh.write(b"not a real checkpoint, the process died here")
        raise OSError("simulated crash mid-torch.save")

    monkeypatch.setattr(torch, "save", _crash_partway)
    with pytest.raises(OSError):
        save_checkpoint(path, model, optimizer, scheduler, None, {}, epoch=1, global_step=20,
                        val_loss=2.0, best_val_loss=1.0)
    monkeypatch.setattr(torch, "save", real_torch_save)

    # The tmp file may be left behind (that's fine — it's a crash), but `path` itself must be
    # exactly what it was before the failed save, and must still load.
    assert os.path.isfile(path)
    assert open(path, "rb").read() == good_bytes
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["global_step"] == 10


def test_prune_old_epoch_checkpoints_keeps_last_n(tmp_path) -> None:
    """docs/remote-training-readiness-plan.md §2.C1 / §5 item 1.1: only the most recent
    `keep_last_n` epoch_NNN.pt files survive; best.pt/last.pt (not matched by the glob) are
    untouched."""
    ckpt_dir = str(tmp_path)
    for i in range(6):
        open(os.path.join(ckpt_dir, f"epoch_{i:03d}.pt"), "wb").write(b"x")
    open(os.path.join(ckpt_dir, "best.pt"), "wb").write(b"x")
    open(os.path.join(ckpt_dir, "last.pt"), "wb").write(b"x")

    prune_old_epoch_checkpoints(ckpt_dir, keep_last_n=3)

    remaining = sorted(os.listdir(ckpt_dir))
    assert remaining == ["best.pt", "epoch_003.pt", "epoch_004.pt", "epoch_005.pt", "last.pt"]
