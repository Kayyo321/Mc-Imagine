# Mc-Imagine Remote Training Readiness Plan — Making a Rented GPU Box a Safe Bet

**Status:** findings + fix plan. One deliverable is **implemented**: the §3 preflight harness now
exists as `model/src/mc_imagine_model/scripts/preflight.py` and runs today. Everything in §2 is
diagnosis; no fix in §5 has been applied.
**Written:** 2026-08-05, against `main` @ `01f3688` (Phase 4.1).
**Trigger:** the first attempt to run the full training on a separate machine crashed repeatedly
with an out-of-memory error and never made real progress.

**Goal:** identify *every* class of problem that would complicate training on a machine other than
the development Mac — including the ones that have not been hit yet — so that renting an external
GPU box is a priced, bounded decision rather than a gamble. The reported OOM is diagnosed in §1;
§2 is the full issue register, most of which has never been observed because the code has only
ever run at rehearsal scale (400 regions) on an M-series Mac.

**How to use this document:** §1 and §2 are the findings. §3 is the preflight harness that must
exist *before* the meter starts. §4 is the machine spec and budget. §5 is the sequencing, with the
go/no-go conditions that let you press "rent" with a straight face. §6 is the measurement log —
every number quoted in this document, with the command that produced it, so none of it has to be
taken on faith.

**One thing this plan deliberately does not do:** it does not relitigate Phase 4.1. The occupancy
head still has no verdict (`docs/phase4.1-plan.md` §2.2's confirmatory run has not been executed).
That is tracked here only as §2.H3, a scheduling constraint on when to rent — not as work this
document owns.

---

## 0. The short version

Five things are true at once, and only the first one has actually bitten you:

1. **The shipped `config.cuda.yaml` asks for 77–155 GB of host RAM.** Not "a lot" — a specific,
   computable number that no rentable box has. This is the crash you hit. §1.
2. **The second crash is already loaded and waiting:** the DataLoader needs ~284 MB of `/dev/shm`,
   and every containerised GPU rental (RunPod, Vast.ai, Lambda, plain `docker run`) defaults to
   64 MB. It presents as `Bus error` / `DataLoader worker (pid …) is killed by signal`, which reads
   like a hardware fault and is not one. §2.B1.
3. **The disk budget does not fit either.** 60 per-epoch checkpoints × 123 MB + a 7.3 GB dataset +
   CUDA wheels ≈ 25–30 GB against a typical 20–50 GB container disk, and it runs out at an epoch
   boundary hours in, leaving a truncated `.pt`. §2.C.
4. **The 9.15-hour runtime figure recorded in `config.cuda.yaml` is stale.** It was measured before
   Phase 4 added the `band` array, which made every shard 12× heavier on the data path — and the
   data path was already the bottleneck ("GPU sat at 30–60% while all CPU threads were pinned
   decompressing .npz shards"). Budget wall-clock accordingly. §2.F1.
5. **`ConsistencyLoss` is mathematically degenerate and is on at weight 0.25 in all three shipped
   configs.** It does not measure what its docstring says it measures. This is not a portability
   problem; it is a "you will pay for GPU hours and get a Gate-2-shaped failure again" problem.
   §2.H1.

Fix order: **A1 → B1/B2 → C1 → H1 → the preflight harness (§3) → rent.** Everything else is either
cheap to fold in along the way or is a documented risk you accept knowingly.

---

## 1. The OOM you hit — root cause

### 1.1 The arithmetic

`training/train.py:387-404` sizes the dataset's shard LRU cache like this:

```python
cache_cap = data_cfg.get("shard_cache_size", None)      # 512 in config.cuda.yaml

def _cache_size(n_paths: int) -> int:
    want = max(32, n_paths)
    return min(want, cache_cap) if cache_cap else want
```

and hands the result to **both** `McImagineDataset` instances. `data/dataset.py:116-155` populates
that cache with the *fully decompressed* shard:

| array | shape | dtype | size |
|---|---|---|---|
| `band` | `(520, 520, 64)` | `bool` | **17.31 MB** |
| `heightfield` | `(520, 520)` | `float32` | 1.08 MB |
| `profile_map` | `(520, 520)` | `uint8` | 0.27 MB |
| `biome_map` | `(520, 520)` | `uint8` | 0.27 MB |
| | | | **18.93 MB / shard** |

`shard_cache_size: 512` × 18.93 MB = **9.69 GB per process holding a cache.**

The comment at `train.py:385-386` says the cap exists precisely because "with `num_workers>0` each
worker holds its own copy of that cache". It does — but the cap was not lowered when Phase 4 added
`band`. Before Phase 4 a shard was ~1.6 MB and 512 of them was ~0.8 GB per worker, which is why
this was survivable then and is not now.

Processes that end up holding a 512-entry cache under `config.cuda.yaml`:

| process | count | cache |
|---|---|---|
| train DataLoader workers (`num_workers: 8`, `persistent_workers=True`) | 8 | 9.69 GB each |
| **val** DataLoader workers — same `loader_kwargs` dict is reused at `train.py:428` | 8 | 9.69 GB each |
| main process (fills ~21 shards via the `check_loader` at `train.py:431`) | 1 | ~0.4 GB |

**Worst case: 155 GB. Realistic case once both loaders have been iterated once: ~78 GB and
climbing.** There is no rentable single-GPU instance with that much RAM. The process is killed by
the kernel OOM killer, `train.py` has no handler, the shell script restarts it, and it dies again
at the same point — exactly the "continuous crashes, no real progress" you described.

### 1.2 Why it never showed up on the Mac

Everything that has ever been run locally used `--regions 400`, where the train split is 368 shards
and `_cache_size(368)` returns 368, not 512 — but more importantly `config.mps.yaml` sets
`num_workers: 0`, so there is exactly **one** cache, and 368 × 18.93 MB ≈ 7 GB fits in a 32/64 GB
Mac. Two independent multipliers (worker count, shard count) are both pinned to their smallest
values on the only machine this has ever run on.

### 1.3 The cache is not even buying anything

With `shuffle=True` drawing uniformly across 7 360 train shards and a 512-entry LRU, the steady-state
hit rate is ≈ **7 %**. 93 % of 736 000 samples per epoch trigger a full 18.9 MB gzip decompression.
So the 78 GB is being spent to avoid one decompression in fourteen. The comment at `train.py:380-384`
("each shard is decompressed exactly once (ever, not once per epoch) and held resident — a few GB")
describes the `shard_cache_size: null` case, which is now 139 GB, not "a few GB".

### 1.4 Fix

**Immediate (unblocks, ~30 minutes of work):**

- `config.cuda.yaml`: `data.shard_cache_size: 8`. Budget becomes 8 × 8 × 18.93 MB ≈ 1.2 GB across
  the train workers. Throughput loss is ~7 %, because that is all the cache was delivering.
- `train.py`: give the val loader its own kwargs — `num_workers: 2`, `persistent_workers=False`,
  `shard_cache_size` 4. Eight persistent processes each holding 9.7 GB for a loop that runs for
  ~90 seconds once per epoch is pure waste.
- `train.py`: log resident set size and the per-process cache budget at startup, next to the
  existing `DataLoader:` line, so the number is visible before it is fatal.

**Structural (recommended before the full run; makes the cache worth having again):**

- **Bit-pack `band` along y in `world_generator.generate_shards`** (`np.packbits(band, axis=2)` →
  `(520, 520, 8)` `uint8`), unpack the 16×16 chunk slice in `dataset._load_shard`. The decompressed
  shard drops **18.93 MB → 2.16 MB (8.8×)**. A 512-entry cache then costs 1.1 GB/process instead of
  9.7 GB, decompression time drops proportionally (§2.F1), and the dataset version guard already in
  `_load_shard` gives you a clean place to reject old shards. This one change fixes A1's magnitude,
  most of F1's cost, and does not touch the training objective.
- Optional, larger: pre-extract the 736 000 samples into flat memmapped `.npy` files (~2.4 GB packed
  total). Removes the per-worker cache concept entirely and lets the OS page cache do the work. Do
  this only if the packed-band change proves insufficient.

**Verification:** run the full 8000-region config for 200 steps with `/usr/bin/time -v` (or
`resource.getrusage`) and assert peak RSS < 12 GB. This becomes preflight check P4 (§3).

---

## 2. Issue register

Severity: **BLOCKER** = will stop the run. **HIGH** = will cost hours or money. **MEDIUM** = will
cost some hours or silently degrade the result. **LOW** = worth folding in while you are there.

### A. Host RAM

| # | Sev | Issue |
|---|---|---|
| **A1** | BLOCKER | Per-process shard cache at 512 entries × 18.93 MB × up to 17 processes = 78–155 GB. **This is the crash you hit.** Full diagnosis and fix in §1. |
| **A2** | HIGH | `train.py:412-428` builds `val_loader` from the *same* `loader_kwargs` as `train_loader`, so validation gets 8 workers with `persistent_workers=True`. Eight processes sit resident for the whole run, each holding a second full-size cache, to serve 250 batches once per epoch. Fix: separate kwargs, `num_workers: 2`, non-persistent. |
| **A3** | MEDIUM | `config.mps.yaml` sets `data.shard_cache_size: null`, which `_cache_size` reads as "cache the entire split". At 8000 regions that is 7 360 × 18.93 MB = **139 GB**. Nothing prevents someone on a rented CPU/Linux box from picking that config (it is the one with the friendlier name). Fix: make `null` mean a bounded default, not unbounded, and cap `_cache_size` by an absolute byte budget rather than a shard count. |
| **A4** | MEDIUM | `model/training_runs/gate2_sweep_artifacts/measure_relief_retention.py:46` hardcodes `shard_cache_size=len(val_paths)` — 640 shards = **12.1 GB** in one process. This is a post-training evaluation tool, so it OOMs *after* the expensive part, on the small box you rented. Same pattern to audit in the other `gate2_sweep_artifacts/` scripts and in `scripts/diagnose_*.py`. |
| **A5** | LOW | Once A1 is fixed, the 7.3 GB dataset comfortably fits in page cache and the I/O problem largely evaporates — but only on a box with ≥ 16 GB free after the worker caches. This is an argument for the RAM floor in §4, not a code change. |

### B. Shared memory (`/dev/shm`) — the crash you have not hit yet

| # | Sev | Issue |
|---|---|---|
| **B1** | BLOCKER (predicted) | PyTorch DataLoader workers return batches through shared memory. At `batch_size: 256`, `num_workers: 8`, `prefetch_factor: 2` there are up to 16 batches in flight at **17.7 MB each = 284 MB**, plus the pin-memory thread's staging buffers. Docker's default `--shm-size` is **64 MB**, and every major GPU rental (RunPod, Vast.ai, Lambda, Paperspace) runs your job in a container. The failure mode is `RuntimeError: DataLoader worker (pid N) is killed by signal: Bus error` or `unable to write to file </torch_…>` — it looks like failing hardware and is not. It would have been the *next* crash after A1. Fix: none in code — this is an operator requirement. Launch with `--shm-size=8g` or `--ipc=host`, and add a startup assertion in `train.py` that reads the size of `/dev/shm` and refuses to start if it is under 1 GB while `num_workers > 0`. |
| **B2** | HIGH | `data/dataset.py:223` emits `occupancy_band` as **float32**, from a source array that is `bool` on disk. That single field is **95 % of every batch's bytes** (64 KiB of a 67.6 KiB sample). Emitting it as `uint8` and letting `OccupancyLoss`/`OverhangLoss` do the existing `.float()` cast on-GPU cuts batch size 67.6 → 20.2 KiB: 4× less shm pressure, 4× less pinned-memory traffic, 4× less PCIe per step. It is a one-line dataset change; both loss classes already call `.float()` on the target (`losses.py:252, 273`). |

### C. Disk

| # | Sev | Issue |
|---|---|---|
| **C1** | HIGH | Measured checkpoint size: **123.4 MB**. `run_validation_and_checkpoint` writes `epoch_XXX.pt` every epoch (`train.py:332-335`) plus `best.pt` and `last.pt`. 60 epochs = **7.6 GB of checkpoints**. Add the 7.3 GB dataset, ~8–10 GB of CUDA-enabled torch wheels, the 87 MB MiniLM, and per-epoch PNGs, and you need **25–30 GB** against the 20–50 GB container disk most rentals ship. It runs out at an epoch boundary hours in — and `torch.save` onto a full filesystem leaves a **truncated, unloadable** `.pt` where `last.pt` should be, so the resume path is destroyed by the same event. Fix: (a) a `keep_last_n_checkpoints` setting defaulting to 3, (b) write to a temp file and `os.replace` so a partial write can never clobber a good `last.pt`, (c) preflight free-space check (§3, P3). |
| **C2** | MEDIUM | Every checkpoint serialises the **frozen** 22.5 M-parameter MiniLM (`n_total 25.3 M` vs `n_trainable 2.75 M`). That is ~90 MB of byte-identical weights written 62 times — **5.6 GB of pure duplication**, and it is the dominant term in C1. Fix: exclude `text_encoder.*` from `model_state_dict` and reload it from `model/checkpoints/` on resume; `export_onnx.py` already knows how to find the vendored checkpoint. Checkpoints drop 123 MB → ~33 MB and C1 mostly disappears. |
| **C3** | LOW | No free-space check anywhere, and the qualitative PNGs (`train.py:548`) accumulate without bound. |

### D. GPU memory

| # | Sev | Issue |
|---|---|---|
| **D1** | HIGH | Estimated peak VRAM at `batch_size: 256`, fp32: **~6–8 GB** of stored activations. Dominated by the 2D conv stack (~8.2 MB/sample: six valid 3×3 convs at 192 channels over 32²→22²) and `Occupancy3DHead` (~8 MB/sample: the `[B,32,16,22,22]` 3-D stack plus the `[B,32,64,16,16]` transpose-conv output), plus a ~500 MB transient for MiniLM's `[256,12,128,128]` attention scores. Comfortable on ≥24 GB (RTX 4090, A10, L4, A5000, A100). **Marginal on 16 GB** (T4, V100) once cuDNN workspaces and allocator fragmentation are counted. **Will OOM on 12 GB.** This is a *different* OOM from §1 and would present as `torch.cuda.OutOfMemoryError`, not a kernel kill. |
| **D2** | MEDIUM | `TRAINING.md`'s troubleshooting row says "lower `batch_size` 256 → 128 → 64" — correct, but `learning_rate: 0.0006` is documented as "scaled with the 4× larger batch vs `config.mps.yaml`". Halving the batch without halving the LR silently changes the optimisation regime, and this project has a documented history of silently-wrong training. Fix: derive LR from batch size, or make the doc say both numbers move together. |
| **D3** | MEDIUM | The per-epoch qualitative dump (`train.py:547-553`) runs six prompts × a 64-chunk batch on-device on top of the live training allocator pool. It is wrapped in `try/except` so it cannot kill the run — good — but on a marginal card it will fail every epoch and you get no PNGs, which are explicitly "the real signal, more than the loss number". Fix: run it on CPU, or `torch.cuda.empty_cache()` first, or drop `chunk_span` to 4. |
| **D4** | LOW | AMP is deliberately off (fp16 overflow killed the 2026-07-28 run — the reasoning in `config.cuda.yaml` is sound and should not be reverted). But **TF32** is neither enabled nor mentioned; on Ampere-and-later it is a near-free ~2–3× matmul/conv speedup with far better numerics than fp16. Consider `torch.backends.cuda.matmul.allow_tf32 = True` + `torch.backends.cudnn.allow_tf32 = True`, and `torch.backends.cudnn.benchmark = True` (shapes are fixed apart from the last val batch). Treat as an experiment with a measured before/after, not a blind flip. |

### E. Silently wrong, or silently slow

These are the ones that cost you the whole rental without raising an error. This project has been
bitten by exactly this class twice already (`docs/phase2-plan.md` §0).

| # | Sev | Issue |
|---|---|---|
| **E1** | HIGH | `resolve_device` (`train.py:67-82`) downgrades a `device: "cuda"` request to CPU with `logger.warning` and continues. A CPU-only torch wheel — the single most common install mistake on a fresh Linux box, and the exact thing `requirements.txt` warns about at length — therefore produces a run that starts, prints normal-looking progress, and takes weeks. On a rented machine you are paying for a GPU that is idle. Fix: if the config explicitly names a device (not `"auto"`) and it is unavailable, **raise**. A warning is the wrong severity when the config is unambiguous. |
| **E2** | HIGH | `PromptEncoder.__init__` (`text_encoder.py:198-208`) falls back to a randomly-initialised frozen encoder when `model/checkpoints/` is absent, with a `print` and no exception. `model/checkpoints/` is gitignored, so **a fresh clone is exactly one forgotten `bootstrap` away from this**. The run trains, converges, exports, and has learned nothing about text — every prompt produces the same terrain. Fix: make the fallback opt-in behind an explicit `--allow-random-text-encoder` flag; default to raising. Then assert `model.text_encoder._loaded_from is not None` at training startup, next to the existing target-range tripwire. |
| **E3** | MEDIUM | `_load_shard`'s stale-shard guards (`dataset.py:122-151`: band presence, band shape, biome-id range, height band) are `assert` statements, stripped by `python -O`/`PYTHONOPTIMIZE`. Some container images set `PYTHONOPTIMIZE=1`. These are the guards that make "mixing generations silently trains a confused model" impossible. Fix: convert to explicit `raise`. |
| **E4** | MEDIUM | `load_checkpoint` (`train.py:243-267`) restores model/optimizer/scheduler/scaler but **not** the DataLoader's RNG state. A resumed run replays the same sample order from the start of the epoch. Given that resume is the documented recovery path for a multi-hour run, and that this run *will* be resumed (see G1), this quietly changes what the model sees. Fix: persist `loader_generator.get_state()` and the epoch's step offset. |
| **E5** | LOW | `generate_data.py`'s docstring claims "the same `--seed` and `--regions` on any machine produce byte-comparable shards", and `TRAINING.md` repeats it as the basis for "two machines following these steps get byte-identical inputs". This has never been checked across platforms — the generator uses `np.float64` FBM throughout, which is stable, but the claim is load-bearing and untested. Fix: preflight P6 (§3) — hash the first 8 shards on both machines and compare. |

### F. Throughput and cost

| # | Sev | Issue |
|---|---|---|
| **F1** | HIGH | **The stale-benchmark problem.** `config.cuda.yaml` records "Measured 9.15 h for 60 epochs in fp32" and "this workload is dataloader-bound, not GPU-bound (GPU sat at 30–60 % while all CPU threads were pinned decompressing .npz shards)". That measurement predates Phase 4's `band` array, which took the decompressed shard from ~1.6 MB to 18.9 MB — **a 12× increase on the exact resource that was already the bottleneck**. Measured: 11.1 ms to fully decompress one shard. At a 7 % cache hit rate that is **684 800 decompressions per epoch = 2.1 core-hours of zlib per epoch** and ~627 GB read through the miss path per epoch. On 8 workers that alone is ~16 min/epoch of pure decompression, i.e. **~16 h across 60 epochs before any GPU work**. The 9.15 h figure must be treated as void until re-measured. The bit-packing fix in §1.4 cuts this ~8×. |
| **F2** | MEDIUM | MiniLM re-encodes the same captions on **every step**. There are at most 7 360 unique captions (one per region, assigned once in `dataset.__init__`), and 172 500 training steps × 256 samples. At ~2.8 GMAC/sample forward, MiniLM is roughly **30 % of total training FLOPs** and the source of the ~500 MB attention transient (D1) — all of it recomputing 7 360 fixed vectors. Fix: precompute the `[n_regions, 384]` embedding table once at startup (one pass, seconds) and index it; keep the live encoder path behind a flag for export parity. Cuts GPU time ~30 % and VRAM meaningfully. |
| **F3** | **HIGH** | `generate_shards` (`world_generator.py:1897-1933`) is a serial `for` loop, and `generate_data.py` wraps it in a `ThreadPoolExecutor(max_workers=1)` purely for progress reporting. Cost is **bimodal by archetype** and much worse than a single sample suggests: regions with `carve_strength > 0` (`valley_mountains_craters`, `taiga_forest`, `snow_peaks`, `deep_canyon`, `mesa_plateaus` — the 3-D `_fbm3d` carve over `[520,520,64]`) take **~8.0 s**; uncarved regions take **0.33 s**. Measured over 12 consecutive regions: 5/12 carve, **mean 3.55 s/region**. That is **~7.9 hours of serial generation for 8000 regions**, before a single training step — roughly a third of the whole rental, against `TRAINING.md`'s claim of "tens of minutes". Fix: parallelise with `ProcessPoolExecutor` over region ids (regions are independent — each draws from `_region_rng(seed, rx, rz)` — so the determinism contract is preserved and P7 verifies it), which takes it to ~35 min on 16 cores; **or** generate once locally and `rsync` the 7.3 GB up. Doing neither costs ~8 h of metered time per run. |
| **F4** | LOW | `CombinedLoss.forward` (`losses.py:365-402`) computes `loss_occ`, `loss_ovh`, `loss_con` and the standalone `loss_s` **unconditionally**, then only adds them if their weight is > 0. `OverhangLoss` alone allocates ~10 tensors of `[B,64,16,16]` plus `cummax` index tensors. Wasted whenever a weight is 0 — which is the whole point of the sweep configs. |

### G. Crash resilience and preemption

| # | Sev | Issue |
|---|---|---|
| **G1** | HIGH | `last.pt` is written **only at epoch end**. At an estimated ~20–30 min/epoch that is a tolerable loss window; at F1's un-fixed rate it is over an hour. More importantly, the cheapest GPU rentals are **spot/interruptible** and can be reclaimed with ~30 seconds' notice. Without mid-epoch checkpointing you cannot safely use spot pricing, which is typically 50–70 % of on-demand. Fix: `save_every_steps` (e.g. 500) writing `last.pt` only, plus a `SIGTERM`/`SIGINT` handler that flushes a checkpoint before exiting. This single change is what makes the cheap instance tier usable. |
| **G2** | MEDIUM | `logging.basicConfig` writes to stderr only (`train.py:42`). An ssh drop loses the entire log unless the operator remembered `tmux`/`nohup`. For a run whose primary diagnostic is "did the loss components look sane at step 3000", that is a real loss. Fix: add a `FileHandler` into `checkpoint_dir`, and mention `tmux` in `TRAINING.md`'s step 4. |
| **G3** | MEDIUM | No throughput or ETA logging. The `log_every: 50` line prints loss and LR but no steps/sec and no projected finish. You cannot tell at hour one whether the run is on track for 20 hours or 200 — which is precisely the decision you need to make early on a metered machine. Fix: log samples/sec, a rolling steps/sec, and a projected completion time on every log line. |
| **G4** | LOW | On resume, `total_steps` for the cosine schedule is recomputed from the *current* `len(train_loader) * num_epochs` (`train.py:458-464`). Resuming with a different dataset size or epoch count silently changes the LR curve mid-run. Fix: persist `total_steps` in the checkpoint and warn on mismatch. |
| **G5** | LOW | No handling for a transient CUDA error or a single bad batch; any exception in the training loop terminates the run. Given that checkpoints exist, this is survivable — but only once G1 makes the loss window small. |
| **G6** | **HIGH — and it gates G1** | **Resuming a mid-epoch checkpoint silently skips the rest of that epoch.** `save_checkpoint` records the *current* `epoch`, and `load_checkpoint` (`train.py:260`) does `start_epoch = ckpt["epoch"] + 1`. That is correct for a checkpoint written at an epoch boundary and wrong for any written mid-epoch. Observed directly by preflight check P11: a run stopped at step 50 of a 575-step epoch 0 resumed at **epoch 1, global_step 50** — 525 steps of training data silently dropped, with a `Resumed from …` line that looks entirely normal. Today this only affects the `--max-steps` path. **But G1 proposes mid-epoch checkpointing, which would make this fire on every preemption recovery** — turning the fix for one problem into a silent data-skip on every restart. Fix: persist `step_in_epoch` alongside `epoch`, resume into the same epoch, and skip forward within the loader (or accept the partial epoch explicitly and log it). **Implement G6 before or with G1, never G1 alone.** |

### H. Correctness landmines that would waste the rented hours

These are not portability bugs. They are listed because the purpose of this plan is "rent with
confidence", and renting a GPU to re-run a broken objective is the most expensive failure available.

| # | Sev | Issue |
|---|---|---|
| **H1** | HIGH | **`ConsistencyLoss` (`losses.py:298-318`) does not compute what its docstring claims, and its target is degenerate.** It returns `MSE(pred_height/S, (pred_height.detach() + pred_surface)/S)`. The residual is `(pred_height − pred_height.detach() − pred_surface)/S`, so the loss is minimised when `pred_surface → 0`. With `BAND_BOTTOM_OFFSET = −61`, `pred_surface = −61 + soft_topmost`, so this term **pushes `soft_topmost` toward 61** — the top of the 64-block band — regardless of the data, i.e. toward "solid all the way up". It is on at `consistency_weight: 0.25` in **all three** shipped configs, including the one Gate 2's sweep used. This is a live candidate contributor to Gate 2's uniform 0.0000 % result and must be resolved *before* Gate 3 buys any hours. |
| **H2** | MEDIUM | Related: `soft_topmost` is documented as the "soft-argmax of topmost solid cell" but is computed as `(probs * y).sum(1) / probs.sum(1)` — an occupancy-weighted **centre of mass**. For a column solid from y=0..k it returns ≈ k/2, not k. Even with H1's target fixed, the quantity being matched is wrong by roughly a factor of two. |
| **H3** | HIGH (scheduling) | `docs/phase4.1-plan.md` §2.2's confirmatory run has **not been executed**, so Gate 2 still has no verdict and Gate 3 is correctly not started. Renting for Gate 3 before Gate 2 passes means paying for the full 8000-region run to rediscover a rehearsal-scale failure. The confirmatory run is 400 regions / 3 000 steps and runs on the Mac. **Do that first, locally, for free.** |
| **H4** | LOW | `CombinedLoss` computes `loss_s` (`losses.py:370-372`) and reports it in `components_dict`, but never adds it to `total_loss` — `TerrainLoss` already includes an identical slope term internally. Harmless, but it means the logged `slope=` value is not a component of the loss being optimised, which is confusing when reading a 60-epoch log. |

### I. Environment and install

| # | Sev | Issue |
|---|---|---|
| **I1** | HIGH | The torch wheel must match the box's CUDA runtime. `requirements.txt` documents this well (cu124/cu126/cu128, Blackwell needs cu128+) and `TRAINING.md` repeats the `torch.cuda.is_available()` check. The gap is that nothing **enforces** it — see E1. Also note `torch>=2.4` has no upper bound; a future 3.x could break `torch.amp.GradScaler("cuda", …)` or `torch.load(weights_only=False)`. Pin an upper bound (`torch>=2.4,<3`) once you know the box's wheel. |
| **I2** | MEDIUM | `bootstrap.py` defaults to `--revision main` for the HuggingFace download. The upstream repo can change under you, which would silently alter the frozen encoder between the machine that trains and the machine that exports. Fix: pin a commit SHA as the default revision, and record it in the `.mcim` manifest. |
| **I3** | MEDIUM | Two large gitignored directories (`model/checkpoints/`, `model/data/`) must be recreated. `bootstrap` needs outbound network + HF availability; some rental images have restrictive egress. Fix: preflight P1/P2 (§3), and have a fallback plan (scp the 87 MB checkpoint directory up). |
| **I4** | MEDIUM | Version pins (`numpy==2.3.2`, `scipy==1.18.0`, `transformers==5.14.1`, `tokenizers==0.22.2`) were resolved against macOS / Python 3.13. Wheel availability for these exact versions on the box's Python (often 3.10/3.11 in CUDA base images) is not guaranteed. `pyproject.toml` says `requires-python = ">=3.11"`. Fix: preflight P1 does a clean `pip install` in a fresh venv and fails loudly, before anything else. |
| **I5** | LOW | On Windows (and anywhere using the `spawn` start method) the `Dataset` object is **pickled** to each worker. By the time workers spawn, the main process's cache holds ~21 shards (~400 MB) from the `check_loader` at `train.py:431` — that gets pickled 8 times. Linux `fork` avoids this. Not a problem for a Linux rental; a problem for the Windows collaborator path `TO_TRAIN.md` explicitly anticipates. Fix: clear `_shard_cache` after the target-range check. |
| **I6** | LOW | `run_phase4_sweep.sh` and `TO_TRAIN.md` call `python3` and rely on `PYTHONPATH=model/src` rather than the `pip install -e model/` that `TRAINING.md` calls mandatory. Two mechanisms for the same thing; pick one. |

### J. Export and packaging — the stage that runs *after* training, on the same rented box

`export/export_onnx.py`, `export/package_mcim.py` and `scripts/build_mcim.py` were read specifically
to close this gap. **Good news first: export is not a resource risk.** `ChunkExportWrapper.forward`
builds ~20 `[16,384,16]` int64 intermediates (~786 KB each, ~16 MB total), the ONNX file is ~90 MB,
and verification uses `onnxruntime` on `CPUExecutionProvider`. Export can run on a small CPU machine
and does not need to be part of the rental at all. The risks that *are* there are silent-wrong ones.

| # | Sev | Issue |
|---|---|---|
| **J1** | HIGH | `load_imagine_net` (`export_onnx.py:204-210`) calls `net.load_state_dict(ckpt["model_state_dict"], strict=False)` and **discards the returned `(missing, unexpected)` lists without inspecting or printing them**. A checkpoint missing `occupancy_head.*` — the exact thing Phase 4 exists to train — exports a **randomly initialised occupancy head**, passes `onnx.checker`, passes the determinism check, passes shape/dtype conformance, and ships. Compare `text_encoder.py:242-256`, which makes the same call and *does* inspect the result and raise. Fix: capture both lists, allow only a documented allowlist, raise otherwise. |
| **J2** | HIGH | **C2 is unsafe on its own and must be implemented together with J1 and E2.** Dropping the frozen encoder from `model_state_dict` (C2) is only viable because `ImagineNet.__init__` reloads MiniLM from `model/checkpoints/`. But if that directory is absent at *export* time, `PromptEncoder` falls back to a random encoder (E2), `strict=False` swallows the now-missing keys (J1), and you package a `.mcim` whose text encoder is noise — with no error at any stage. Today the checkpoint carries the encoder so this cannot happen. **Do not land C2 until J1 raises and E2 is fatal.** |
| **J3** | MEDIUM | `torch.onnx.export(..., dynamo=False)` (`export_onnx.py:226-233`) pins the legacy TorchScript exporter. `requirements.txt` allows `torch>=2.4` with **no upper bound**; verified here at torch 2.13.0. A newer major that removes the legacy exporter breaks packaging *after* training has already been paid for. Fix: pin `torch>=2.4,<3` once the box's wheel is known, and run the export smoke test as part of preflight on that machine. |
| **J4** | LOW | `verify_chunk_graph`'s seam check is explicitly informational ("NOT expected to be zero") and prints a diagnostic rather than asserting. That is a defensible design, but it means the `.mcim` build has **no automated gate** on chunk-boundary artifacts — the property `docs/poc-plan.md` §1b is built around. The real check lives in `tests/test_model.py`; make sure P12 runs on the machine that exports. |

---

## 3. Preflight harness — **built, working, use it**

`model/src/mc_imagine_model/scripts/preflight.py` exists and runs. It is the literal first command
to run on a rented box, and it is also runnable on the Mac right now.

```bash
# on the Mac today — catches the OOM without renting anything
PYTHONPATH=model/src python -m mc_imagine_model.scripts.preflight \
    --config model/src/mc_imagine_model/training/config.cuda.yaml --quick --keep-going

# on the rented box, full
python -m mc_imagine_model.scripts.preflight \
    --config model/src/mc_imagine_model/training/config.cuda.yaml --budget-hours 40
```

**The design decision that matters:** the budget checks (P3/P4/P5) are computed from
`--target-regions` (default 8000), **not** from whatever dataset happens to be on the machine. The
2026-08 OOM was invisible at rehearsal scale; computing the budget at target scale is what makes it
visible from a laptop. Verified — run against `config.cuda.yaml` on the 16 GB Mac, P4 reports:

```
[FAIL] P4  Host RAM vs shard-cache budget
  shard_cache_size: 512 -> train cache 512, val cache 512
  num_workers: 8 (train loader AND val loader — same kwargs dict)
  cache per process: 9.69 GB
  TOTAL CACHE BUDGET: 155.68 GB
  host RAM: 17.18 GB
  !! shard caches alone want 155.68 GB of 17.18 GB host RAM (906%, ceiling 25%).
```

P9 measures **process-tree** RSS, not just the parent — the whole point of the OOM is that the
memory lives in worker processes a naive `getrusage` never sees. P9/P10/P11 shell out to the real
`train.py` with a temp config, so they exercise the actual code path rather than a model of it.

Flags: `--quick` skips the timed checks (P9/P10/P11/P12), `--keep-going` runs everything despite
failures, `--skip P7,P12` drops individual checks, `--record-hashes` rewrites P7's fixture,
`--budget-hours N` makes P10 fail when the projection exceeds your budget.

Checks, and what each one catches:

| # | Check | Pass criterion |
|---|---|---|
| **P1** | Clean install | Fresh venv, `pip install -e model/` + the CUDA torch wheel, all imports resolve. Catches I4. |
| **P2** | GPU is real | `torch.cuda.is_available()` is True, print device name, total VRAM, driver and CUDA runtime versions. **Fail, do not warn**, if the config asks for CUDA and it is absent. Catches E1. |
| **P3** | Disk headroom | `shutil.disk_usage` on the checkpoint and data volumes. Require ≥ 60 GB free. Catches C1. |
| **P4** | Host RAM headroom | Total RAM ≥ 32 GB, and the computed cache budget (`shard_cache_size × 18.93 MB × (train_workers + val_workers + 1)`) is < 25 % of total. **Print the number.** Catches A1/A2/A3. |
| **P5** | `/dev/shm` | Size of `/dev/shm` ≥ 1 GB when `num_workers > 0`. Print the computed in-flight requirement (`batch_size × per_sample_bytes × num_workers × prefetch_factor`). Catches B1. |
| **P6** | Text encoder is real | `model/checkpoints/all-MiniLM-L6-v2/model.safetensors` present, loads, and `PromptEncoder._loaded_from is not None`. Catches E2. |
| **P7** | Dataset determinism | Generate 8 regions with `--seed 0` into a temp dir, SHA-256 each shard, compare against hashes committed to the repo from the Mac. Catches E5, and any silent generator drift. |
| **P8** | Target-range tripwire | Run the existing `assert_targets_representable` against the real dataset. Already exists; surface it here so it runs before the long job. |
| **P9** | Memory-bounded smoke run | 200 steps at the **full** `config.cuda.yaml` (real `batch_size`, real `num_workers`, real 8000-region dataset), while sampling peak RSS and `torch.cuda.max_memory_allocated()`. Assert RSS < 12 GB and VRAM < 80 % of the card. **This is the check that would have caught the OOM.** |
| **P10** | Throughput and ETA | From P9's 200 steps, report measured steps/sec and the projected wall-clock for `num_epochs × steps_per_epoch`. **This is the number you use to decide whether to keep the machine.** Catches F1's stale benchmark. |
| **P11** | Checkpoint round-trip | Save a checkpoint, kill the process, `--resume` it, and confirm the loss continues rather than restarting. Print the checkpoint's size on disk. Catches C1/E4/G1 regressions. |
| **P12** | Unit tests | `PYTHONPATH=src pytest tests/ -q` — 88 tests currently collect. A green suite on the target machine catches platform-specific numerics before they cost anything. |

**P7 needs its fixture committed.** The first run writes `model/tests/fixtures/shard_hashes_seed0.json`
from the machine it ran on and reports WARN. Commit that file; from then on P7 is a real
cross-platform comparison and will FAIL if the rented box's generator output diverges — which is
the untested load-bearing claim behind "both machines train on the same data" (§2.E5).

**Additionally, `train.py` itself should print a resource banner at startup** — device, VRAM, host
RAM, `/dev/shm` size, computed cache budget, computed in-flight shm requirement, free disk, and
projected checkpoint total. Most of the issues in §2 are visible in that banner, and none of them
are visible today.

---

## 4. Machine spec and budget

### 4.1 Recommended instance

| Resource | Minimum | Recommended | Why |
|---|---|---|---|
| GPU VRAM | 16 GB | **≥ 24 GB** (RTX 4090 / A10 / L4 / A5000) | D1: ~6–8 GB activations at `batch_size 256` fp32, plus workspaces. 12 GB will OOM. |
| vCPU | 8 | **≥ 16** | F1: the data path is the bottleneck, not the GPU. Every extra core is a linear win until the packed-band fix lands. |
| Host RAM | 32 GB | **≥ 64 GB** | A1 after the fix needs ~12 GB of caches + 7.3 GB of page-cached dataset + headroom. 64 GB removes the whole class of concern. |
| Disk | 60 GB | **≥ 100 GB** | C1: 25–30 GB minimum with no margin, and you want room for a second run's checkpoints. |
| `/dev/shm` | 1 GB | **≥ 8 GB** (`--shm-size=8g` / `--ipc=host`) | B1. Non-negotiable and free — just a launch flag. |
| Pricing tier | on-demand | on-demand **until G1 lands**, then spot | G1: without mid-epoch checkpointing, a preemption costs a full epoch. |

An A10 / L4 / RTX 4090 class instance with 16 vCPU / 64 GB / 100 GB satisfies all of it and sits in
the cheapest tier that does.

### 4.2 Wall-clock budget

Estimate, not a measurement — P10 replaces it with a real number on the actual box:

| Stage | Estimate | Basis |
|---|---|---|
| Install + preflight (§3) | 0.5 h | |
| Dataset generation, 8000 regions | **~7.9 h serial**; ~0.6 h on 16 cores if F3 is fixed; ~0.3 h if rsync'd up | measured, mean 3.55 s/region across archetypes (§2.F3) |
| Data path per epoch | ~16 min at 8 workers today; ~2–4 min after the packed-band fix | measured 11.1 ms/shard × 684 800 misses/epoch |
| GPU compute per epoch | ~10–25 min depending on card | ~18.5 GFLOP/sample × 736 k samples ≈ 13.6 PFLOP/epoch at an assumed 25–35 % fp32 utilisation |
| **60 epochs** | **~20–35 h unfixed; ~12–20 h with the packed-band and MiniLM-cache fixes** | |
| Export + `.mcim` build | 0.5 h, and can run off the rented box entirely (§2.J) | |

**Budget ~40 wall-clock hours for the first full run as things stand**, of which ~8 h is dataset
generation that F3 removes almost entirely. Treat F1/F2/F3 as directly buying down the number.
The 9.15 h figure in `config.cuda.yaml` should be struck from the docs until re-measured (§2.F1).

Do not take these on faith — **preflight P10 prints a measured projection on the actual machine**,
and `--budget-hours` turns it into a pass/fail gate.

### 4.3 Abort triggers — decide early, not at hour 20

Kill the run and stop paying if any of these are true:

- P10's projected wall-clock exceeds 2× your budget. Fix the data path locally; do not run it out.
- The startup banner shows `amp=False` **and** device `cpu`. E1 — you have the wrong torch wheel.
- The startup banner does not report a loaded MiniLM checkpoint. E2 — every prompt will look alike.
- GPU utilisation sits below ~40 % for the first 500 steps. F1 — you are paying for an idle GPU;
  raise `num_workers` or land the packed-band fix first.
- Peak RSS exceeds 50 % of host RAM in the first 500 steps. A1 has not actually been fixed.
- The epoch-4 qualitative PNG shows six visually identical prompts. E2, or the flat-terrain failure
  this project has shipped before.

---

## 5. Sequencing, with acceptance criteria

Each item below has a **Done when** line: the command to run and the output that constitutes proof.
An item is not complete because the code changed; it is complete when its check passes.

**Decision authority.** Items marked 🔒 are *not* open to redesign by whoever implements this — the
decision is already made and deviating from it re-opens a failure this project has already paid for.
Items marked 🔧 are ordinary engineering judgement; implement them however is cleanest.

### Stage 0 — local, free, before renting anything

| # | Item | Done when |
|---|---|---|
| 0.1 | 🔒 **H3** — run `docs/phase4.1-plan.md` §2.2's confirmatory run on the Mac (400 regions, 3 000 steps). **Do not skip to make progress on the rest of this plan.** Gate 2 has no verdict; renting for Gate 3 before it does is buying a known-unknown. | `diagnose_overhangs.py` prints a `multi_run_column_pct` line and §2.2's branch has been taken in writing. |
| 0.2 | 🔒 **H1/H2** — `ConsistencyLoss`. **Set `consistency_weight: 0.0` in all three configs and record why.** Do *not* invent a corrected loss as part of this plan; that is a research change and belongs in its own experiment with its own before/after. | `grep consistency_weight model/src/mc_imagine_model/training/config.*.yaml` shows `0.0` everywhere, with a comment citing this section. |
| 0.3 | 🔧 **A1 + A2 + A3** — `shard_cache_size: 8` in `config.cuda.yaml`; separate `loader_kwargs` for the val loader (`num_workers: 2`, `persistent_workers=False`); `null` no longer means unbounded. | `preflight --config config.cuda.yaml --quick` → **P4 PASS** at `--target-regions 8000`. |
| 0.4 | 🔧 **B2** — `occupancy_band` emitted as `uint8`; both loss classes already `.float()` it. | P5's printed per-sample size drops from 67.6 KiB to ~20 KiB; `pytest tests/ -q` still green. |
| 0.5 | 🔧 **E3** — `_load_shard`'s four `assert`s become explicit `raise`s. | `python -O -m pytest tests/ -q` green, and a deliberately corrupted shard still raises under `-O`. |
| 0.6 | 🔒 **E1/E2** — an explicitly named-but-unavailable device raises; a missing MiniLM checkpoint raises unless `--allow-random-text-encoder` is passed. **These must be fatal, not warnings** — both have already cost real runs. | `preflight` P2 and P6 FAIL when they should, and `train.py` itself refuses to start in both conditions. |
| 0.7 | 🔧 Commit `model/tests/fixtures/shard_hashes_seed0.json`. | `preflight --skip …` → P7 reports PASS (comparison), not WARN (recording). |
| 0.8 | — Regression gate for all of Stage 0. | `PYTHONPATH=src pytest tests/ -q` — all 88 green. |

### Stage 1 — the safety net

| # | Item | Done when |
|---|---|---|
| 1.1 | 🔧 **C1** — `keep_last_n_checkpoints` (default 3); every `torch.save` writes to a temp file then `os.replace`. | P11 reports a projected checkpoint total under 1 GB; killing the process mid-save leaves the previous `last.pt` intact and loadable. |
| 1.2 | 🔒 **C2 + J1 + J2 together, in that order, or not at all.** J1 (export raises on unexpected missing keys) must land *first*; C2 (drop the frozen encoder from checkpoints) is only safe behind it and behind 0.6. | Export from a C2-format checkpoint with `model/checkpoints/` **deleted** must **fail loudly**, not silently ship a random encoder. That negative test is the acceptance criterion. |
| 1.3 | 🔒 **G6 before or with G1.** Mid-epoch checkpointing without the resume fix turns preemption recovery into a silent data-skip. | P11 shows a mid-epoch resume returning to the *same* epoch at the correct step, not `epoch+1`. |
| 1.4 | 🔧 **G1** — `save_every_steps` (500) writing `last.pt`; `SIGTERM`/`SIGINT` flush a checkpoint before exit. | `kill -TERM` mid-epoch produces a resumable `last.pt`; P11 passes. |
| 1.5 | 🔧 **G2/G3** — `FileHandler` into `checkpoint_dir`; steps/sec and projected finish on every log line. | The log file exists after a run and every step line carries a rate and an ETA. |
| 1.6 | 🔧 Startup resource banner in `train.py` (device, VRAM, RAM, `/dev/shm`, cache budget, in-flight shm, free disk, projected checkpoint total). | The banner appears before the target-range check and its numbers match preflight's. |
| 1.7 | 🔧 **E4/G4** — persist loader RNG state and `total_steps`; warn on mismatch. | A resumed run does not replay the same sample order. |

### Stage 2 — earn the throughput back (pays for itself in one rental)

| # | Item | Done when |
|---|---|---|
| 2.1 | 🔒 **F3 first** — parallelise `generate_shards` with `ProcessPoolExecutor`. Highest ratio of saved money to risk in this plan (~8 h → ~35 min), and it cannot change the training objective. **P7 is the guard: the parallel path must produce byte-identical shards.** | P7 PASS against the committed fixture, generated via the parallel path. |
| 2.2 | 🔒 **F1** — bit-pack `band` along y on disk. **This changes the on-disk data format and is the single most dangerous change in this plan.** `np.unpackbits` must pass `count=BAND_HEIGHT` explicitly; an axis or bit-order slip produces occupancy that is *shifted*, not obviously broken — exactly this repo's signature failure mode. **Required: a round-trip unit test asserting `unpack(pack(band)) == band` elementwise on a real shard, plus `_load_shard` rejecting the old format with a clear message.** Do not land without both. | New round-trip test green; P7 fixture re-recorded; measured shard decompression drops from ~11.1 ms to ~1–2 ms. |
| 2.3 | 🔧 **F2** — precompute the per-region prompt embedding table; keep the live encoder path for export parity. | A test asserts the cached embedding equals the live encoder's output for the same caption; P10's rate improves. |
| 2.4 | 🔧 **D4** — measure TF32 on/off on the CUDA box. Keep only if the numbers justify it. | A recorded before/after in §6, not a blind flip. |
| 2.5 | 🔧 **J3** — pin `torch>=2.4,<3` once the box's wheel is known. | `pip check` clean; export smoke test passes on that wheel. |

### Stage 3 — rent

1. Provision per §4.1. **Launch the container with `--shm-size=8g` or `--ipc=host`.**
2. `python -m mc_imagine_model.scripts.preflight --config config.cuda.yaml --budget-hours 40`.
   **Do not start training until it exits 0.**
3. Read P10's projection against §4.3's abort triggers. Decide here, at minute five, not at hour twenty.
4. Rehearsal first, exactly as `TO_TRAIN.md` prescribes: `--regions 400`, `--max-steps 50`, export a
   `.mcim`, load it in Minecraft.
5. Full run in `tmux` with the log tee'd to disk. Check the epoch-4 qualitative PNG before going to bed.

### Definition of done — when to press "rent"

- `preflight.py` exits 0 on the Mac with `--quick --keep-going`, except P2 (correctly FAILs: no CUDA
  here) and P5 (correctly SKIPs: no `/dev/shm` on Darwin). **P4 must PASS** — that is the direct,
  measured refutation of the crash you hit.
- P9 has run locally against `config.cuda.yaml` with `device: cpu` and `num_workers: 8` and peak
  process-tree RSS stayed under 12 GB.
- Every BLOCKER and HIGH in §2 is fixed or has a written, accepted justification.
- Gate 2 has a verdict (0.1).

---

## 6. Measurement log

Everything quoted above, with the command that produced it. Measured 2026-08-05 on the development
Mac (Darwin 25.3.0, Apple Silicon, Python 3.13, repo `.venv`), against the 400-region dataset in
`model/data/`.

| Quantity | Value | How |
|---|---|---|
| Shard on disk (compressed) | 0.915 MB avg | `du -sh model/data` ÷ 400 |
| Shard decompressed, all arrays | **18.93 MB** | `np.load` member shapes/dtypes: band `(520,520,64)` bool = 17.31 MB, heightfield `(520,520)` f32 = 1.08 MB, profile_map + biome_map u8 = 0.54 MB |
| Full-shard decompression time | **11.1 ms** | timed loop over 6 shards, page cache warm |
| Heightfield-only load | 4.0 ms | same |
| `param_*`-only load (dataset `__init__` path) | 0.8 ms | same |
| `render_region`, **uncarved** archetypes (`carve_strength == 0`) | **0.33 s** | savanna_plains, desert_dunes, rolling_grassland, swamp |
| `render_region`, **carved** archetypes (`carve_strength > 0`) | **~8.0 s** | valley_mountains_craters 8.05, taiga_forest 7.84, snow_peaks 8.19, deep_canyon 8.31, mesa_plateaus 7.95 — the 3-D `_fbm3d` carve over `[520,520,64]` |
| `render_region`, **mean** | **3.55 s/region** (5 of 12 regions carve) | 12 consecutive regions at `seed=0`, `region_layout_width=64` |
| 8000-region generation, serial | **~7.9 h** | 3.55 s × 8000. *An earlier single-sample measurement of 0.46 s/region was wrong — region (0,0) is `savanna_plains`, an uncarved archetype. Caught by preflight P7.* |
| `savez_compressed` of one region | 0.12 s → 0.90 MB | uncarved region |
| Model parameters | 25 318 135 total / **2 752 759 trainable** | `ImagineNet(config.cuda.yaml['model'])` |
| Checkpoint size on disk | **123.4 MB** | `torch.save` of the exact dict `save_checkpoint` writes, with populated AdamW state |
| Per-sample collated bytes | **67.6 KiB**, of which `occupancy_band` is **95 %** | sum over the tensor fields `dataset.__getitem__` returns |
| Batch bytes @ 256 | 17.7 MB | 67.6 KiB × 256 |
| DataLoader in-flight shm @ 8 workers × prefetch 2 | **284 MB** | 17.7 MB × 16 |
| 8000-region split | 7 360 train / 640 val, **7.3 GB** on disk | `val_region_fraction: 0.08` |
| Samples per epoch | 736 000 (`chunks_per_region: 100`) | |
| Steps per epoch @ `batch_size 256` | 2 875; **172 500 steps over 60 epochs** | |
| 512-entry LRU hit rate over 7 360 shards | **7.0 %** | 512 / 7 360 |
| Shard decompressions per epoch | **684 800** = 2.1 core-hours of zlib | 736 000 × 0.93 × 11.1 ms |
| Bytes read per epoch via the miss path | **627 GB** | 684 800 × 0.915 MB |
| Cache RAM, `shard_cache_size: 512` | **9.69 GB/process**; ×8 train workers = 77.5 GB; ×16 train+val = **155.1 GB** | 512 × 18.93 MB |
| Cache RAM, `shard_cache_size: 8` (proposed) | 0.15 GB/process; ~1.2 GB across 8 workers | |
| Band bit-packed along y (proposed) | shard 18.93 MB → **2.16 MB** (8.8×) | `(520,520,8)` uint8 |

Additionally verified by running the harness itself (`config.mps.yaml`, 60 steps, 400-region dataset):

| Quantity | Value | How |
|---|---|---|
| Peak process-tree RSS, MPS smoke run | 2.53 GB | preflight P9 |
| Throughput, MPS, `batch_size 64` | 1.17 steps/s (75 samples/s) | preflight P10 |
| `last.pt` size | **123.4 MB** (independently confirms the estimate above) | preflight P11 |
| Mid-epoch resume behaviour | stopped at step 50 of 575 → **resumed at epoch 1, step 50** | preflight P11 — this is §2.G6 |
| Development Mac | 16 GB RAM, 12 cpus, Darwin arm64, torch 2.13.0 | preflight banner |

**Not measured — flagged as estimates:** VRAM at `batch_size 256` (§2.D1, ~6–8 GB, derived from
activation shapes) and CUDA step time / epoch wall-clock (§4.2, derived from a ~18.5 GFLOP/sample
FLOP count at an assumed utilisation). No CUDA hardware was available. Both are replaced by real
numbers the moment preflight P9 and P10 run on the rented machine — which is precisely why those
two checks exist rather than a table in this document.

**Known coverage limits of this review.** Read end to end: `training/train.py`, `data/dataset.py`,
`training/losses.py`, `model/imagine_net.py`, `model/heads.py`, `model/text_encoder.py`,
`export/export_onnx.py`, `tokenizer_utils.py`, `scripts/bootstrap.py`, `scripts/generate_data.py`,
all three configs, `requirements.txt`, `TRAINING.md`, `TO_TRAIN.md`. Sampled or grepped only:
`data/world_generator.py` (1 933 lines — its *cost* profile is now measured, but its numerics are
not reviewed), `scripts/diagnose_*.py`, `viz.py`, `export/package_mcim.py`. **Not examined at all:**
the Java/mod side (`mod/`), and the `tests/` bodies. The archetype cost surprise in §2.F3 came from
territory this review had only grepped — treat the un-read regions as the most likely home of the
next surprise.

---

## 7. Handoff — implementing this with another agent

This document is diagnosis plus acceptance criteria. When handing it to an implementing agent:

1. **Point it at §5 and tell it §2 is reference, not a task list.** §2 has 40 numbered issues; §5 is
   the ordered subset that actually needs doing, with proof conditions.
2. **Enforce the 🔒 markers.** Those are decisions already made. In particular: `ConsistencyLoss`
   gets zeroed, not redesigned (0.2); C2 does not land without J1 (1.2); G1 does not land without
   G6 (1.3); F1 does not land without its round-trip test (2.2).
3. **Require verification per item, not at the end.** Every row in §5 has a "Done when". An agent
   reporting "implemented Stage 0" without per-item check output has not finished Stage 0.
4. **Do not let it touch Stage 2 before Stage 0 and 1 are green.** F1 changes the on-disk data
   format; doing that while other things are in flight makes a silent-wrong result unattributable.
5. **Re-run `preflight` after every stage.** It is the regression test for this entire plan.
