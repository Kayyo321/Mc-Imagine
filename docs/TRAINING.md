# Training Mc-Imagine on a GPU box

This is the complete path from a fresh `git clone` to a `.mcim` model you can drop into the mod and
play. It assumes an NVIDIA GPU; everything also runs on CPU or Apple Silicon (MPS), just slower.

Two large directories are **deliberately not in git** and must be recreated locally:
`model/checkpoints/` (the ~87 MB MiniLM text encoder) and `model/data/` (the training set, several
GB). Steps 2 and 3 below regenerate them. Both are fully deterministic given a seed, so two machines
following these steps get byte-identical inputs.

---

## 1. Install

```bash
git clone <this repo>
cd Mc-Imagine

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e model/
```

`pip install -e model/` is **required** — without it every `python -m mc_imagine_model.…` command
below fails with `ModuleNotFoundError`.

Then install the torch build that matches your CUDA version (the shared pins deliberately leave
torch loose, because the correct wheel differs per machine):

```bash
nvidia-smi                                            # read the CUDA version, top right
pip install torch --index-url https://download.pytorch.org/whl/cu124   # or cu126 / cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

That last line must print `True`. If it prints `False`, training will silently fall back to CPU and
take days instead of hours — check it before starting a long run.

## 2. Fetch the text encoder

```bash
python -m mc_imagine_model.scripts.bootstrap
```

Downloads `sentence-transformers/all-MiniLM-L6-v2` into `model/checkpoints/all-MiniLM-L6-v2/` and
verifies the weights load and the tokenizer parses. Idempotent — safe to re-run. Needs internet
once; everything after this is offline.

## 3. Generate the training data

```bash
python -m mc_imagine_model.scripts.generate_data --regions 8000 --out model/data --seed 0
```

~8000 macro-regions is a good GPU-scale set (each is a 512×512-block region; expect a few GB and
tens of minutes). Use `--regions 400` for a quick end-to-end rehearsal first. `--force` clears an
existing directory.

> If you pull changes that touch `data/world_generator.py` or `spec_constants.py`, **regenerate**.
> Terrain shape, the height band and the biome palette are all baked into the shards, and mixing
> generations silently trains a confused model rather than raising an error.

## 4. Train

```bash
python -m mc_imagine_model.training.train \
    --config model/src/mc_imagine_model/training/config.cuda.yaml
```

Use `config.mps.yaml` on Apple Silicon, or edit `device:` to `cpu`. Useful flags:

- `--max-steps 50` — smoke test; proves the whole pipeline runs before you commit hours to it.
- `--resume model/training_runs/cuda_run1/last.pt` — continue an interrupted run (restores model,
  optimizer, LR schedule, epoch and step).

**What you should see immediately.** Before the first optimizer step the run prints a target-range
check:

```
Target range check passed over 4 batch(es): targets in [50.98, 251.40], head band [-96.0, 288.0]
Model params: 25132118 total, 2566742 trainable
```

If that check *fails*, stop — it means the generated terrain doesn't fit what the model can
represent, and training would silently produce flat worlds (that exact bug shipped in Day 1; see
`docs/phase2-plan.md` §0). Regenerate the data rather than disabling the check.

Each epoch writes a checkpoint (`epoch_XXX.pt`, plus `best.pt` / `last.pt`). Separately, every
`training.qualitative_dump_every_epochs` epochs — **5** in `config.cuda.yaml`, 1 in
`config.mps.yaml` — it writes `epoch_XXX_qualitative.png`, rendering the six held-out benchmark
prompts at a fixed seed. **Look at those PNGs** — they are the real signal, more than the loss
number. (Don't be alarmed when none appears after epoch 0 on CUDA; the first lands at epoch 4.)

## 5. Export a `.mcim`

```bash
python -m mc_imagine_model.scripts.build_mcim \
    --checkpoint model/training_runs/cuda_run1/best.pt \
    --out imaginator-low_intensity-no_structures-1.1.0.mcim
```

Runs checkpoint → ONNX → ORT verification → packaging in one step, defaulting the block/biome
palettes from `spec_constants.py` so they cannot drift out of sync with what the model was trained
against. Expect ~90 MB (the frozen fp32 MiniLM dominates) and these checks to pass: `onnx.checker`,
coordinate-purity (must be exactly `0.0`), shape/dtype conformance, and determinism.

## 6. Play it

Copy the `.mcim` into the mod's model directory:

- running from source: `mod/fabric/run/mcimagine/models/`
- a normal install: `.minecraft/mcimagine/models/`

```bash
cd mod
export JAVA_HOME="$(/usr/libexec/java_home -v 17)"    # macOS; any JDK 17 works
./gradlew :fabric:runClient
```

Create a world → **Customize** → pick the model, type a prompt, set a seed → Create.

Good prompts to sanity-check with, and what each should look like:

| Prompt | Expect |
|---|---|
| `towering snow-capped peaks` | Tall mountains, snow on top, spawn well above y≈120 |
| `eroded red mesa plateaus` | Red/orange terraced rock, arid, no water pools |
| `vast deep valleys, beautiful mountains overlooking water-filled craters` | Large elevation swings, valley floors below sea level |
| `endless flat desert dunes` | Low, flat, sandy, spawn near y≈63 |

---

## What "better than Day 1" means, numerically

Day 1 shipped a model with these measured properties. Any retrain should beat them; if it doesn't,
something regressed:

| Metric | Day-1 baseline | Why it was bad |
|---|---|---|
| Height std, `snow-capped peaks` | **0.03** (pinned at 158.84–158.98) | tanh saturated at its 159 ceiling — dead gradient, zero relief |
| Height std, all 6 prompts | 0.02 – 3.73 | Terrain was essentially flat everywhere |
| Deepest terrain generated | y = **27.7** | No valleys of any consequence, nothing near sea level's underside |
| Trainable params | ~800 k | Under-capacity |
| Inference latency | 8.9 ms/chunk | Fine — ~10× under the 100 ms budget, so there is headroom to grow |

The Phase-2 changes (see `docs/phase2-plan.md`) address each: a canonical, *learnable* noise field
instead of per-region pseudorandom fields; a head range of [-96, 288] instead of [-33, 159]; real
valley carving in the data; a slope-aware loss; and ~2.57 M trainable params.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: mc_imagine_model` | You skipped `pip install -e model/`. |
| `torch.cuda.is_available()` is `False` | Wrong torch wheel for your CUDA version — reinstall with the right `--index-url`. |
| `No region_*.npz shards found` | Run step 3. |
| `TARGET RANGE CHECK FAILED` | Data and model disagree on the height band — regenerate the data (step 3). Do not disable the check. |
| Terrain is flat / all prompts look alike | The classic failure. Check the qualitative PNGs each epoch, confirm the target-range check passed, and confirm you regenerated data after pulling. |
| Out of memory on GPU | Lower `training.batch_size` in `config.cuda.yaml` (256 → 128 → 64). |
| Slow data loading | Raise `hardware.num_workers`; lower `data.shard_cache_size` if RSS is too high (each worker holds its own cache). |
