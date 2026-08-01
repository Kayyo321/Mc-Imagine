# TO_TRAIN.md — starter prompt for training on a new machine

This file exists for one purpose: you have just cloned this repo onto a fresh machine and
want to train the model, then play the result in Minecraft.

**Open Claude Code in the repo root and paste the block below as your first message.** It
front-loads the platform gotchas that would otherwise cost you an hour each, and it tells
Claude to rehearse the whole pipeline cheaply before committing to a long GPU run.

Edit the "MY STARTING POINT" section to match your actual machine before pasting.

---

```
I've just cloned this repo (Mc-Imagine — a Minecraft mod that generates worlds from a
text prompt using a neural net). I want to train the model on my machine, then test it
in Minecraft and debug whatever looks wrong.

MY STARTING POINT
- Windows/Linux (tell me if the steps differ — the repo was developed on macOS).
- NVIDIA GPU.
- Minecraft installed.
- Nothing else: no Python, no Java, no CUDA toolkit, no build tools. Assume bare metal.

START BY READING
- docs/TRAINING.md — the authoritative clone-to-playing walkthrough. Follow it; don't
  re-derive it.
- docs/phase2-plan.md §0 — the bugs that were found and why they were subtle. Read this
  before debugging anything, so you don't rediscover a known issue.
- docs/model-spec.md — the tensor/manifest contract both sides must obey.

WHAT I WANT, IN ORDER
1. Install prerequisites: Python 3.11+, JDK 17 (specifically 17 — see gotchas), and the
   CUDA-matched PyTorch wheel. Verify each actually works before moving on.
2. Set up the Python package and regenerate the two large gitignored directories
   (the MiniLM encoder and the training dataset).
3. Do a FAST END-TO-END REHEARSAL FIRST, before committing to a long run:
   generate only `--regions 400`, train with `--max-steps 50`, export a .mcim, and load
   it in Minecraft. It'll look bad — that's fine and expected. The point is to prove
   every stage works and to shake out install/path problems in minutes instead of
   discovering them after an hour of training.
4. Only once the rehearsal fully succeeds, do the real run: regenerate at
   `--regions 8000` (a few GB, tens of minutes) and train properly on the GPU.
5. Export the real .mcim and get it loaded in Minecraft.
6. Help me judge whether the output is actually good, and debug it if not.

GOTCHAS THAT WILL WASTE MY TIME IF YOU DON'T KNOW THEM
- Gradle needs JDK 17 exactly. On JDK 21/24 it fails with "Unsupported class file major
  version". Set JAVA_HOME to a real JDK 17 before any gradle command.
- KNOWN LANDMINE: mod/gradle.properties contains
  `org.gradle.java.installations.paths=/opt/homebrew/opt/openjdk@17`
  which is a macOS Homebrew path that does NOT exist on my machine. It was added so
  Gradle could find JDK 17 on the original dev's Mac. Gradle usually tolerates a missing
  search path and falls back to auto-detection, but if it can't find a toolchain you'll
  get a confusing "No locally installed toolchains match" error. If that happens, either
  point that line at my JDK 17 or make it cross-platform / remove it and rely on
  auto-detection. Tell me what you changed and why — my collaborator is still on macOS,
  so we need to agree on the fix rather than silently diverging.
- `mod/fabric/run/` is gitignored, so `mod/fabric/run/mcimagine/models/` does not exist
  on a fresh clone. Create it before copying the .mcim in.
- `pip install -e model/` is mandatory. Without it every `python -m mc_imagine_model.…`
  command fails.
- Do NOT hard-pin torch from requirements.txt — install the wheel matching my CUDA
  version, then confirm `torch.cuda.is_available()` prints True. If it's False, stop and
  fix it; training will silently fall back to CPU and take days.
- Training prints a "Target range check" before the first step. If it FAILS, regenerate
  the dataset. Do not disable or loosen the check — it exists because a silently
  saturated model shipped once already.
- On the CUDA config, the qualitative PNGs are written every 5th epoch, not every epoch.
  Nothing is broken if none appears at epoch 0.
- If you ever change `data/world_generator.py` or `spec_constants.py`, the dataset must be
  regenerated — mixing generations trains a confused model without raising an error.

HOW TO JUDGE THE RESULT
docs/TRAINING.md has a table of Day-1 baseline numbers to beat, and four benchmark
prompts with what each should look like. The per-epoch PNGs matter more than the loss
value. In game I specifically want to check:
- "eroded red mesa plateaus" -> red/orange terraced rock, arid, NO random water pools
- "towering snow-capped peaks" -> tall mountains, snow on top, spawn well above y=120
- "vast deep valleys..." -> valley floors that actually drop below sea level
- walking a few thousand blocks -> no seams or hard discontinuities at chunk borders

KNOWN OPEN ISSUES — don't chase these as if they're new bugs
- Terrain repeats every 2048 blocks. Deliberate trade for learnability; documented.
- Canyon floors are narrow (deep but ~30 blocks wide). One-constant fix if I dislike it.
- Overhangs/caves (Phase 4, docs/phase4-plan.md) are mid-flight, not shipped: the
  architecture, ground truth, and ONNX plumbing are built and verified, but the occupancy
  head has not yet demonstrated it can learn multi-run (walkable-void) columns at
  rehearsal scale — see "Phase 4.1 confirmatory long run" below before assuming this
  works. Until that lands, block volume is still expanded from one height per column;
  caves come from vanilla carvers.

HOW I WANT YOU TO WORK
- Verify each step actually worked instead of assuming — this project has already been
  bitten by "looks right, silently wrong" bugs more than once.
- Show me real command output, especially for the CUDA check and the target-range check.
- If something's ambiguous or looks wrong, stop and ask rather than working around it.
- Long training runs: launch in the background and tell me the ETA, don't block on them.
```

---

## Why the prompt is shaped this way

**The rehearsal step (400 regions, 50 steps) is the highest-value part.** It converts "install
problem discovered 90 minutes into training" into "install problem discovered in 5 minutes," and
it exercises the Java/Minecraft half of the pipeline early — which is exactly where
platform differences (JDK version, hardcoded paths, missing directories) actually live.

**The gotchas are not hypothetical.** Every one of them was hit during development on the
original machine. The JDK-17 requirement, the missing models directory, and the mandatory
editable install each cost real debugging time; the `pip install -e model/` failure in
particular was invisible in the dev environment and only surfaced when installing into a
clean virtualenv.

**"Verify, don't assume" is emphasized deliberately.** This project's two worst bugs — a
saturated activation that silently flattened all terrain, and a block-palette off-by-one
that rendered stone as bedrock and red sand as water — both produced output that looked
plausible and raised no errors. Neither was caught by reading code; both were caught by
measuring. See `docs/phase2-plan.md` §0.

## Phase 4.1 confirmatory long run (overhangs / caves)

If you're picking this repo up mid-Phase-4 (`docs/phase4-plan.md`, `docs/phase4.1-plan.md`):
the "KNOWN OPEN ISSUES" note above about "the model cannot make overhangs or caves" is what
this phase is trying to fix. Gate 2's first sweep (`docs/CHANGELOG.md`) failed uniformly —
0.0000% multi-run columns on all 14 checkpoints — traced to `overhang_weight: 1.0` being an
unnormalized loss term ~50-250x the magnitude of every other loss, destabilizing the
optimizer. `docs/phase4.1-plan.md` §1.2 corrected the shipped configs to `overhang_weight:
0.005` and fixed a `train.py` checkpoint-saving bug (§1.1) that silently wrote zero
checkpoints under `--max-steps` on a config whose epoch outruns the step count. Before
spending Gate 3's full 8000-region/9-hour CUDA budget, §2.2 calls for one confirmatory run
at the corrected weight, long enough to tell "undertrained" from "structurally incapable"
apart — the original Gate 2 sweep only ran 400 steps, half of them inside warmup.

**Run it like this** (400 regions — this is still rehearsal-scale, no need for the full
8000-region dataset yet):

```bash
# 1. Regenerate/confirm the 400-region rehearsal dataset (same one Gate 2's sweep used).
PYTHONPATH=model/src python3 -m mc_imagine_model.scripts.generate_data \
  --regions 400 --out model/data --seed 0

# 2. config.rehearsal.yaml ships num_epochs: 1 (~143 steps total over 400 regions at
#    batch_size 256) — too short to ever reach --max-steps 3000 on its own; the run would
#    silently stop at ~143 steps via the normal epoch-end path instead of the intended
#    3000, an "it finished" result that is quietly not the confirmatory run at all. Bump
#    num_epochs first so there's enough of the dataset to iterate over:
sed 's/num_epochs: 1/num_epochs: 25/' \
  model/src/mc_imagine_model/training/config.rehearsal.yaml \
  > model/src/mc_imagine_model/training/config.confirmatory.yaml

# 3. The confirmatory run itself: 2,000-4,000 steps per §2.2, midpoint 3000, at the
#    corrected overhang_weight. Passed explicitly on the CLI so it can't silently fall back
#    to whatever a given config's default happens to be; occupancy_weight/consistency_weight
#    stay at config defaults pending the §1.2 re-sweep (run_phase4_sweep.sh).
PYTHONPATH=model/src python3 -m mc_imagine_model.training.train \
  --config model/src/mc_imagine_model/training/config.confirmatory.yaml \
  --max-steps 3000 \
  --overhang-weight 0.005

# 4. Evaluate against the same 400-region ground truth, full 5-condition go/no-go:
PYTHONPATH=model/src python3 -m mc_imagine_model.scripts.diagnose_overhangs \
  --ground-truth model/data \
  --checkpoint model/training_runs/rehearsal/best.pt
```

**The decision point (§2.2, §3 Step C) — read the `multi_run_column_pct` line, then branch:**

- **Still exactly 0.0000%.** Treat this as a real negative result, not a "needs more steps"
  excuse to run longer again. Move to `docs/phase4.1-plan.md` §2.3's structural diagnosis in
  order: (1) is the occupancy head initializing near "always air"/"always solid" rather than
  the true ~0.38% overhang-cell marginal, (2) is gradient flow through the y-axis
  `ConvTranspose3d` upsample starved relative to the 2D backbone at matched depth, (3) is
  `OccupancyLoss`'s transition-cell weighting actually strong enough given how rare
  transition cells are in the target, (4) only after 1-3 — capacity. Do not start Gate 3
  while this branch is open.
- **Multi-run columns appear, even a little.** Proceed to the §1.2 re-sweep
  (`./run_phase4_sweep.sh`) and re-run the full 5-condition go/no-go
  (`evaluate_baseline_gate()` in `diagnose_overhangs.py`) properly before considering Gate 3.
  A PASS there is a genuine go; `docs/phase4-plan.md` §7's Gate 3/Gate 4 text applies
  unchanged.

## If you are the person handing this repo over

Point your collaborator at this file and nothing else. `docs/TRAINING.md` is the reference
they will end up living in, but this file is the on-ramp.
