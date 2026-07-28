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
- The model cannot make overhangs or caves — block volume is expanded from one height per
  column. Caves come from vanilla carvers.

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

## If you are the person handing this repo over

Point your collaborator at this file and nothing else. `docs/TRAINING.md` is the reference
they will end up living in, but this file is the on-ramp.
