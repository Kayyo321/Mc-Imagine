# Mc-Imagine Phase 4.1 Plan — Making the Occupancy Head Learn

**Goal:** close the gap between `docs/phase4-plan.md` as written and `docs/phase4-plan.md` as
*realized*. That document is not wrong — its architecture (§1.3), its ground-truth generator (§2),
its ONNX plumbing (§1.5, proven in Gate 1), and its loss design (§3) are all built, committed, and
individually correct. What Phase 4.1 exists to fix is that **Gate 2, run for the first time on
2026-07-31, failed uniformly**: 14 rehearsal-scale checkpoints, sweeping every weight and capacity
knob §3.4 and §4 specify, produced **exactly 0.0000% multi-run columns** on every single one. This
document is the diagnosis-and-recovery plan for that result — it does not propose a different
architecture or a different loss family. `docs/phase4-plan.md` §8's own risk table named this exact
failure mode first: *"Occupancy collapses to the marginal probability → mush or a heightfield... This
exact failure has shipped three times in 2D."* This is the fourth time, in 3D, and this document is
where it gets diagnosed rather than shipped.

**Definition of done:** `docs/phase4-plan.md` §7's Gate 2 go/no-go conjunction — all five conditions,
not the three `diagnose_overhangs.py` checks natively — passes on a rehearsal-scale run, for real,
independently reproducible. At that point Gate 3 and Gate 4 need no new plan; §7's own text already
covers them, and nothing in this document changes a single word of §1 through §6.

**How to use this document:** it is short on purpose. Every fix below is scoped to unblock Gate 2 and
nothing else. Read `docs/phase4-plan.md` first — this document assumes it, cites it by section number
throughout, and does not re-explain the architecture, the loss math, or why the go/no-go is a
five-condition conjunction rather than the single metric an earlier draft used (§7's own long
discussion of the speckle/shattering adversarial fixtures is required reading before touching any of
the weights below).

---

## 0. Where Phase 4 actually is (2026-07-31)

| `docs/phase4-plan.md` gate | Plan's exit criterion (§7) | Actual status |
|---|---|---|
| Gate 0 — ground truth exists | Rendered cross-section showing an overhang; measured cavity-height distribution | **Done.** §9's Gate 0 rows are filled in: 10-region shard audit (~915KB/region compressed, ~19x), periodicity and frequency-pinning verified both by the implementing session and independently re-derived numerically by a follow-up review. |
| Gate 1 — plumbing works, no model involved | Deterministic-carve export, loaded in Minecraft, walk under the overhang, screenshot | **Plumbing done and verified by direct ONNX Runtime inference** (a real walkable, correctly-dressed overhang was confirmed in `imaginator-gate1.mcim`). **The literal exit criterion — launch the game, walk under it, screenshot — has never been done.** §9's own row for this is still blank. Nothing in this document changes that; §2.5 below just flags it as still open. |
| Gate 2 — learnability and capacity | §7's five-condition conjunction, at rehearsal scale (400 regions/400 steps) | **Run. Failed uniformly.** 14 checkpoints (one-factor-at-a-time sweep of §3.4's three weights, plus §4's `conv_channels` capacity axis), every one scoring 0.0000% multi-run columns. §9's Gate 2 rows are filled in with real numbers — this document exists because of what they say. |
| Gate 3 — full run | 8000 regions on the chosen config | **Correctly not started.** §7 gates this behind Gate 2's pass; Gate 2 has not passed. |
| Gate 4 — export, play, release | Four benchmark prompts + overhang prompt, `Imgs/` capture, §6 spec bump | **Not started.** §6's spec/versioning mechanics are actually already done (model-spec 0.6.0, `mod_version` decision, CHANGELOG) — what's outstanding here is purely the trained-model deliverables, which need Gate 3 first. |

**The one-line version:** everything in `docs/phase4-plan.md` that could be built and tested without a
GPU is built, tested, and — as of the Gate 2 run — has told us something true and uncomfortable about
the loss design at rehearsal scale. That is exactly what Gate 2 is *for* (§7: *"Diagnose first. A
9-hour run that produces nothing while printing normal-looking progress has happened here before...
this is the place to apply it"*). Phase 4.1 is that diagnosis.

---

## 1. Known issues — do these first

These block a *trustworthy* re-run of Gate 2. Fix them before spending any more rehearsal-scale
compute, the same way `docs/phase3.1-plan.md` §1 gated its own Gate 2 diagnosis behind fixing the
measurement machinery first.

### 1.1 `train.py`'s checkpoint-saving code is unreachable under `--max-steps` on a config whose epoch is longer than the requested step count — **blocking**

`train.py`'s `train()` function (`training/train.py:270`) writes `epoch_NNN.pt`/`best.pt`/`last.pt`
in a block that runs *after* the per-epoch `for step, batch in enumerate(train_loader)` loop
(`:409`) completes. But the `--max-steps` early-exit (`:446-448`) is a bare `return` **from inside**
that inner loop, the instant `global_step >= max_steps`. On the shipped `config.mps.yaml`
(`batch_size: 64`, 368 train regions × 100 chunks/region ÷ 64 ≈ 575 steps/epoch), a
`--max-steps 400` run — the exact invocation the Gate 2 sweep needed — returns at step 400, never
reaching line 486, and **writes zero checkpoints**. Training logs normal-looking progress the entire
time; nothing errors. This is the same failure shape §1.1's epigraph above is quoting almost
verbatim, just relocated from the loss config to the training loop.

The Gate 2 sweep that already ran worked around this **without touching `train.py`**: every sweep
config set `batch_size: 92` so that `36800 samples ÷ 92 = 400.0` batches exactly, `num_epochs: 1`,
and omitted `--max-steps` entirely, letting the loop exit naturally into the checkpoint block. That
workaround is legitimate for what it evaluated, but it is not a fix, and every other rehearsal/smoke
invocation in this repo's own documentation — `docs/phase3.1-plan.md` §1.1's own
`--regions 400 --max-steps 50`, `docs/CHANGELOG.md`'s `run_sweep.sh` (`--max-steps 400` against the
*unmodified* `config.rehearsal.yaml`) — uses `--max-steps` against a config where it silently
produces zero checkpoints today.

**Do:** move (or duplicate) the checkpoint-writing block so it also runs on the `--max-steps`
early-exit path, not only at natural epoch boundaries. The simplest correct fix: before the
`return` at `train.py:448`, run the same validation + checkpoint-save sequence that currently lives
after the inner loop (factor it into a helper if that keeps the diff small). Add a regression test —
train.py has no test coverage of the checkpoint path at all right now — that runs a couple of steps
with `--max-steps` set below one epoch's length and asserts `best.pt`/`last.pt` exist afterward.

### 1.2 The `occupancy_weight` / `consistency_weight` sweep axes are confounded by `overhang_weight`'s broken default — re-sweep needed

`docs/phase4-plan.md` §3.4 specifies sweeping the three weights; the Gate 2 run did so
one-factor-at-a-time, holding the other two at their shipped defaults while varying one. That is a
reasonable design *if the shared baseline point is itself sane*. It isn't: the shipped
`overhang_weight: 1.0` (identical across `config.cuda.yaml`, `config.mps.yaml`,
`config.rehearsal.yaml`) is an unnormalized per-chunk-count MSE with an observed magnitude around
243 at initialization, against ~1–5 for every other loss term — confirmed in the Gate 2 training logs
as a live optimizer instability at step 150 (terrain loss 3.8→46.1 in one step while overhang crashed
239→9.1). Every `occupancy_weight` and `consistency_weight` sweep point inherited this instability,
because the OFAT design held `overhang_weight` at 1.0 for both of those axes. The result: the shared
baseline scoring best within each of those two sweeps is more likely an artifact of that shared
confound than evidence 1.0/0.25 are good values.

**Do:** re-sweep `occupancy_weight` and `consistency_weight` holding `overhang_weight` at the
corrected value §2.2 below settles on — not at 1.0. Until that re-sweep exists, treat every
`occupancy_weight`/`consistency_weight` number currently in `docs/phase4-plan.md` §9 as provisional,
not a conclusion.

### 1.3 Gate 2's default `--baseline-margin` may be miscalibrated for this dataset — flag, don't silently retune

`docs/phase4-plan.md` §7's go/no-go condition 1 requires the model's multi-run column rate to clear
the marginal-probability baseline by **≥5.0 points** (`diagnose_overhangs.py --baseline-margin`,
default 5.0). Measured against the full 400-region rehearsal set: ground truth's own multi-run column
rate is 2.9833%, against a marginal baseline of 0.0000% — a **+2.98 point** margin. A model that
reproduced this dataset's ground truth *exactly* would not clear condition 1 at the default margin.

This did not change the Gate 2 verdict (every trained checkpoint scored +0.00 points, nowhere near
either bar), so it is not the reason Gate 2 failed. But it means condition 1's 5.0-point default —
presumably calibrated against a denser archetype mix, or against the full 8000-region distribution
rather than the 400-region rehearsal sample — needs a decision before anyone tunes toward it in
isolation on rehearsal-scale data.

**Do:** when §2.2's confirmatory run produces a non-zero multi-run rate for the first time, re-check
this margin against the *rehearsal* ground truth's own 2.98-point figure, not blindly against the
5.0-point default. If the rehearsal sample's archetype mix is genuinely sparser in carve-heavy
archetypes than the full dataset will be, that is a §2's-generator sampling question
(`docs/phase4-plan.md` §2.1's archetype table), not a go/no-go tuning question — do not lower the bar
to make a rehearsal-scale pass easier without checking which explanation is true first.

### 1.4 Gate 1's real exit criterion has never been executed — not blocking Gate 2, but still open

`docs/phase4-plan.md` §7: *"Export a graph whose occupancy head is replaced by a hand-written
deterministic 3D carve, package it, load it in Minecraft, walk under the overhang... If the
screenshot at the end of Gate 1 is right, the rest of the plan is a training problem rather than an
engineering one."* Everything upstream of the screenshot is verified — the plumbing produces a real,
correctly-dressed, walkable overhang when run through ONNX Runtime directly — but the screenshot
itself has never been taken. This requires a human at a keyboard (or desktop GUI automation this
environment doesn't have); it does not block anything in this document, but it is the one Gate 1
deliverable still owed. See §2.5.

---

## 2. Next steps — getting Gate 2 to a verdict that means something

`docs/phase4-plan.md` §7's own text, quoted in this document's goal statement, is the governing
instruction here: diagnose before spending more compute. The order below is deliberate — cheapest and
most likely to explain the result first.

### 2.1 Fix `train.py`'s checkpoint bug (§1.1) before any further training

Small, mechanical, and every later step in this section depends on checkpoints actually existing. Add
the regression test described in §1.1 so this class of bug can't recur silently a second time — it is
exactly the "logs normal, ships nothing" failure shape this project has hit before
(`config.cuda.yaml`'s GradScaler collapse, cited in `docs/phase3.1-plan.md` §1.1; the terracing bug
in `docs/phase3.1-plan.md` §2.1).

### 2.2 One confirmatory long run at corrected `overhang_weight`

`docs/phase4-plan.md` §7 specifies 400 steps for Gate 2; the Gate 2 run's own analysis (recorded in
`docs/CHANGELOG.md`) found the occupancy loss was still falling fast at step 400 — from ~0.72 toward
~0.03–0.07 within 250–300 steps, well below what a purely marginal-collapsed predictor could
achieve — and roughly half those steps were inside `warmup_steps`. That is *some* evidence for
"undertrained," not "structurally incapable," but it is not yet distinguishable from the data on
hand.

**Do:** one run at 2,000–4,000 steps, `overhang_weight` in the sweep's best-supported range
(0.003–0.01 — relief retention 19.8–20.7% there, comparable to baseline's 22.0%; biome accuracy
77.6–79.4%, clearly better than baseline's 62.8%), other weights at shipped defaults pending §1.2's
re-sweep. Evaluate with the same `scripts/diagnose_overhangs.py --checkpoint` invocation the Gate 2
sweep used, plus the two ad hoc scripts it wrote for conditions 3 and 5 (mean-floor-area retention;
relief retention / biome accuracy — `diagnose_overhangs.py`'s own `evaluate_baseline_gate()` only
implements 3 of §7's 5 conditions, a gap worth closing in the tool itself, not just working around
per-run — see §2.4).

**Decision point:** if multi-run columns are still exactly 0.0000% at this length, treat that as a
real negative result, not a "needs more steps" excuse to keep extending — move to §2.3.

### 2.3 If §2.2 still produces zero multi-run columns: structural diagnosis

`docs/phase4-plan.md` §3.2 states `OverhangLoss` has *"a known stationary point at a bit-exactly
uniform prediction... measure-zero, a local maximum, unreachable by a real conv stack"* and directs
pinning that behavior in `test_model.py` "the way `ReliefLoss`'s is" — done, and the pinned test
passes, which rules out the *literal* uniform-prediction stationary point as the mechanism. It does
not rule out a *nearby* one. In order of cheapest to check:

1. **Initialization.** Does the occupancy head start closer to "always air" or "always solid" than to
   the marginal probability the ground truth's ~0.38% overhang-cell rate would imply? A head that
   initializes near a boundary condition may need many more steps to escape it than one initialized
   near the true marginal.
2. **Gradient flow through the y-upsample.** §1.3's `ConvTranspose3d` stride-4 y-upsample is the one
   piece of the architecture with no dedicated unit test beyond the forward/backward dry-run in
   `test_model.py`. Check whether gradients reaching the pre-upsample 16-level tensor are
   meaningfully smaller than gradients reaching the 2D backbone at the same depth — a systematically
   starved upsample path would explain "learns something (falling occupancy BCE) but never enough to
   flip a single cell's threshold."
3. **Loss balance at the *cell* level, not just the *run* level.** §3.1 already anticipates this:
   *"Most band cells are trivially solid (deep) or trivially air (high); those dominate an unweighted
   mean. Weight toward cells near a solid/air transition in the target."* Confirm
   `OccupancyLoss`'s transition weighting (`training/losses.py`) is actually up-weighting by enough,
   given ground truth's own transition-cell rate is well under 1% of the band (§9: overhang cells
   0.38% of band cells) — the "weight toward hard examples" mechanism can be present in the code and
   still be too weak in practice.
4. **Only after 1–3 are checked:** consider whether §1.2's architecture (halo 9, 3× valid conv3d,
   192ch backbone) is capacity-starved specifically for the occupancy task at rehearsal data volume,
   as distinct from the `docs/phase4-plan.md` §4 capacity sweep already run (which varied
   `conv_channels` alone, at the same broken `overhang_weight`, and found no signal — confounded by
   §1.2 the same way the weight sweeps were). A clean capacity read needs the corrected weight first.

**Do not** reach for a different loss family or representation (§1.2's rejected alternatives — 
multi-layer heightmaps, SDF/implicit surfaces) before exhausting this list. Nothing so far indicates
the *representation* is wrong; everything so far indicates either an undertrained or unbalanced
*objective* on top of a representation that Gate 1 already proved is expressive and exportable.

### 2.4 Close the go/no-go tooling gap while you're in `diagnose_overhangs.py` anyway

`evaluate_baseline_gate()` currently implements conditions 1, 2, and 4 of §7's five and prints its own
`VERDICT` as if that were the complete answer. The Gate 2 sweep computed conditions 3 and 5 by hand,
correctly, but externally to the tool — a future near-miss sweep (unlike this uniform-zero one) could
easily trust the tool's printed verdict and skip a condition that would have failed it. Fold both
missing conditions into `evaluate_baseline_gate()` itself, sourcing relief retention from wherever
§2.2 script keeps it and mean-floor-area retention from the existing `floor_cells`/`void_count` keys
already in the JSON output (per the Gate 2 write-up: `aggregate_overhang_metrics` omits
`mean_floor_cells_per_void` when pooling multiple region shards, so it must be *derived* from the
pooled totals, not read as a key that may not exist).

### 2.5 Gate 1's screenshot

Independent of everything above and not gated by it. Build (or reuse) `imaginator-gate1.mcim`, launch
the client, drop the model into `mcimagine/models`, create a world, and get under the overhang.
`docs/phase4-plan.md` §9 has a row waiting for exactly this (`Walk-under screenshot, hand-carved
volume`). This can happen at any point relative to §2.1–2.4; it's listed last here only because it
doesn't touch code.

---

## 3. Sequencing

**Step A — the blocking fix.** §1.1 (checkpoint bug) + its regression test. Nothing below is
trustworthy without it, the same way `docs/phase3.1-plan.md` Gate 1 required per-term logging before
any sweep result could be read.

**Step B — the confirmatory run.** §2.2, at the corrected `overhang_weight`. This is still
rehearsal-scale — no CUDA box needed, same as the original Gate 2 (`docs/phase4-plan.md` §0.1: *"the
GPU is the bottleneck and §7's Gate 0 and Gate 1 need no GPU at all"* — Gate 2 turned out to be
MPS-feasible too, and stays that way here).

**Step C — branch on B's result.**
- **Multi-run columns appear, even a little:** proceed to §1.2's re-sweep (occupancy/consistency
  weights, now at a sane `overhang_weight`) and §2.4's tooling fix, then re-run the full
  `docs/phase4-plan.md` §7 five-condition conjunction properly. If it passes: Gate 2 is a genuine go,
  and `docs/phase4-plan.md` §7's Gate 3/Gate 4 text applies unchanged — this document has nothing
  further to add.
- **Multi-run columns are still exactly zero:** work through §2.3's diagnosis list in order. Do not
  start Gate 3 while this branch is open, per `docs/phase4-plan.md` §7's own explicit instruction:
  *"If any condition fails, stop and diagnose — do not start the full run."*

**Step D — Gate 1's screenshot (§2.5).** Unblocked at any point; do it whenever convenient.

**Explicitly unchanged from `docs/phase4-plan.md`:** the architecture (§1.3), the ground-truth
generator (§2), the ONNX export rules (§1.5), the spec/versioning mechanics (§6, already shipped),
and Gate 3/Gate 4 themselves (§7). This document's scope is entirely "why did Gate 2 fail and how do
we get a trustworthy answer" — it is not a redesign.

---

## 4. Measurements log

Carried forward from `docs/phase4-plan.md` §9 where already measured; new rows are what Phase 4.1
adds. Empty rows are the work — same convention as both prior plan documents.

| Date | Step | Measurement | Result | Conclusion |
|---|---|---|---|---|
| 2026-07-31 | (context) | Gate 2, 14-checkpoint sweep, multi-run columns | 0.0000% on all 14 | See `docs/phase4-plan.md` §9 for the full table; this document exists because of this row |
| 2026-07-31 | (context) | Ground truth's own multi-run rate vs. marginal baseline (400 rehearsal regions) | +2.98 pts (2.9833% vs. 0.0000%) | Below the default 5.0-pt bar — §1.3 |
| 2026-07-31 | (context) | Baseline-config training instability | terrain 3.8→46.1, overhang 239→9.1 at step 150 | Root cause candidate for uniform-zero result — §1.2 |
| 2026-07-31 | A | `train.py` checkpoint-on-`--max-steps` regression test | added, green (`model/tests/test_train.py`) | Confirms §1.1's fix — and this run's own `--max-steps 3000` wrote `epoch_NNN.pt`/`last.pt` correctly, a second confirmation |
| 2026-08-05 | B | Confirmatory run, `overhang_weight: 0.005`, 3000 steps (`config.mps.yaml`, 400 regions): multi-run column % | **0.0000%** | Branch point for Step C — **zero.** Full gate: all of conditions 1/2/3/4 NOT MET, VERDICT FAIL (`scripts/diagnose_overhangs.py --checkpoint training_runs/mps_run1/last.pt --ground-truth model/data`) |
| 2026-08-05 | B | Same run: occupancy loss curve past step 400 | falls 0.72→~0.025 by step ~450, then plateaus in [0.016, 0.05] through step 3000 with no further descent | **Structural, not undertrained** — a genuinely-still-learning loss would show further descent over 2,550 additional steps; this one is flat. Resolves §2.2's "not yet distinguishable" note |
| | C (if B non-zero) | Re-swept `occupancy_weight` at corrected `overhang_weight` | not applicable | B was zero — see the row below instead |
| | C (if B non-zero) | Re-swept `consistency_weight` at corrected `overhang_weight` | not applicable | B was zero — see the row below instead |
| | C (if B non-zero) | Full 5-condition go/no-go, corrected config | not applicable | B was zero — see the row below instead |
| | C (if B zero) | Occupancy head init bias (air/solid/marginal) | **not yet done** | §2.3 item 1 — next step, out of scope for `docs/remote-training-readiness-plan.md` (see that doc's header) |
| | C (if B zero) | Gradient magnitude, pre-upsample vs. 2D backbone at matched depth | **not yet done** | §2.3 item 2 |
| | C (if B zero) | Transition-cell BCE weight, effective magnitude vs. ground-truth transition rate | **not yet done** | §2.3 item 3 |
| | D | Walk-under screenshot, `imaginator-gate1.mcim` | | `docs/phase4-plan.md` §9's still-blank Gate 1 row |
| 2026-08-06 | §1.3 follow-up (`docs/phase4.2-plan.md` §1.4) | `CARVE_THRESHOLD` retuned -0.35 → -0.24 (`world_generator.py`), `model/data` regenerated at 400 regions, seed 0. New ground truth's own multi-run rate vs. marginal baseline | +10.14 pts (10.1363% vs. 0.0000%) | **Decision: leave `--baseline-margin` at the 5.0 default, unchanged.** §1.3 flagged the margin because ground truth's own +2.98 pts couldn't clear 5.0 — that was a density problem, not a margin-calibration problem. The retuned generator's ground truth now clears the default margin by more than 2x with no change to the gate itself, so the correct fix was raising density (`docs/phase4.2-plan.md` §1.1), not loosening the bar — resolves §1.3's parked question in favor of "the gate was right; the data was thin." |
