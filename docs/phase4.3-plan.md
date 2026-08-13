# Phase 4.3 — Resolve Learnability, Fix the Objective/Gate Mismatch, Clear the Rental Decision

**Purpose.** Phase 4.2 fixed the dataset, the configs, and the loss *shapes*, and Gate 2 still
failed — but for a reason no prior phase had named. Measured (`phase4.2-plan.md` §6.1): the
objective prefers real caves to speckle by three orders of magnitude, the gradients are clean, and
the model nonetheless emits sub-surface air at **chance placement** (Pearson r = 0.031 against a
0.0001 control). It learned the marginal rate, not the function. This document resolves whether
that mapping is learnable at all, fixes the objective so that imperfect placement still yields
cave-shaped output, and terminates in an explicit rent / don't-rent decision.

**What this document can and cannot promise.** The engineering readiness is guaranteeable and is
finished here (§6). The learnability question is research: it has an unknown answer, and no plan can
promise it resolves in our favour. What this plan *does* guarantee is that **every branch terminates
in a decision** — either a green light with evidence, or a named, bounded architectural finding —
rather than in another sweep. A plan that promised Gate 2 would pass would be lying; the last two
phases each ended in a surprise, and both surprises were upstream of what the plan was tuning.

**Scope.** Supersedes `phase4.2-plan.md` §5's stop-condition framing (its remaining sweep slots are
withdrawn — see §2). Inherits §6.1's measurements as settled. Unlike 4.1 and 4.2, this document is
permitted to change the architecture, but only along the paths §3 names and only after §3.1 says
which one.

---

## 1. Settled — do not re-derive

Carried from `phase4.2-plan.md` §1, §2, §6.1. All measured, all reproducible with committed tools.

| Claim | Evidence | Status |
|---|---|---|
| Head can represent overhangs | overfit to 100% acc, 42.6% multi-run reproduced exactly | **Closed** |
| Objective prefers real caves to speckle | real 0.0005, heightfield 1.3903, speckle 1.5415 | **Closed** |
| Gradients are well-scaled, no dead zone | occ 0.0066–0.0077, 4.0×ovh 0.0007–0.0008, 0 dead cells | **Closed** |
| Coordinate encoder covers the carve field's dominant octaves | encoder λ ∈ {2048…8}; carve base λ=102.4, finest octave λ≈12.8 at `carve_frequency` 5.0 | **Closed for the dominant octaves; marginal at the finest — see 3.3b** |
| Dataset can clear its own gate | ground-truth multi-run 10.14%, quality bounds hold | **Closed** |
| Model did not learn placement | sub-surface r = 0.031, IoU 0.015, all cavities exactly 1 block tall | **The open problem** |

**Two things are explicitly withdrawn as next steps.** Sweeping `extra_transition_weight` or
`overhang_weight` further: both change the *rate* of speckle, not its randomness, and neither can
move r = 0.031. `phase4.2-plan.md` §5's "two settings remain" is void.

---

## 2. The reframe — why the model is behaving correctly and still failing

Per-cell BCE is a **placement** objective: it asks "is *this* cell air?" Gate 2's conditions are
**shape** conditions: cavity height ≥ 3, walkable-void retention, floor-area retention. Nothing in
Gate 2 requires a cave to be in the right place; it requires caves to be cave-shaped.

When placement is hard — and a 3D FBM field at block resolution is hard — the placement-optimal
prediction is the per-cell **marginal**. Thresholding a near-uniform marginal at 0.5 produces
isolated 1-block air cells: the single worst possible *shape*, and exactly what was observed
(3,331 cavities, every one of them mean = p50 = p90 = max = 1). The training log confirms the
mechanism: the Stage 4 tripwire on a **fixed** batch swings 0% → 56.8% → 23.5% → 87.5% → 0% within
a few hundred steps, which is what a globally near-threshold field does and what a model with
learned placement cannot do.

So two independent conditions must both hold:

- **(A) Placement must be learnable to a non-trivial degree** — enough that the model is not purely
  hedging. §3 determines whether it is.
- **(B) The objective must reward coherent 3D shape**, so that *imperfect* placement still produces
  caves rather than speckle. §4 fixes this.

(A) and (B) are independent, and §3 and §4 may be worked in parallel. That is deliberate: it splits
the risk, and §4 has value even if §3's answer is disappointing.

---

## 3. Stage A — the learnability ladder 🔒

The decisive experiment, and the cheapest thing in this document. No rental, no full run.

### 3.1 The single-region overfit — do this first, before anything else

| # | Item | Done when |
|---|---|---|
| A.1 | 🔒 New script `scripts/learnability_probe.py`. Train the **full `ImagineNet`** (not just the head, not random frozen features) on **one** carved region's chunks, through the **real conditioning path** — real prompt tokens, real world seed, real global chunk coords, real `CoordinateEncoder`. Occupancy loss only: set every other head's weight to 0.0 so nothing competes. ≤4000 steps, CPU or MPS. | Reports sub-surface Pearson r **on that same region** (train r — memorisation is the goal here, not generalisation), plus predicted vs. target cavity-height distribution. |

**Success: r ≥ 0.7.** This is the fork the whole plan turns on:

- **r ≥ 0.7 → placement is learnable.** The mapping is expressible and reachable by SGD; Phase 4.2's
  failure was budget/scale, not design. Go to 3.2.
- **r < 0.3 → the model cannot learn this mapping even when memorising a single region.** No amount
  of data or steps fixes that. Go to 3.3.
- **0.3 ≤ r < 0.7 → partial.** Treat as the r < 0.3 branch (go to 3.3) but run 3.2 in parallel; the
  answer is likely "expressible but badly conditioned."

**Why this and not `test_occupancy_capacity.py`:** that test feeds the head *random frozen features*
and overfits 8 chunks. It proves the head can **represent** structure. It says nothing about whether
the network can **learn placement from coordinates**, which is the actual failure. `phase4.2-plan.md`
§2.1 over-credited it, and this plan's §1 records it as narrower.

### 3.2 The scale ladder — only if A.1 succeeds

| # | Item | Done when |
|---|---|---|
| A.2 | 🔒 Repeat A.1's protocol at **1 → 8 → 64 → 400** regions, fixed step budget per region (so total steps scale), recording train r **and** held-out r at each rung. | A table of r vs. region count, both splits. |

This is the rental decision in numeric form. If train r stays high while held-out r collapses, the
model is memorising and 8000 regions will not generalise — that is a capacity/conditioning finding,
not a budget one. If both degrade gracefully, extrapolate to 8000 regions and report the projected r;
that projection is what justifies (or refuses) the spend.

### 3.3 Localisation — only if A.1 fails

Run in this order and stop at the first one that moves r. Each is a single A.1 re-run.

| # | Probe | What a positive result means |
|---|---|---|
| A.3a | 🔧 **Oracle-features probe.** Feed `Occupancy3DHead` the true carve-field value at each cell as an extra input channel, bypassing the backbone. | If r jumps, the head is fine and the deficit is upstream: the backbone/encoder cannot *deliver* the features the head would need. |
| A.3b | 🔧 **Raise the encoder's frequency ceiling.** `DEFAULT_WAVELENGTHS` currently floors at λ = 8 blocks. Add λ = 4 and λ = 2 (+8 channels, 36 → 44). | The carve field's finest octave is λ ≈ 12.8 at `carve_frequency` 5.0 but ≈ 4.6 at 14.0 — below the current floor. If r moves, fine-scale coverage was the binding constraint. Cheap; try before A.3c. |
| A.3c | 🔧 **Explicit y conditioning.** The encoder encodes (x, z) only; the head decodes y positionally from a 2D feature map. Give the head Fourier features in y. | If r moves, the y axis was the weak axis and the 16→64 `ConvTranspose3d` was carrying more than it can. |

If none of the three moves r above 0.3: **stop.** That is a genuine architectural finding, it is
reportable as a successful outcome of this plan, and it means the current design cannot produce
volumetric terrain. Do not rent. Do not attempt a fourth probe without a new decision.

---

## 4. Stage B — fix the objective/gate mismatch 🔒

Independent of §3. Do this even if §3 is still running.

### 4.1 Pin the objective's preference ordering as a test — before changing any loss

| # | Item | Done when |
|---|---|---|
| B.1 | 🔒 New test `tests/test_objective_ranking.py`. Builds four synthetic predictions against real ground truth — **real caves**, **cave-shaped but displaced** (target's own structure shifted 8 blocks in x), **1-block speckle at matched multi-run rate**, **heightfield-only** — and asserts the total objective ranks them. | Test committed and passing against the **current** objective for the parts that already hold. |

Measured today, the current objective gives: real **0.0005** < heightfield **1.3903** < speckle
**1.5415**. Speckle is already ranked worst — good, and the test must lock that in.

**The missing property is the displaced-cave case.** Under pure per-cell BCE, a plausible cave in
the wrong place scores about as badly as speckle: both are near-chance per-cell. That is precisely
the mismatch in §2 — the objective cannot tell "right shape, wrong place" from "no shape at all",
while Gate 2 cares enormously about the difference.

**Required after B.2:** `real < displaced < speckle`, with `displaced` closer to `real` than to
`speckle`. That single inequality is the acceptance criterion for this entire stage.

### 4.2 Add the shape term

| # | Item | Done when |
|---|---|---|
| B.2 | 🔒 Add a **vertical-coherence** term to `losses.py`. Differentiable 1-tall-cavity count: `p_air(y) · p_solid(y−1) · p_solid(y+1)` summed per chunk; penalise its **excess over the target's own rate** (not its absolute value — ground truth has some 1-block cavities and the term must not forbid them). Keep the existing cummax/flip machinery untouched. | B.1's inequality holds; `loss_mass.py` reports the new term's share. |
| B.3 | 🔧 Optionally extend to a soft **cavity-height histogram** match if B.2 alone does not satisfy B.1. | Same criterion. One escalation only. |
| B.4 | 🔧 Re-tune weights via `loss_mass.py`, all configs. | Each term's share at a heightfield-only prediction inside its band; occupancy remains dominant. |

**Do not** implement B.2 by penalising all isolated air — that forbids ground truth's own structure
and simply re-creates the heightfield optimum from the other direction. The term is an *excess*
penalty, and B.1's `real` case is what proves it (real caves must stay near zero loss).

---

## 5. Stage C — confirmatory run and Gate 2

Only after §3 has answered and §4's B.1 inequality holds.

| # | Item | Done when |
|---|---|---|
| C.1 | 🔧 Rehearsal-scale run, 400 regions, tripwire on. | Tripwire shows **stable** non-zero structure — not the 0%/87.5% oscillation. A swinging tripwire means stop and return to §3; it is the marginal-hedge signature. |
| C.2 | 🔒 Full Gate 2, `diagnose_overhangs.py --checkpoint … --ground-truth model/data`. | **VERDICT: PASS** on all five conditions, reproducibly. |
| C.3 | 🔧 Re-measure sub-surface r on the trained checkpoint, **through `McImagineDataset`**. | r reported alongside the gate verdict. A gate pass with r ≈ 0 means the gate was satisfied by shape statistics alone — record it explicitly; it is a real (if weaker) result and changes what the model is claimed to do. |

**Gate 2 is not to be loosened, and the diagnosis sample is not to be narrowed.** If a condition
fails, that is the finding.

---

## 6. Stage D — readiness engineering (guaranteeable; finish regardless of §3's answer)

| # | Item | Done when |
|---|---|---|
| D.1 | 🔧 `preflight` clean on **every** config for a CUDA target, except the known Darwin-only failures (P2/P9/P11, all one root cause: they launch `train.py` against `device: cuda` on a Mac). | Documented per config, with the Darwin exemptions named individually rather than as a count. |
| D.2 | 🔧 Regenerate at **8000 regions** using the parallel path — but **only after C.2 passes**. At 6.62× this is ~2h, versus ~13h sequential. | Shard count, wall time, and P13 recorded. Regenerating before Gate 2 passes pays 20× to rediscover the same failure. |
| D.3 | 🔒 **Early-abort throughput check for the rented box.** P10 currently projects 60.8h at 8000 regions × 15 epochs — but that is an **MPS** number and the budget is 40h. Add a documented first-15-minutes procedure: measure steps/s on the rented hardware, project total, and abort the rental if it exceeds budget. | Procedure written into `docs/TRAINING.md` with the exact command and the abort threshold. |
| D.4 | 🔧 Confirm resume-from-checkpoint end to end on the rehearsal run (kill mid-epoch, resume, verify same epoch and loss continuity). | Already implemented in 4.2; this is the live rehearsal of it, not a re-implementation. |

D.3 is the one item that protects the money directly: it is the difference between discovering a
2× throughput miss in the first 15 minutes and discovering it 40 hours in.

---

## 7. Decision tree — every branch terminates

| Branch | Action |
|---|---|
| A.1 r ≥ 0.7, B.1 holds, C.2 PASS | **RENT.** Complete §6, then go. |
| A.1 r ≥ 0.7, B.1 holds, C.2 fails on shape conditions only | **One** bounded iteration of B.3. Then stop either way. |
| A.1 r ≥ 0.7, A.2 shows held-out r collapsing by 64 regions | **Do not rent.** Memorisation, not learning. Report the ladder. |
| A.1 fails, one of A.3a–c moves r above 0.7 | Apply that change, re-run A.1 once, then rejoin at 3.2. |
| A.1 fails, no probe moves r above 0.3 | **Stop, do not rent.** Architectural finding — report with curves. This is a successful outcome. |
| Any point where a 4th setting of the same hyperparameter is the proposed next step | **Stop and report.** |

Reporting "here is the ladder, here are the curves, this architecture cannot learn block-resolution
3D placement" is a **complete and valuable** result. It is worth far more than a rented GPU
reproducing speckle at scale.

---

## 8. Measurement log

| Date | Item | Measurement | Result | Notes |
|---|---|---|---|---|
| 2026-08-07 | (context) | Sub-surface placement r, `phase42_rehearsal/last.pt`, via `McImagineDataset` | 0.031 (control 0.0001) | §6.1 of 4.2 — the open problem |
| 2026-08-07 | (context) | Objective ranking: real / heightfield / speckle | 0.0005 / 1.3903 / 1.5415 | §2 — speckle already ranked worst |
| 2026-08-07 | A.1 | Single-region overfit, `scripts/learnability_probe.py`, region_00366 (34.56% multi-run), 4000 steps, flat LR 3e-4 | train r = 0.6686 (control 0.0003), IoU 0.529 | Partial bucket (0.3≤r<0.7) — loss still bouncing 0.26-0.44, not converged |
| 2026-08-07/08 | A.1 (extended) | Same protocol, 12000 steps, 300-step warmup + cosine decay to 0 | train r = **0.8836** (control 0.0002), IoU 0.804, lift 12.80x | **r≥0.7 — placement is learnable.** Occupancy loss fell smoothly 0.98→0.08, no oscillation. Anomalous wall-clock: 13.1h vs ~78min projected (throughput dropped from 2.6 to 0.25 steps/s partway through — machine sleep/throttle, not a code issue). Predicted cavity mean 8.33 vs target 10.36 (p50 6 vs 8) — real distributions, not degenerate. Per §7: proceed to §3.2 |
| 2026-08-08 | A.2 | `scripts/scale_ladder_probe.py`, rungs 1/8/64 regions (nested, most cave-dense pool minus a fixed 8-region held-out set), 300 steps/region, cosine warmup+decay | train r: 0.0100 / 0.4227 / 0.5124 — held-out r: 0.0169 / 0.0552 / 0.0481 | **Train r climbs steadily with scale; held-out r stays flat/near-chance at every rung, even ticking down 8→64.** Matches §7's "held-out r collapsing" branch — memorisation, not learning. 400/8000-region rungs not run locally (operator decision, bounded to 64); throughput held steady ~2.65 steps/s throughout (no repeat of A.1's anomalous slowdown, `caffeinate -i` used). Reported to operator before drawing the rent/no-rent conclusion — see conversation |
| | A.3a/b/c | | | |
| 2026-08-08 | B.1 | `tests/test_objective_ranking.py`, 128 real chunks at moderate density (~7% multi-run, brackets dataset's 10.14% average) | real<heightfield<speckle holds (regression guard, PASS); displaced vs speckle ordering FAILS pre-B.2 (XFAIL, displaced scores *worse* than speckle: 2.2027 vs 1.9987 raw occ+ovh) | Confirms §4.1's diagnosis directly: per-cell BCE cannot tell "right shape, wrong place" from noise — a coherently-shifted full cave racks up MORE mismatched cells than sparse speckle does |
| 2026-08-08 | B.2 | `VerticalCoherenceLoss` added to `losses.py`; weight sweep on same 128-chunk batch, `test_objective_ranking.py` | ordering (`real<displaced<speckle`) closes at weight≈240 (shipped at 300, headroom); "displaced closer to real than speckle" needs weight≈2826 | **Operator decision (2026-08-08): ship at 300 (ordering only), record "closer" as an open, unresolved finding** — at weight~2800 the term would outweigh OccupancyLoss/OverhangLoss 1000x and stop being a refinement term. Same conclusion expected to hold for B.3 (histogram term): root cause is per-cell BCE's own real-vs-imperfect magnitude gap, not the shape metric chosen. `pytest`: 103 passed, 1 skipped, 1 xfailed (matches §9 checklist's ≥103 target); `pytest -O`: same. `loss_mass.py` reports the new term's share (0% at heightfield-only by construction; 8.9-16.4% at a matched-rate speckle prediction across weight 150-300). Not shipped in either config yet — B.4 (retuning) not attempted |
| | C.1 | | | |
| | C.2 | | | |
| | D.1–D.4 | | | |

---

## 9. Ready-to-rent checklist

Renting is justified when **all** hold.

- [ ] §3 answered — A.1 reported with a number, and either r ≥ 0.7 or a named probe that fixed it.
- [ ] A.2's scale ladder shows held-out r surviving to 400 regions.
- [ ] B.1's inequality holds: `real < displaced < speckle`, displaced nearer real.
- [ ] Gate 2 **VERDICT: PASS**, reproducibly, un-loosened (C.2).
- [ ] Sub-surface r recorded for the passing checkpoint (C.3), whatever it says.
- [ ] `preflight` clean on every config, Darwin exemptions named individually (D.1).
- [ ] 8000-region dataset regenerated and P13 green (D.2).
- [ ] Rented-box early-abort procedure documented in `docs/TRAINING.md` (D.3).
- [ ] `pytest` and `pytest -O` green, ≥103 passed (101 + B.1 + A.1's regression guard).
- [ ] Everything committed.

---

## 10. Handoff notes

- **§3.1 first, and report before starting anything else.** It is a few hours and it determines
  which half of this document is real work. Do not begin §5 before it answers.
- §4 may proceed in parallel with §3 — they are independent.
- **Report per item with actual command output.** "A.1 implemented" without an r value is not done.
- Re-run `preflight` after each stage; hold `pytest` and `pytest -O` green throughout.
- §7's stop conditions are instructions, not advice. A terminating negative result is a success.
- **Do not loosen Gate 2, narrow the diagnosis sample, or re-scope a criterion to make it pass.**
- Any placement measurement goes through `McImagineDataset` — see §6.1's note in `phase4.2-plan.md`
  about the halo offset that corrupted the first attempt.
- Do not commit without asking.
