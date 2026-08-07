# Phase 4.2 — Make Overhangs Learnable, and Finish Everything That Can Be Done Locally

**Purpose.** Phase 4.1 corrected `overhang_weight` and re-ran Gate 2. It failed again, identically:
**0.0000% multi-run columns, mean 1.0000 transitions per column, max 1** — a pure heightfield.
This document explains *why* (it is not what Phase 4.1 assumed), fixes it, and closes out every
remaining item that can be done without renting a GPU.

**Definition of done.** Gate 2's five-condition go/no-go passes at rehearsal scale on corrected
data, reproducibly — and `preflight.py` exits 0 for a CUDA target on every config in the repo. At
that point renting is a purchasing decision, not a technical risk.

**Scope.** This document supersedes `docs/phase4.1-plan.md` §2.3 (see §2 below — two of its three
hypotheses are now ruled out by measurement). It inherits, and does not repeat,
`docs/remote-training-readiness-plan.md`'s issue register; the one unfinished item from that plan is
carried here as 0.1. It does **not** change the architecture, the ONNX plumbing, or Gates 0/1.

---

## 1. Why Gate 2 failed — measured, not hypothesized

### 1.1 The model converged to the exact optimum of the objective it was given

| Quantity | Value |
|---|---|
| Analytic occupancy BCE of a **perfect heightfield-only** prediction, measured on 512 real chunks | **0.02623** |
| The 2026-08-05 confirmatory run's observed plateau, steps ~450 → 3000 | **[0.016, 0.05]** |

The run did not stall, undertrain, or diverge. It **found the global optimum of its loss**, and that
optimum is a heightfield. Every remedy framed as "help it optimize better" is therefore aimed at the
wrong thing.

### 1.2 Why the optimum is a heightfield

Measured on 512 chunks drawn from `model/data`:

```
columns with >=2 transitions (overhang-bearing):   1.499%
OccupancyLoss weight mass on those columns:        1.733%
   ...on the extra transition cells themselves:    0.414%   <-- the entire overhang signal
```

`OccupancyLoss`'s `transition_weight=5.0` was intended to focus the model on solid/air boundaries.
But it computes `dy` over **all** transitions, and every column has a surface transition. So the
weighting overwhelmingly re-emphasises the heightfield the model already gets right. The transitions
that actually *constitute* an overhang carry **0.414%** of the gradient mass.

`OverhangLoss` is not a rescue, because it matches a **per-chunk scalar count** (`p_overhang.sum() /
256`) across 16,384 cells. It can express *how many* overhang cells a chunk should have and never
*where*, so its gradient is smeared uniformly over the chunk. This is why sweeping its weight from
0.0003 to 0.03 across 14 checkpoints changed nothing: the term was never the binding constraint, and
its shape — not its magnitude — is the defect.

### 1.3 The gate is unreachable with the current dataset — the finding that was parked

`docs/phase4.1-plan.md` §1.3 flagged this on 2026-07-31 and correctly declined to silently retune it.
It was never resolved, and it is now blocking:

| | multi-run column % |
|---|---|
| Ground truth, `model/data` (368 regions), measured 2026-08-05 | **1.82%** |
| Ground truth, 400-region rehearsal set (§1.3's figure) | 2.98% |
| Gate 2 condition 1 requirement | **≥ 5.0 points over baseline** |

Baseline is 0.0000%, so **a model that reproduced the ground truth exactly would score +1.82 and
fail condition 1.** Any tuning toward this gate on this data is tuning toward an unreachable target.
This must be resolved *before* the loss work, or the loss work has no success signal.

---

## 2. Ruled out — do not spend time here

`docs/phase4.1-plan.md` §2.3 lists three follow-ups for the "B is zero" branch. Two are now closed
by direct measurement:

- **§2.3 item 1 — occupancy head init bias.** Ruled out. See 2.1 below.
- **§2.3 item 2 — gradient magnitude, pre-upsample vs. 2D backbone.** Ruled out. See 2.1 below.
- **§2.3 item 3 — transition-cell BCE weight.** *Correct thread*, but insufficient as stated. §1.2
  measures the mass at 0.414%; a scalar re-weight of the existing (all-transitions) mask cannot fix
  the fact that the mask is dominated by the surface. Item 3 is superseded by 4.2 below.

### 2.1 The capacity evidence

`Occupancy3DHead` was overfit against 8 overhang-rich chunks (42.6% multi-run columns), random
frozen features, no regularisation:

```
step    0  BCE 0.75558  acc   7.70%  pred multi-run cols  0.0%  (target 42.6%)
step  300  BCE 0.00007  acc 100.00%  pred multi-run cols 42.6%  (target 42.6%)
step 1500  BCE 0.00000  acc 100.00%  pred multi-run cols 42.6%  (target 42.6%)
```

The head reaches **100% cell accuracy and reproduces the multi-run rate exactly**. The architecture —
including the `y_in=16 → 64` `ConvTranspose3d` upsample that §2.3 item 2 suspected — represents
overhangs perfectly and learns them in a few hundred steps when the objective asks for them.
Initialisation and gradient flow are not the constraint. **The objective is.**

Preserve this test (4.0 below). If a future change breaks it, the architecture regressed and nothing
downstream is worth debugging until it is green again.

---

## 3. Instruments — build these first, they are how every later item is judged

Both scripts below were written during diagnosis and must be promoted into the repo, because every
acceptance criterion in §4 is expressed in their output. Neither is a test; both are measurement
tools an operator runs.

| # | Item | Done when |
|---|---|---|
| 3.1 | 🔧 `model/src/mc_imagine_model/scripts/loss_mass.py` — reports, for a sample of real shards: multi-run column %, `OccupancyLoss` weight mass on overhang-bearing columns and on extra-transition cells, and each loss term's magnitude and share at a perfect heightfield-only prediction, for a range of weights. | Running it on `model/data` reproduces §1.2's three percentages within 0.1 pts. |
| 3.2 | 🔧 `model/tests/test_occupancy_capacity.py` — §2.1's overfit test, as a unit test, bounded (≤400 steps, CPU, <60s). | `pytest -k capacity` green, asserting final accuracy ≥99% and predicted multi-run % within 5 pts of target. |
| 3.3 | 🔧 Extend `preflight.py` with **P13 — dataset structure density**: reports ground-truth multi-run column %, and FAILs below `--min-multi-run-pct` (default 8.0). | P13 FAILs on today's `model/data` at 1.82%, with a message naming §1.3. |

**Why 3.3 is a preflight check and not a note:** the unreachable-gate problem survived a full phase
because it lived in a document instead of in a tool. Preflight is what actually gets run.

---

## 4. Stages

Each item has a **Done when** line: the command and the output that constitutes proof. An item is
not complete because the code changed; it is complete when its check passes.

**Decision authority.** 🔒 items are not open to redesign — deviating re-opens a failure already paid
for. 🔧 items are ordinary engineering judgement.

**Acceptance criteria are measurements, not file edits.** Where a criterion says "every config",
it means every config in `model/src/mc_imagine_model/training/`, verified by running the check
against each — not by editing the one file this document happens to name. (Phase 4.1's rehearsal
config kept a 56 GB shard cache for exactly this reason.)

### Stage 0 — carry-over, do first, ~10 minutes

| # | Item | Done when |
|---|---|---|
| 0.1 | 🔧 `shard_cache_size` in `config.rehearsal.yaml` (currently 512 with `num_workers: 8` → **56.49 GB**, the original OOM shape; rehearsal is the *first* thing that runs on a rented box). | `preflight --config <each config> --quick --target-regions 400` and `--target-regions 8000`: **P4 PASS or WARN on every config**, never FAIL. |

### Stage 1 — make the target reachable 🔒

**Do not start Stage 2 until Stage 1 passes.** Fixing the loss against 1.8%-density data gives no
usable success signal: the run can improve substantially and still measure ~0.

| # | Item | Done when |
|---|---|---|
| 1.1 | 🔒 Raise ground-truth overhang density to **≥8% multi-run columns** (from 1.82%). Knobs, in preference order: `CARVE_THRESHOLD` (`world_generator.py:373`, currently −0.35; toward 0 carves more), then per-archetype `carve_strength` / `undercut_strength` / `arch_strength` / `cave_mouth_strength`. **Do not** change `carve_frequency` or `hoodoo_frequency` — both are PINNED to the `CARVE_FREQ_GRID` quantizer and are not amplitude knobs. | `diagnose_overhangs.py --ground-truth <new data>` reports multi-run columns **≥8.0%**; preflight **P13 PASS**. |
| 1.2 | 🔒 **Quality guard — density must not become mush.** The failure mode of turning carve up is speckle/shattering, which scores well on cell counts and is unplayable. `docs/phase4-plan.md` §7's discussion of the speckle/shattering fixtures is required reading before touching any knob in 1.1. | On the new data: solid fraction stays in **[0.93, 0.98]**; p90 cavity height **≥3** and max **≤48**; `clipped_at_band_floor` **= 0**; walkable voids per 1000 columns **≥ today's 0.685**. All from the same `diagnose_overhangs.py --ground-truth` run. |
| 1.3 | 🔧 Regenerate `model/data` at 400 regions with the corrected generator, then re-record the P7 fixture. | `preflight` **P7 PASS**, **P8 PASS** (targets still inside the head's representable band), **P13 PASS**. |
| 1.4 | 🔧 Re-check Gate 2 condition 1's `--baseline-margin` against the new ground truth, per §1.3's "Do" clause. Record the decision in `docs/phase4.1-plan.md`'s log table. | The margin is either left at 5.0 with ground truth now clearing it, or changed **with the measured justification written down** — never silently retuned. |

### Stage 2 — do F3 now: it moved onto the critical path 🔒

| # | Item | Done when |
|---|---|---|
| 2.1 | 🔒 Parallelise `generate_shards` (`world_generator.py:1898`) with `ProcessPoolExecutor`. Filed in the readiness plan as cost optimisation; Stage 1 puts dataset regeneration in the **inner loop**, so this is now ~8 h vs ~35 min *per iteration*. Region cost is bimodal (carved ~8.0 s, uncarved ~0.33 s) — chunk by region, do not split by contiguous range, or workers finish wildly unevenly. | **P7 PASS against the fixture, generated via the parallel path** — byte-identical shards or it did not work. Measured speedup recorded in this document's §6. |

### Stage 3 — fix the objective 🔒

The two changes below are the substance of this plan. Both are in `model/src/mc_imagine_model/training/losses.py`.

| # | Item | Done when |
|---|---|---|
| 3.1 | 🔒 **`OccupancyLoss` — weight *extra* transitions, not all transitions.** Every column has a surface transition; today's mask spends its weight there. Separate the topmost transition per column (the surface) from all others, and weight them independently (`transition_weight` for the surface, a new, larger `extra_transition_weight` for the rest). Keep the existing `transition_weight` parameter and default so nothing silently changes for other callers. | `loss_mass.py` reports **≥20%** of `OccupancyLoss` weight mass on overhang-bearing columns, up from 1.733%. |
| 3.2 | 🔒 **`OverhangLoss` — localise it.** Replace the per-chunk count MSE with a **per-cell BCE** between the predicted overhang-probability field and the target overhang-indicator field. Both fields are already computed inside the current `forward` (`p_overhang` and `target_overhang`) — this is a change of what they are compared *with*, not new math. **Keep the cummax/flip construction exactly as-is**; it is correct and differentiable. | The term produces a spatially-varying gradient: on a batch where a single chunk's overhang is moved, the loss changes. Add a unit test asserting that. Count-matching alone must no longer be satisfiable. |
| 3.3 | 🔧 Re-tune `overhang_weight` for the new term's scale. Its old range [0.003, 0.01] was calibrated against a count MSE and is **meaningless** for a BCE. Start at 1.0 and sweep. | `loss_mass.py` shows the overhang term at **15–40%** of total loss at a heightfield-only prediction — present, not dominant. |
| 3.4 | 🔒 `consistency_weight` stays **0.0**. Unchanged from the readiness plan §5 item 0.2 — a corrected consistency loss is separate research, not part of this fix. | `grep consistency_weight` shows `0.0` in every config. |

### Stage 4 — see failure in 2 minutes instead of 35 🔧

| # | Item | Done when |
|---|---|---|
| 4.1 | 🔧 **Structure tripwire.** On a fixed validation batch, every `log_every` steps, log predicted **multi-run column %** (threshold sigmoid at 0.5, count columns with ≥2 y-transitions) alongside the target's. Cache the batch once; this must not add a data-loading path. | A rehearsal run's log shows the metric from step `log_every` onward, and the value is comparable to `diagnose_overhangs.py`'s on the same checkpoint. |
| 4.2 | 🔧 Abort guidance in the log: if the metric is still 0.0000 at step 1000, the run is a heightfield and will stay one. | A one-line WARN fires at step 1000 when the metric is zero, naming this section. |

### Stage 5 — the confirmatory run

| # | Item | Done when |
|---|---|---|
| 5.1 | 🔧 Overfit check, then short run, then full rehearsal — **in that order**, stopping at the first zero. | 3.2's capacity test green → 400-region / 500-step run shows **nonzero** multi-run % on the tripwire → full rehearsal run. |
| 5.2 | 🔒 Gate 2 five-condition go/no-go on the rehearsal checkpoint. | `diagnose_overhangs.py --checkpoint <last.pt> --ground-truth model/data` prints **VERDICT: PASS**. Record the full table in `docs/phase4.1-plan.md`'s log. |
| 5.3 | 🔧 Full regression. | `pytest tests/ -q` **and** `python -O -m pytest tests/ -q` both green, count ≥ the 98 passed / 1 skipped baseline. |

### Stage 6 — final readiness sweep

| # | Item | Done when |
|---|---|---|
| 6.1 | 🔧 `preflight --config config.cuda.yaml --budget-hours 40` on this Mac. | Everything PASS/WARN except **P2** (correctly FAILs: no CUDA here) and **P5** (correctly SKIPs: no `/dev/shm` on Darwin). **P10's projection is measured on MPS and is not the CUDA number** — record it, do not gate on it. |
| 6.2 | 🔧 Regenerate the full 8000-region dataset *only if* Stage 2.1 makes it cheap enough to be worth having ready. Otherwise leave it for the rented box and say so. | Either the data exists and P3/P4/P7/P8/P13 pass at 8000 scale, or this document records the explicit decision to generate remotely. |
| 6.3 | 🔧 Commit everything, including `model/tests/fixtures/shard_hashes_seed0.json`. | `git status` clean; P7 is a real cross-platform comparison rather than a local recording. |

**6.2 decision, recorded 2026-08-07: deferred to the rented box, not generated locally.** Stage
2.1 makes the mechanics cheap (6.62x speedup measured; an 8000-region run extrapolates to roughly
2h on this Mac at 8 workers), but cheap mechanics is not the blocker — Gate 2 is FAIL (§5's log),
so an 8000-region dataset trained against the current `extra_transition_weight`/`overhang_weight`
would still hit the same shattering mode, just at 20x the cost to discover it again. Regenerating
at full scale is worth doing once the objective is re-iterated (this document's Gate 2 finding is
open, per the operator's direction to stop and document rather than iterate now), not before.

---

## 5. Stop conditions — read before starting Stage 3

Stages 1 and 3 are **research changes with uncertain outcomes**, not repairs. The architecture is
proven capable (§2.1) and the objective's defect is measured (§1.2), so the direction is
well-founded — but whether the reweighted loss *finds* the overhang solution is empirical. Expect
to iterate on `extra_transition_weight` and `overhang_weight`; the first values will probably not
work.

**Stop and report rather than continuing, if:**

- Stage 1 cannot reach 8% multi-run columns without violating 1.2's quality bounds. That is a real
  finding about the generator, not a failure to try hard enough — report the frontier you measured.
- The Stage 4 tripwire stays at 0.0000 through step 1000 across **three** different
  `extra_transition_weight` / `overhang_weight` settings. Do not sweep a fourth. Report the three
  settings and their curves; the hypothesis in §1.2 is then incomplete and needs re-diagnosis, not
  more search.
- Gate 2 passes conditions 1–2 but fails 3–4 (floor-area / cavity-height retention). That is the
  shattering failure mode §7's fixtures exist to catch, and it means Stage 1 went too far. Report it;
  do not compensate by loosening the gate.

**Never:** loosen a Gate 2 condition, raise `--baseline-margin` tolerance, or narrow the diagnosis
sample to make a number pass. The entire cost of Phase 4.1 was a metric that looked fine.

---

## 6. Measurement log

Fill in as items land — command, result, date. Same convention as every prior plan document.

| Date | Item | Measurement | Result | Notes |
|---|---|---|---|---|
| 2026-08-05 | (context) | Ground-truth multi-run columns, `model/data` | 1.82% | §1.3 — below the 5.0-pt gate; a perfect model fails |
| 2026-08-05 | (context) | Heightfield-only BCE vs. observed plateau | 0.02623 vs [0.016, 0.05] | §1.1 — the run found its optimum |
| 2026-08-05 | (context) | `OccupancyLoss` mass on extra-transition cells | 0.414% | §1.2 — the entire overhang signal |
| 2026-08-05 | (context) | `Occupancy3DHead` overfit, 8 overhang-rich chunks | 100% acc, 42.6% vs 42.6% target | §2.1 — capacity is not the constraint |
| 2026-08-06 | 1.1 | Ground-truth multi-run % after retune (`CARVE_THRESHOLD` -0.35 -> -0.24, 400 regions regenerated seed 0) | **10.1363%** (vs. 0.0000% baseline, +10.14 pts) | Clears the 8.0% floor and the default 5.0-pt gate margin with room to spare |
| 2026-08-06 | 1.2 | Solid fraction / p90 cavity / max cavity / clipped-at-floor / walkable voids per 1000 cols | 0.9514 / 21 / 36 / 0 / 3.317 | All inside bounds ([0.93,0.98], >=3, <=48, =0, >=0.685) — no shattering, quality guard clears |
| 2026-08-06 | 2.1 | Parallel `generate_shards` speedup, 400 regions, seed 0, `region_layout_width` 64 | 8 workers: **362.2s** vs sequential **2398.0s** = **6.62x** | Byte-identical: all 400 shard SHA256 hashes match the sequential set (0 mismatched); P7 (parallel path, 4 workers) PASS against the sequentially-recorded fixture |
| 2026-08-06 | 3.1 | `OccupancyLoss` weight mass on overhang-bearing columns (`extra_transition_weight=100.0`) | **21.268%**, of which 16.514% is the extra-transition cells themselves | Up from 1.733% pre-fix; clears the >=20% floor |
| 2026-08-06 | 3.3 | `OverhangLoss` (per-cell BCE) term share of total loss at heightfield-only prediction, swept `overhang_weight` from 1.0 | 1.0 -> 7.37%, 2.0 -> 13.74%, **4.0 -> 24.15%**, 5.0 -> 28.47% | 4.0 shipped in every config — inside the [15%, 40%] band with margin both directions |
| 2026-08-06 | 5.1 | First nonzero tripwire reading (`config.mps.yaml`, 400 regions, `overhang_weight=4.0`/`extra_transition_weight=100.0`) | Step 150: **45.97%** (target on fixed batch 22.35%) | Capacity test green -> 500-step run showed real (if oscillating) nonzero structure, not a flat 0.0000% -> proceeded to full rehearsal run |
| 2026-08-06 | 5.1/5.2 | Full rehearsal run, 3000 steps, `training_runs/phase42_rehearsal/last.pt`. Gate 2 five-condition verdict (`diagnose_overhangs.py --checkpoint ... --ground-truth model/data --max-regions 32`) | multi-run 81.32% [MET, +81.32 pts]; walkable-void retention 0.00% [NOT MET]; mean floor area retention 0.00% [NOT MET]; p90 cavity height **1 block** [NOT MET, need >=3]; relief n/a. **VERDICT: FAIL** | **Stop condition hit (§5): "Gate 2 passes conditions 1-2 but fails 3-4 — the shattering failure mode."** Every one of the model's 3,331 "cavities" is exactly 1 block tall (mean=p50=p90=p95=p99=max=1), zero walkable voids — the model satisfies condition 1 by producing 1-block speckle in 81% of columns, not real caves. Tripwire log showed the same signature throughout training (0%/near-target/87.5%-speckle oscillation, never stably converging). Root cause is **not** Stage 1's generator — §1.2 already confirmed ground truth itself has healthy structure (p90 cavity 21, 0 speckle) — so per §5's own text this reads as `extra_transition_weight=100.0` over-rewarding *any* extra transition regardless of whether it forms real 3D structure, not the dataset being too aggressively carved. Not compensating by loosening Gate 2 or the diagnosis sample. Reported to operator for a decision on whether to iterate the weight (2 settings remain before the "three settings, stop" limit) or accept this as a documented stop.|

### 6.1 Independent verification of the Stage 5 failure (2026-08-07)

The Stage 5 row above attributes the shattering to `extra_transition_weight=100.0` "over-rewarding
any extra transition regardless of whether it forms real 3D structure." **That attribution is not
supported by measurement, and acting on it would have wasted the two remaining sweep slots.** Three
checks, all on `model/data` (the retuned dataset) and `training_runs/phase42_rehearsal/last.pt`:

| Check | Result | Reading |
|---|---|---|
| Loss of a heightfield-only vs. real-cave vs. matched-rate-speckle prediction, under the shipped objective (`extra_transition_weight=100.0`, `overhang_weight=4.0`) | heightfield **1.3903**, real caves **0.0005**, speckle **1.5415** | The objective **strongly prefers real caves** — by ~3 orders of magnitude — and rates speckle *worse than doing nothing*. The loss is not what produced speckle. |
| Per-term gradient norms on cave-bearing chunks, at three confidence levels | occupancy 0.0066–0.0077, `4.0 ×` overhang 0.0007–0.0008 (ratio 0.1×); clamp dead-zone cells: **0** | No gradient pathology, no dead zone, overhang term is a minor contributor. Not a scale problem. |
| Sub-surface (below topmost solid) cave-placement agreement, 12 carved regions, each fed its **own** world seed and chunk coords, 6.75M cells | model sub-surface air 1.36% vs truth 1.95%; IoU **0.0147**; Pearson **r = 0.0133**; lift over chance **1.81×**; model cavity heights mean 1.00 / max **1**, truth mean 8.20 / p90 16 / max 28 | **The model never learned cave placement at all.** It reproduces roughly the right *amount* of sub-surface air at essentially random locations. |

The failure is therefore **not** a loss-design or loss-weight problem. It is that the model did not
learn the coordinate→structure mapping, and so fell back on predicting the marginal sub-surface air
rate; thresholding a near-uniform ~0.5 field at 0.5 is what renders as 1-block speckle. The training
log corroborates this directly: the Stage 4 tripwire, evaluated on a **fixed** batch, reads 0.0000%
→ 56.82% → 23.47% → 87.50% → 0.0000% → 40.12% → 87.50% within a few hundred steps, with occupancy
loss oscillating in [0.21, 0.59] and not descending. A model that had learned placement cannot swing
a fixed batch from 0% to 87.5% and back; a model sitting on a global near-threshold field can.

**Consequence for the next session:** sweeping `extra_transition_weight` (or `overhang_weight`)
changes the *rate* of speckle, not its randomness, and cannot move `r = 0.013`. The open question is
upstream of every knob Stage 3 touches — whether this architecture, conditioning, and step budget
can learn a block-resolution 3D field over the 2048×2048 canonical domain at all. Candidate lines,
in cost order: (a) confirm learnability by overfitting **one** region to convergence with the real
conditioning path (not `test_occupancy_capacity.py`'s random frozen features, which prove only that
the head can *represent* structure, not that the network can *learn* placement); (b) if that
succeeds, the deficit is optimization budget/schedule, not design; (c) if it fails, the coordinate
encoder's frequency content and the head's 16→64 y-expansion are the suspects.

`test_occupancy_capacity.py` should be read as narrower than §2.1 credited it: it establishes
representational capacity only.

---

## 7. Ready-to-rent checklist

Renting is justified when **all** of these hold. Not four of five.

- [ ] Gate 2 prints **VERDICT: PASS** on a rehearsal-scale run, reproducibly (5.2). **NOT MET** —
      2026-08-06 full rehearsal run (3000 steps): VERDICT FAIL, shattering mode (§5 stop condition
      hit; see §6's log). Documented and stopped per operator direction, not iterated further.
      **§6.1 (2026-08-07) supersedes that row's root-cause attribution:** measured, the objective
      prefers real caves over speckle by ~3 orders of magnitude and the gradients are clean — the
      model simply never learned cave placement (sub-surface Pearson r = 0.013, i.e. chance). This
      is an open research question, not a weight left untuned, and it is the single thing standing
      between this project and a justified rental.
- [x] `preflight` on `config.cuda.yaml` is clean except the two known-correct Darwin failures (6.1).
      **Partially — three FAILs, not two, but one root cause.** 2026-08-07:
      `preflight --config config.cuda.yaml --budget-hours 40 --keep-going` gives P2, P9, P11 FAIL —
      P9/P11 fail for the IDENTICAL reason as P2 (both actually launch `train.py` as a subprocess
      against the unmodified `hardware.device: cuda`, which correctly refuses to run without CUDA on
      this Mac). Not three independent problems. P5 correctly SKIPs as expected. Real MPS-side
      throughput measured separately via `config.mps.yaml`: 0.79 steps/s, P10 projects 60.8h at
      8000 regions/15 epochs — over the 40h budget, but per §4's own instruction this is an MPS
      number, not the CUDA number, and is recorded rather than gated on.
- [x] **P4 never FAILs on any config**, at both 400 and 8000 target scale (0.1). MET — Stage 0.
- [x] P13 PASS — the dataset can actually clear its own gate (1.3). MET.
- [x] `pytest` and `pytest -O` both green (5.3). MET — 101 passed, 1 skipped, both ways.
- [x] The capacity test (3.2) is green — the architecture has not regressed. MET.
- [ ] Everything committed (6.3). Not committed — awaiting explicit go-ahead per this document's own
      instruction ("Do not commit anything without asking").

The readiness work from `docs/remote-training-readiness-plan.md` is already done and verified
(2026-08-05: P4 155.68 GB → 1.97 GB, checkpoints 123 MB → 33 MB, mid-epoch resume fixed, 98 tests
green). Once this document's boxes are ticked, the rented machine is a purchasing decision.

---

## 8. Handoff notes for the implementing agent

- Work the stages **in order**. Stage 1 before Stage 3 is not stylistic: without it there is no
  success signal. Stage 4 before Stage 5 is not stylistic: without it each attempt costs 35 minutes
  to learn one bit.
- **Report per item, with the check's actual output.** "Stage 3 implemented" without
  `loss_mass.py`'s numbers is not done.
- Re-run `preflight` after every stage.
- §5's stop conditions are real instructions. Reporting "three settings, all zero, here are the
  curves" is a **successful** outcome of this plan. Quietly sweeping a fourth is not.
- Do not commit without asking.
