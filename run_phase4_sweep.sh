#!/usr/bin/env bash
set -euo pipefail

# Phase 4.1 §1.2 re-sweep of occupancy_weight/consistency_weight at the corrected
# overhang_weight, plus a standalone overhang_weight sweep — docs/phase4.1-plan.md §1.2,
# §2's Step C. The original Gate 2 sweep (docs/CHANGELOG.md) held overhang_weight at its
# broken shipped default (1.0) while sweeping the other two weights, confounding the whole
# result; every run below passes --overhang-weight explicitly on the CLI so it never
# silently inherits config.rehearsal.yaml's default.
#
# Each run targets exactly 400 total steps (docs/phase4-plan.md §7's Gate 2 step count),
# reached via train.py's --max-steps early-exit path (fixed in Phase 4.1 §1.1) rather than
# the batch_size=92 workaround the original sweep used — num_epochs is bumped to 3 per-run
# so the 400-step cutoff lands mid-epoch and exercises that exact path on every run.

CONFIG="model/src/mc_imagine_model/training/config.rehearsal.yaml"
CORRECTED_OVERHANG_WEIGHT=0.005
MAX_STEPS=400
RUN_ROOT="training_runs/phase4_sweep"  # relative to model/, matching config checkpoint_dir convention

echo "Ensuring rehearsal dataset with 400 regions is present..."
PYTHONPATH=model/src python3 -m mc_imagine_model.scripts.generate_data --regions 400 --out model/data --seed 0

run_one() {
    local name="$1"
    local occupancy_weight="$2"
    local overhang_weight="$3"
    local consistency_weight="$4"

    local run_dir="${RUN_ROOT}/${name}"
    mkdir -p "model/${run_dir}"
    local run_config="model/${run_dir}/config.yaml"

    # checkpoint_dir and num_epochs have no CLI override in train.py, so they're the two
    # fields that need a per-run config copy; every loss weight below is overridden
    # explicitly on the CLI instead, so none of them silently falls back to this config's
    # defaults.
    sed \
        -e "s#checkpoint_dir: \"training_runs/rehearsal\"#checkpoint_dir: \"${run_dir}\"#" \
        -e "s#num_epochs: 1#num_epochs: 3#" \
        "${CONFIG}" > "${run_config}"

    echo "========================================================"
    echo "Run ${name}: occupancy_weight=${occupancy_weight} overhang_weight=${overhang_weight} consistency_weight=${consistency_weight}"
    echo "========================================================"
    PYTHONPATH=model/src python3 -m mc_imagine_model.training.train \
        --config "${run_config}" \
        --max-steps "${MAX_STEPS}" \
        --occupancy-weight "${occupancy_weight}" \
        --overhang-weight "${overhang_weight}" \
        --consistency-weight "${consistency_weight}"
}

# --- Section 1: occupancy_weight sweep at the corrected overhang_weight, consistency_weight
#     held at the config default (0.25). Same occupancy_weight grid as the original Gate 2
#     sweep, for comparability. ---
for w in 0.25 0.5 1.0 2.0; do
    run_one "occupancy_${w}" "${w}" "${CORRECTED_OVERHANG_WEIGHT}" 0.25
done

# --- Section 2: consistency_weight sweep at the corrected overhang_weight, occupancy_weight
#     held at the config default (1.0). Same consistency_weight grid as the original sweep. ---
for w in 0.05 0.1 0.25 0.5; do
    run_one "consistency_${w}" 1.0 "${CORRECTED_OVERHANG_WEIGHT}" "${w}"
done

# --- Section 3: overhang_weight swept on its own, other weights held at config defaults
#     (occupancy_weight 1.0, consistency_weight 0.25) — docs/phase4.1-plan.md §2.2's range. ---
for w in 0.001 0.003 0.005 0.01 0.03; do
    run_one "overhang_${w}" 1.0 "${w}" 0.25
done

echo "Phase 4 sweep complete. Evaluate each run with, e.g.:"
echo "  PYTHONPATH=model/src python3 -m mc_imagine_model.scripts.diagnose_overhangs \\"
echo "    --ground-truth model/data --checkpoint model/${RUN_ROOT}/<name>/best.pt"
