#!/usr/bin/env bash
set -euo pipefail

# Sweep relief_weight in {5, 10, 25, 50, 100} using train.py and config.rehearsal.yaml
# with --regions 400 --max-steps 400.

CONFIG="model/src/mc_imagine_model/training/config.rehearsal.yaml"

# Ensure training data exists for 400 regions
echo "Ensuring rehearsal dataset with 400 regions is present..."
PYTHONPATH=model/src python3 -m mc_imagine_model.scripts.generate_data --regions 400 --out model/data --seed 0

weights=(5 10 25 50 100)

for w in "${weights[@]}"; do
    echo "========================================================"
    echo "Running sweep for relief_weight=${w}"
    echo "========================================================"
    PYTHONPATH=model/src python3 -m mc_imagine_model.training.train \
        --config "${CONFIG}" \
        --max-steps 400 \
        --relief-weight "${w}"
done

echo "Sweep complete."
