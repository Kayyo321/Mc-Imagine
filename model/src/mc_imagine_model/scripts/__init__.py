"""Operator-facing CLIs that make a fresh clone of this repo trainable on another machine.

`model/data/` (the training shards) and `model/checkpoints/` (the vendored MiniLM encoder) are
gitignored — correctly, they are ~1.6 GB and ~90 MB respectively — which means a clone has neither.
Everything in this package exists so that gap is closed by a documented command instead of by
undocumented manual steps (docs/phase2-plan.md §0 "handoff blockers", Phase 4):

    python -m mc_imagine_model.scripts.bootstrap        # vendors the MiniLM checkpoint
    python -m mc_imagine_model.scripts.generate_data    # regenerates the training shards
    python -m mc_imagine_model.scripts.build_mcim       # checkpoint -> ONNX -> .mcim, one command
"""
