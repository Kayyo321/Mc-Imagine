"""
One command from a trained checkpoint to a loadable `.mcim`: ONNX export -> ORT verification ->
manifest + zip.

Day 1 did this by hand as two invocations, the second of which carried the entire block and biome
palette as literal comma-separated strings typed at the shell. Retyping a 13-item list is how the
manifest drifts away from what the model actually predicts (and the ids in `block_volume`/
`biome_grid` mean nothing without a correct palette — a wrong one is silently wrong *in world*, not
at load time). Here the palettes and every other capability default come from `spec_constants` and
the checkpoint's own config, so there is nothing to retype.

This deliberately calls `export/export_onnx.py` and `export/package_mcim.py` rather than
reimplementing either — the export graph and the manifest schema each have exactly one owner.

Usage:
    python -m mc_imagine_model.scripts.build_mcim \
        --checkpoint model/training_runs/cuda_run1/best.pt \
        --out imaginator-low.mcim
"""

import argparse
import os
import sys
import time
from argparse import Namespace

from mc_imagine_model.spec_constants import BIOME_PALETTE, BLOCK_PALETTE

_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # …/mc_imagine_model
MODEL_ROOT = os.path.dirname(os.path.dirname(_PACKAGE_DIR))                 # …/model
DEFAULT_TOKENIZER = os.path.join(MODEL_ROOT, "checkpoints", "all-MiniLM-L6-v2", "tokenizer.json")

DEFAULT_MODEL_ID = "imaginator-low_intensity-no_structures"
DEFAULT_PROMPT_TAGS = "terrain,biomes"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a .mcim from a training checkpoint (export ONNX, then package)."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a training checkpoint (.pt)")
    parser.add_argument("--out", default="model.mcim", help="Output .mcim path")
    parser.add_argument("--onnx", default=None,
                        help="Intermediate model.onnx path (default: alongside --out)")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                        help="tokenizer.json to ship inside the .mcim "
                             "(default: the vendored MiniLM one — run scripts.bootstrap if missing)")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip the ONNX Runtime round-trip check (not recommended)")
    parser.add_argument("--keep-onnx", action="store_true",
                        help="Keep the intermediate .onnx instead of deleting it")
    parser.add_argument("--deterministic-carve", action="store_true",
                        help="Gate 1 fixture mode: export a fixed hand-carved overhang instead of the trained occupancy head")

    meta = parser.add_argument_group("manifest metadata")
    meta.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    meta.add_argument("--name", default="Mc-Imagine Low Intensity")
    meta.add_argument("--version", required=True, help="Version string for manifest (e.g. 1.1.0)")
    meta.add_argument("--author", default="Mc-Imagine")
    meta.add_argument("--description", default="Prompt-conditioned terrain and biome generator.")
    meta.add_argument("--license", default="CC-BY-4.0")
    meta.add_argument("--intensity", choices=["low", "medium", "high"], default="low")
    meta.add_argument("--structure_support", choices=["none", "basic", "intricate"], default="none")
    meta.add_argument("--prompt_tags", default=DEFAULT_PROMPT_TAGS)
    meta.add_argument("--max_prompt_tokens", type=int, default=128)
    meta.add_argument("--min_ram_mb", type=int, default=2048)
    meta.add_argument("--recommended_vram_mb", type=int, default=1024)
    args = parser.parse_args()

    # Imported here (not at module import time) so `--help` works without torch/onnx loaded.
    from mc_imagine_model.export.export_onnx import export_chunk_graph, verify_chunk_graph
    from mc_imagine_model.export.package_mcim import write_mcim

    if args.version != "1.1.2":
        print(f"ERROR: --version 1.1.2 is required for Phase 4 Gate 1 builds (got '{args.version}')", file=sys.stderr)
        return 1

    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.tokenizer):
        print(
            f"ERROR: tokenizer.json not found at {args.tokenizer}\n"
            "Run `python -m mc_imagine_model.scripts.bootstrap` to vendor the MiniLM checkpoint.",
            file=sys.stderr,
        )
        return 1

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = args.onnx or os.path.splitext(os.path.abspath(args.out))[0] + ".onnx"

    t0 = time.time()
    print(f"[1/3] Exporting {args.checkpoint} -> {onnx_path} (opset {args.opset})")
    export_chunk_graph(args.checkpoint, onnx_path, args.opset, deterministic_carve=args.deterministic_carve)

    if args.skip_verify:
        print("[2/3] Skipping ORT verification (--skip-verify)")
    else:
        print("[2/3] Verifying the exported graph with ONNX Runtime")
        verify_chunk_graph(onnx_path, args.checkpoint)

    print(f"[3/3] Packaging -> {args.out}")
    print(f"      block_palette: {len(BLOCK_PALETTE)} entries (from spec_constants)")
    print(f"      biome_palette: {len(BIOME_PALETTE)} entries (from spec_constants)")

    # Reuse package_mcim's manifest builder and zip layout verbatim by handing it the same Namespace
    # its own argparse would have produced. `block_palette`/`biome_palette` stay None so it takes its
    # spec_constants defaults — the whole point of this script.
    pkg_args = Namespace(
        onnx_model=onnx_path,
        macro_onnx=None,
        detail_onnx=None,
        tokenizer=args.tokenizer,
        output=args.out,
        model_id=args.model_id,
        name=args.name,
        version=args.version,
        author=args.author,
        description=args.description,
        license=args.license,
        intensity=args.intensity,
        structure_support=args.structure_support,
        biomes=True,
        caves=True,  # format version 0.6.0 volumetric chunk generation with 3D caves
        prompt_tags=args.prompt_tags,
        block_palette=None,
        biome_palette=None,
        max_prompt_tokens=args.max_prompt_tokens,
        max_rooms=12,
        template_library_version="0.1.0",
        structure_spacing_regions=4,
        requires_macro_field=False,  # docs/phase2-plan.md Phase 3 is deferred; this tier never needs it
        detail_passes=0,
        flavor_zones=None,
        min_ram_mb=args.min_ram_mb,
        recommended_vram_mb=args.recommended_vram_mb,
        opset=args.opset,
    )

    write_mcim(pkg_args)

    if not args.keep_onnx and not args.onnx:
        os.remove(onnx_path)

    size = os.path.getsize(args.out)
    print(f"Done in {time.time() - t0:.1f}s — {args.out} ({size / 1e6:.1f} MB)")
    print("Install it by dropping the .mcim into the mod's models directory.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
