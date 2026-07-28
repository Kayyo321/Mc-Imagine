"""
Packaging script that bundles ONNX graph(s), tokenizer, and manifest into a .mcim ZIP file.

Manifest shape follows docs/model-spec.md (format_version 0.5.0) — keep this in sync with that
document, not the other way around. As of 0.4.0 a model may ship up to three ONNX graphs:
model.onnx (per-chunk, always required), macro.onnx (per-macro-region, required iff
capabilities.requires_macro_field), and detail.onnx (per-detail-pass, optional even when
capabilities.detail_passes > 0). 0.5.0 adds `capabilities.biome_palette` and pins the axis order
documented in "Output Tensors" (this script doesn't encode axis order itself — that's a property of
what export_onnx.py actually produces — but the manifest's capabilities block is what the mod uses
to interpret the ids inside those tensors).
"""

import argparse
import json
import os
import zipfile
from typing import Any, Dict, List


FORMAT_VERSION = "0.5.0"


def _chunk_output_names(structure_support: str) -> List[str]:
    names = ["heightmap", "block_volume", "biome_grid"]
    if structure_support == "basic":
        names.append("structure_markers")
    elif structure_support == "intricate":
        names += ["structure_graph_nodes", "structure_graph_edges", "structure_origin"]
    return names


def _chunk_input_names(intensity: str, structure_support: str, requires_macro_field: bool) -> List[str]:
    names = ["prompt_tokens", "chunk_x", "chunk_z", "seed"]
    if intensity in ("medium", "high"):
        names.append("intensity_scale")
    if structure_support == "intricate":
        names.append("structure_request")
    if requires_macro_field:
        names += ["macro_local_height", "flavor_zone_ids", "flavor_zone_weights"]
    return names


def _macro_io() -> Dict[str, List[str]]:
    return {
        "input_names": ["prompt_tokens", "seed", "region_x", "region_z"],
        "output_names": ["region_heightfield", "flavor_zone_ids", "flavor_zone_weights", "structure_candidate"],
    }


def _detail_io() -> Dict[str, List[str]]:
    return {
        "input_names": ["prompt_tokens", "chunk_x", "chunk_z", "seed", "pass_index", "block_volume"],
        "output_names": ["detail_overrides"],
    }


def build_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    """Builds a manifest.json dict conforming to docs/model-spec.md (format_version 0.5.0)."""
    capabilities: Dict[str, Any] = {
        "intensity": args.intensity,
        "terrain": True,
        "biomes": args.biomes,
        "caves": args.caves,
        "structure_support": args.structure_support,
        "prompt_tags": [t.strip() for t in args.prompt_tags.split(",") if t.strip()],
        "block_palette": [b.strip() for b in args.block_palette.split(",") if b.strip()],
        "max_prompt_tokens": args.max_prompt_tokens,
        "requires_macro_field": args.requires_macro_field,
        "detail_passes": args.detail_passes,
    }
    if args.biome_palette:
        capabilities["biome_palette"] = [b.strip() for b in args.biome_palette.split(",") if b.strip()]
    if args.structure_support == "intricate":
        capabilities["max_rooms"] = args.max_rooms
        capabilities["template_library_version"] = args.template_library_version
        capabilities["structure_spacing_regions"] = args.structure_spacing_regions
    if args.flavor_zones:
        # Expected format: "1:blighted_moor:minecraft:coarse_dirt|minecraft:cobweb;2:shadowbark_forest:minecraft:dark_oak_log"
        zones = []
        for entry in args.flavor_zones.split(";"):
            entry = entry.strip()
            if not entry:
                continue
            zone_id, name, blocks = entry.split(":", 2)
            zones.append({
                "id": int(zone_id),
                "name": name,
                "block_bias": [b.strip() for b in blocks.split("|") if b.strip()],
            })
        capabilities["flavor_zones"] = zones

    io: Dict[str, Any] = {
        "chunk": {
            "input_names": _chunk_input_names(args.intensity, args.structure_support, args.requires_macro_field),
            "output_names": _chunk_output_names(args.structure_support),
        }
    }
    if args.requires_macro_field:
        io["macro"] = _macro_io()
    if args.detail_onnx:
        io["detail"] = _detail_io()

    return {
        "format_version": FORMAT_VERSION,
        "model": {
            "id": args.model_id,
            "name": args.name,
            "version": args.version,
            "author": args.author,
            "description": args.description,
            "license": args.license,
        },
        "capabilities": capabilities,
        "requirements": {
            "min_ram_mb": args.min_ram_mb,
            "recommended_vram_mb": args.recommended_vram_mb,
            "onnx_opset": args.opset,
        },
        "io": io,
    }


def main() -> None:
    """
    Main packaging function.
    Creates manifest.json and packages it with the model graph(s) into a .mcim file.
    """
    parser = argparse.ArgumentParser(description="Package a model into .mcim (see docs/model-spec.md)")
    parser.add_argument("--onnx_model", type=str, required=True, help="Path to the per-chunk model.onnx")
    parser.add_argument("--macro_onnx", type=str, default=None, help="Path to macro.onnx (required iff --requires_macro_field)")
    parser.add_argument("--detail_onnx", type=str, default=None, help="Path to detail.onnx (optional; omit to use the mod's built-in procedural decorator)")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer.json")
    parser.add_argument("--output", type=str, default="model.mcim", help="Output .mcim file path")

    parser.add_argument("--model_id", type=str, required=True, help="e.g. imaginator-low_intensity-no_structures")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--version", type=str, default="1.0.0")
    parser.add_argument("--author", type=str, default="Unknown")
    parser.add_argument("--description", type=str, default="A Minecraft Imagine model.")
    parser.add_argument("--license", type=str, default="CC-BY-4.0")

    parser.add_argument("--intensity", choices=["low", "medium", "high"], default="low", help="Attention to detail — see docs/model-spec.md")
    parser.add_argument("--structure_support", choices=["none", "basic", "intricate"], default="none")
    parser.add_argument("--biomes", action="store_true")
    parser.add_argument("--caves", action="store_true")
    parser.add_argument("--prompt_tags", type=str, default="terrain")
    parser.add_argument("--block_palette", type=str, default="minecraft:stone,minecraft:dirt,minecraft:grass_block")
    parser.add_argument("--biome_palette", type=str, default=None, help="Comma-separated namespaced biome ids biome_grid indexes into (0.5.0+)")
    parser.add_argument("--max_prompt_tokens", type=int, default=128)
    parser.add_argument("--max_rooms", type=int, default=12, help="Only used when structure_support=intricate")
    parser.add_argument("--template_library_version", type=str, default="0.1.0", help="Only used when structure_support=intricate")
    parser.add_argument("--structure_spacing_regions", type=int, default=4, help="Only used when structure_support=intricate")
    parser.add_argument("--requires_macro_field", action="store_true", help="Ship macro.onnx and require cross-chunk macro context")
    parser.add_argument("--detail_passes", type=int, default=0, help="0/1/2 conventionally for low/medium/high intensity")
    parser.add_argument("--flavor_zones", type=str, default=None, help='e.g. "1:blighted_moor:minecraft:coarse_dirt|minecraft:cobweb;2:shadowbark_forest:minecraft:dark_oak_log"')

    parser.add_argument("--min_ram_mb", type=int, default=2048)
    parser.add_argument("--recommended_vram_mb", type=int, default=1024)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    if args.requires_macro_field and not args.macro_onnx:
        parser.error("--macro_onnx is required when --requires_macro_field is set")

    print(f"Packaging {args.onnx_model} into {args.output}")

    manifest = build_manifest(args)
    manifest_path = os.path.join(os.path.dirname(args.output) or ".", "_manifest_tmp.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    try:
        with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(args.onnx_model, arcname="model.onnx")
            if args.macro_onnx:
                zf.write(args.macro_onnx, arcname="macro.onnx")
            if args.detail_onnx:
                zf.write(args.detail_onnx, arcname="detail.onnx")
            zf.write(args.tokenizer, arcname="tokenizer.json")
            zf.write(manifest_path, arcname="manifest.json")
    finally:
        os.remove(manifest_path)

    print(f"Packaging complete: {args.output}")


if __name__ == "__main__":
    main()
