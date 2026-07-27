"""
ONNX export script.

Exports a trained PyTorch model to ONNX format. A single .mcim can bundle up to three separate
ONNX graphs (see docs/model-spec.md "Two-Model File Structure") — this script exports one graph
per invocation, selected via --graph, since each has a different checkpoint, input signature, and
output signature:
  - chunk  -> model.onnx   (ImagineNet, model/imagine_net.py)  — always required
  - macro  -> macro.onnx   (MacroFieldNet, model/macro_net.py) — only if requires_macro_field
  - detail -> detail.onnx  (optional ML decorator)             — only if using ML-driven detail passes
"""

import argparse
import torch
import onnx
import onnxruntime

# TODO: Import once implemented
# from mc_imagine_model.model.imagine_net import ImagineNet
# from mc_imagine_model.model.macro_net import MacroFieldNet

def main() -> None:
    """
    Main export function. Loads a checkpoint and exports the selected graph to ONNX.
    """
    parser = argparse.ArgumentParser(description="Export a model graph to ONNX (see docs/model-spec.md)")
    parser.add_argument("--graph", choices=["chunk", "macro", "detail"], default="chunk", help="Which of the up-to-three graphs to export")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PyTorch checkpoint")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output ONNX file path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    print(f"Exporting checkpoint {args.checkpoint} ({args.graph} graph) to {args.output} with opset {args.opset}")

    if args.graph == "chunk":
        # TODO: Load ImagineNet from checkpoint, model.eval()
        # TODO: Create dummy inputs (prompt_tokens, chunk_x, chunk_z, seed, and — depending on the
        # trained model's declared capabilities — intensity_scale, structure_request,
        # macro_local_height, flavor_zone_ids, flavor_zone_weights)
        #
        # input_names/output_names must match the model's declared capabilities (see
        # docs/model-spec.md "Input Tensors: model.onnx" / "Output Tensors: model.onnx"):
        #   - intensity in ("medium", "high") adds "intensity_scale" ("low" has no detail knob)
        #   - structure_support == "intricate" adds "structure_request" to inputs, and
        #     ["structure_graph_nodes", "structure_graph_edges", "structure_origin"] to outputs
        #     (instead of "structure_markers", which is structure_support == "basic" only)
        #   - requires_macro_field adds "macro_local_height", "flavor_zone_ids", "flavor_zone_weights"
        # torch.onnx.export(
        #     model,
        #     (dummy_prompt, dummy_x, dummy_z, dummy_seed),
        #     args.output,
        #     opset_version=args.opset,
        #     input_names=["prompt_tokens", "chunk_x", "chunk_z", "seed"],
        #     output_names=["heightmap", "block_volume", "biome_grid"],
        # )
        pass
    elif args.graph == "macro":
        # TODO: Load MacroFieldNet from checkpoint, model.eval()
        # TODO: Create dummy inputs (prompt_tokens, seed, region_x, region_z)
        # torch.onnx.export(
        #     model,
        #     (dummy_prompt, dummy_seed, dummy_region_x, dummy_region_z),
        #     args.output,
        #     opset_version=args.opset,
        #     input_names=["prompt_tokens", "seed", "region_x", "region_z"],
        #     output_names=["region_heightfield", "flavor_zone_ids", "flavor_zone_weights", "structure_candidate"],
        # )
        pass
    elif args.graph == "detail":
        # TODO: Load the detail decorator model from checkpoint, model.eval()
        # TODO: Create dummy inputs (prompt_tokens, chunk_x, chunk_z, seed, pass_index, block_volume)
        # torch.onnx.export(
        #     model,
        #     (dummy_prompt, dummy_x, dummy_z, dummy_seed, dummy_pass_index, dummy_block_volume),
        #     args.output,
        #     opset_version=args.opset,
        #     input_names=["prompt_tokens", "chunk_x", "chunk_z", "seed", "pass_index", "block_volume"],
        #     output_names=["detail_overrides"],
        # )
        pass

    # TODO: Verify exported model using ONNX Runtime
    # onnx_model = onnx.load(args.output)
    # onnx.checker.check_model(onnx_model)
    # print("ONNX model verified successfully.")

if __name__ == "__main__":
    main()
