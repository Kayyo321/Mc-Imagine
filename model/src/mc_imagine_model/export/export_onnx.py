"""
ONNX export script.

Exports a trained PyTorch model to ONNX format.
"""

import argparse
import torch
import onnx
import onnxruntime

# TODO: Import ImagineNet once implemented
# from mc_imagine_model.model.imagine_net import ImagineNet

def main() -> None:
    """
    Main export function. Loads a checkpoint and exports to ONNX.
    """
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PyTorch checkpoint")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output ONNX file path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    print(f"Exporting checkpoint {args.checkpoint} to {args.output} with opset {args.opset}")

    # TODO: Load model from checkpoint
    # model = ImagineNet(...)
    # model.load_state_dict(torch.load(args.checkpoint))
    # model.eval()

    # TODO: Create dummy inputs
    # dummy_prompt = torch.zeros((1, 64), dtype=torch.long)
    # dummy_x = torch.zeros((1,), dtype=torch.long)
    # dummy_z = torch.zeros((1,), dtype=torch.long)
    # dummy_seed = torch.zeros((1,), dtype=torch.long)
    
    # TODO: Export to ONNX
    # torch.onnx.export(
    #     model,
    #     (dummy_prompt, dummy_x, dummy_z, dummy_seed),
    #     args.output,
    #     opset_version=args.opset,
    #     input_names=["prompt_tokens", "chunk_x", "chunk_z", "seed"],
    #     output_names=["heightmap", "block_volume", "biome_grid", "structure_markers"],
    # )
    
    # TODO: Verify exported model using ONNX Runtime
    # onnx_model = onnx.load(args.output)
    # onnx.checker.check_model(onnx_model)
    # print("ONNX model verified successfully.")

if __name__ == "__main__":
    main()
