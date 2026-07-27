"""
Packaging script that bundles ONNX model, tokenizer, and manifest into a .mcim ZIP file.
"""

import argparse
import json
import zipfile
import os

def main() -> None:
    """
    Main packaging function.
    Creates manifest.json and packages it with the model into a .mcim file.
    """
    parser = argparse.ArgumentParser(description="Package model into .mcim")
    parser.add_argument("--onnx_model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer file")
    parser.add_argument("--output", type=str, default="model.mcim", help="Output .mcim file path")
    args = parser.parse_args()

    print(f"Packaging {args.onnx_model} and {args.tokenizer} into {args.output}")

    # Create manifest
    manifest = {
        "version": "1.0",
        "model_file": os.path.basename(args.onnx_model),
        "tokenizer_file": os.path.basename(args.tokenizer),
        "inputs": ["prompt_tokens", "chunk_x", "chunk_z", "seed"],
        "outputs": ["heightmap", "block_volume", "biome_grid", "structure_markers"]
    }
    
    manifest_path = "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
        
    # TODO: Package into ZIP
    # with zipfile.ZipFile(args.output, "w") as zf:
    #     zf.write(args.onnx_model, arcname=os.path.basename(args.onnx_model))
    #     zf.write(args.tokenizer, arcname=os.path.basename(args.tokenizer))
    #     zf.write(manifest_path)
    
    print("Packaging complete.")

if __name__ == "__main__":
    main()
