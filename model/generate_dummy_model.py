#!/usr/bin/env python3
"""
generate_dummy_model.py

Generates a dummy PyTorch model for Mc-Imagine, exports it to ONNX,
creates a valid manifest.json, packages them into dummy-test-v1.mcim,
and places it in mod/fabric/run/mcimagine/models/.
"""

import os
import json
import zipfile
import tempfile
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

class DummyTerrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Dummy linear layer to maintain trainable parameter
        self.linear = nn.Linear(1, 1)

    def forward(self, prompt_tokens, chunk_x, chunk_z, seed):
        """
        Takes chunk_x, chunk_z, seed, prompt_tokens and computes heightmap tensor (16x16 float array).
        Heightmap Y-level depends on chunk_x and chunk_z coordinates.
        """
        cx = chunk_x.to(torch.float32)
        cz = chunk_z.to(torch.float32)

        # 16x16 local chunk grid offsets
        grid_x = torch.arange(16, dtype=torch.float32).reshape(16, 1)
        grid_z = torch.arange(16, dtype=torch.float32).reshape(1, 16)

        # Calculate Y-level based on chunk X coordinate (and chunk Z)
        base_y = 64.0 + cx * 2.0 + cz * 0.5
        heightmap = base_y + (grid_x + grid_z) * 0.1 + self.linear(cx) * 0.0
        return heightmap

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    # Destination directory: mod/fabric/run/mcimagine/models/
    target_dir = os.path.join(project_root, "mod", "fabric", "run", "mcimagine", "models")
    os.makedirs(target_dir, exist_ok=True)
    
    output_mcim_path = os.path.join(target_dir, "dummy-test-v1.mcim")
    temp_dir = os.path.join(script_dir, "_temp_export")
    os.makedirs(temp_dir, exist_ok=True)

    onnx_path = os.path.join(temp_dir, "model.onnx")
    manifest_path = os.path.join(temp_dir, "manifest.json")
    tokenizer_path = os.path.join(temp_dir, "tokenizer.json")

    print(f"Exporting PyTorch model to ONNX: {onnx_path}")
    model = DummyTerrainModel()
    model.eval()

    dummy_prompt_tokens = torch.zeros((1, 128), dtype=torch.int32)
    dummy_chunk_x = torch.tensor([0], dtype=torch.int32)
    dummy_chunk_z = torch.tensor([0], dtype=torch.int32)
    dummy_seed = torch.tensor([12345], dtype=torch.int64)

    input_names = ["prompt_tokens", "chunk_x", "chunk_z", "seed"]
    output_names = ["heightmap"]

    torch.onnx.export(
        model,
        (dummy_prompt_tokens, dummy_chunk_x, dummy_chunk_z, dummy_seed),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        dynamic_axes={
            "prompt_tokens": {0: "batch_size"},
        }
    )

    print("Re-saving ONNX model as self-contained file (save_as_external_data=False)...")
    onnx_model = onnx.load(onnx_path)
    onnx.save_model(onnx_model, onnx_path, save_as_external_data=False)

    external_data_file = onnx_path + ".data"
    if os.path.exists(external_data_file):
        os.remove(external_data_file)

    print("Testing loading model.onnx directly with ONNX Runtime...")
    test_session = ort.InferenceSession(onnx_path)
    print("Successfully loaded model.onnx directly with ONNX Runtime.")

    print(f"Creating manifest.json conforming to docs/model-spec.md")
    manifest_data = {
        "format_version": "0.1.0",
        "id": "dummy-test-v1",
        "name": "Dummy Test Model",
        "version": "1.0.0",
        "author": "Mc-Imagine Team",
        "description": "Baseline dummy terrain generator calculating heightmap based on chunk X coordinate.",
        "onnx_filename": "model.onnx",
        "model": {
            "id": "dummy-test-v1",
            "name": "Dummy Test Model",
            "version": "1.0.0",
            "author": "Mc-Imagine Team",
            "description": "Baseline dummy terrain generator calculating heightmap based on chunk X coordinate.",
            "license": "MIT"
        },
        "capabilities": {
            "terrain": True,
            "biomes": False,
            "caves": False,
            "structures": False,
            "block_palette": ["minecraft:stone", "minecraft:dirt", "minecraft:grass_block"],
            "max_prompt_tokens": 128
        },
        "requirements": {
            "min_ram_mb": 2048,
            "recommended_vram_mb": 1024,
            "onnx_opset": 17
        },
        "io": {
            "input_names": input_names,
            "output_names": output_names
        },
        "input_spec": {
            "prompt_tokens": {"shape": [1, 128], "dtype": "int32"},
            "chunk_x": {"shape": [1], "dtype": "int32"},
            "chunk_z": {"shape": [1], "dtype": "int32"},
            "seed": {"shape": [1], "dtype": "int64"}
        },
        "output_spec": {
            "heightmap": {"shape": [16, 16], "dtype": "float32"}
        }
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # Simple tokenizer placeholder if needed
    tokenizer_data = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "model": {
            "type": "WordPiece",
            "vocab": {"[PAD]": 0, "[UNK]": 1}
        }
    }
    with open(tokenizer_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_data, f, indent=2)

    print(f"Packaging model into {output_mcim_path}")
    with zipfile.ZipFile(output_mcim_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(manifest_path, "manifest.json")
        zf.write(onnx_path, "model.onnx")
        zf.write(tokenizer_path, "tokenizer.json")

    # Cleanup temporary export directory
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("Verifying generated .mcim archive:")
    if not os.path.exists(output_mcim_path):
        raise FileNotFoundError(f"Failed to generate output archive: {output_mcim_path}")

    file_size = os.path.getsize(output_mcim_path)
    print(f"Archive path: {output_mcim_path}")
    print(f"Archive size: {file_size} bytes")

    with zipfile.ZipFile(output_mcim_path, "r") as zf:
        namelist = zf.namelist()
        print(f"Archive contents: {namelist}")
        assert "manifest.json" in namelist, "manifest.json missing in .mcim"
        assert "model.onnx" in namelist, "model.onnx missing in .mcim"
        
        manifest_content = json.loads(zf.read("manifest.json").decode("utf-8"))
        print(f"Verified manifest format_version: {manifest_content.get('format_version')}")
        print(f"Verified manifest model.name: {manifest_content.get('model', {}).get('name')}")

        with tempfile.TemporaryDirectory() as extract_dir:
            extracted_onnx = zf.extract("model.onnx", path=extract_dir)
            session = ort.InferenceSession(extracted_onnx)
            print("Verified ONNX Runtime instantiated InferenceSession directly from extracted model.onnx in .mcim archive.")

    print("Success: dummy-test-v1.mcim generated and verified successfully.")

if __name__ == "__main__":
    main()
