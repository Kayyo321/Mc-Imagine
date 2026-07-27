# Mc-Imagine Model Format Specification (.mcim)

**Version:** 0.1.0 (Draft)

## Overview
The `.mcim` file format is the standard distributable format for Mc-Imagine world generation models. An `.mcim` file is a ZIP archive containing a trained ONNX model, tokenizer configuration, and a metadata manifest.

## File Structure
A typical `.mcim` archive contains the following structure:
```
model.mcim
├── manifest.json
├── model.onnx
└── tokenizer.json
```

## `manifest.json` Schema
The manifest provides the mod with essential information about how to load and use the model.

- **`format_version`**: (String) Version of the .mcim spec (e.g., "0.1.0").
- **`model`**:
  - `name`: (String) Display name of the model.
  - `version`: (String) Model version.
  - `author`: (String) Creator of the model.
  - `description`: (String) Brief description of what the model generates well.
  - `license`: (String) License under which the model is distributed.
- **`capabilities`**:
  - `terrain`: (Boolean) True if the model predicts terrain heightmap/volume.
  - `biomes`: (Boolean) True if the model predicts biomes.
  - `caves`: (Boolean) True if the model generates cave systems.
  - `structures`: (Boolean) True if the model places structures.
  - `block_palette`: (List of Strings) Namespaced IDs of blocks the model can output.
  - `max_prompt_tokens`: (Integer) Maximum context length for the text prompt.
- **`requirements`**:
  - `min_ram_mb`: (Integer) Minimum system RAM required.
  - `recommended_vram_mb`: (Integer) Recommended GPU VRAM for optimal performance.
  - `onnx_opset`: (Integer) The ONNX opset version the model was exported with.
- **`io`**:
  - `input_names`: (List of Strings) Names of the input tensors in the ONNX graph.
  - `output_names`: (List of Strings) Names of the output tensors in the ONNX graph.

## Input Tensors
Models must accept the following standard input tensors:
- `prompt_tokens`: `int32[1, max_tokens]` - Tokenized text prompt.
- `chunk_x`: `int32[1]` - X coordinate of the chunk being generated.
- `chunk_z`: `int32[1]` - Z coordinate of the chunk being generated.
- `seed`: `int64[1]` - The world seed.
- `neighbor_context`: `float32[4, 16, 384]` - (Optional) Encoded context from adjacent chunks.

## Output Tensors
Models must produce the following standard output tensors:
- `heightmap`: `int32[16, 16]` - Surface elevation for the chunk.
- `block_volume`: `int32[16, 384, 16]` - 3D grid of block state IDs mapped to the palette.
- `biome_grid`: `int32[4, 96, 4]` - Biome distribution per 4x4x4 volume.
- `structure_markers`: `float32[N, 5]` - List of potential structures (type_id, x, y, z, probability).

## Tokenizer Format
`tokenizer.json` follows the standard HuggingFace Tokenizers format, providing vocabulary, merges, and normalization rules needed to convert the player's text prompt into `prompt_tokens`.

## Versioning Policy
Breaking changes to the manifest schema or expected tensor shapes will result in a minor version bump (e.g., 0.1.0 to 0.2.0). The mod will warn players if they attempt to load an incompatible format version.

## Example `manifest.json`
```json
{
  "format_version": "0.1.0",
  "model": {
    "name": "Fantasy Lands Base",
    "version": "1.0.0",
    "author": "Mc-Imagine Team",
    "description": "A baseline model trained on high-fantasy terrain.",
    "license": "CC-BY-4.0"
  },
  "capabilities": {
    "terrain": true,
    "biomes": true,
    "caves": false,
    "structures": false,
    "block_palette": ["minecraft:stone", "minecraft:dirt", "minecraft:grass_block"],
    "max_prompt_tokens": 128
  },
  "requirements": {
    "min_ram_mb": 4096,
    "recommended_vram_mb": 2048,
    "onnx_opset": 17
  },
  "io": {
    "input_names": ["prompt_tokens", "chunk_x", "chunk_z", "seed"],
    "output_names": ["heightmap", "block_volume", "biome_grid"]
  }
}
```
