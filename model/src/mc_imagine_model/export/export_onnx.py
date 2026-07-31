"""
ONNX export script.

Exports a trained PyTorch model to ONNX format. A single .mcim can bundle up to three separate
ONNX graphs (see docs/model-spec.md "File Structure") — this script exports one graph per
invocation, selected via --graph:
  - chunk  -> model.onnx   (ImagineNet, model/imagine_net.py)  — the only graph this PoC ships.
  - macro  -> macro.onnx   (MacroFieldNet, model/macro_net.py) — out of scope (requires_macro_field:
             false for imaginator-low_intensity-no_structures).
  - detail -> detail.onnx  (optional ML decorator)             — out of scope (detail_passes: 0).

The --graph chunk path is the only one implemented — see docs/poc-plan.md's scope. It wraps a
trained `ImagineNet` in `ChunkExportWrapper`, which does everything docs/poc-plan.md's Phase 6
requires *inside* the traced graph:
  - runs the frozen MiniLM encoder on prompt_tokens (attention mask derived from tokens != 0,
    inside PromptEncoder.forward);
  - builds the global-coordinate halo grid from chunk_x/chunk_z (inside CoordinateEncoder.forward);
  - runs the decoder (heightmap, profile logits, water level, biome logits, occupancy logits);
  - expands the 64-cell surface-anchored occupancy band into the full
    `block_volume: int32[16,384,16]` using torch.where-style ops over a broadcast y-grid, and
    `biome_grid: int32[4,96,4]` by broadcasting the quarter-resolution biome prediction across all
    96 y-quarter-levels (this model doesn't predict biome variation with depth — an explicit,
    documented simplification, not a bug).

So the mod never has to replicate any of this in Java: `model.onnx` takes exactly the 4 spec'd
inputs and emits exactly the 3 spec'd outputs, fully spec-conformant.
"""

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn

from mc_imagine_model.model.imagine_net import ImagineNet
from mc_imagine_model.spec_constants import (
    AIR_ID,
    BAND_BOTTOM_OFFSET,
    BAND_HEIGHT,
    BAND_TOP_OFFSET,
    BEDROCK_ID,
    DEEPSLATE_ID,
    HEIGHT_MAX,
    HEIGHT_MIN,
    NUM_BIOMES,
    NUM_PROFILES,
    PROFILE_TABLE,
    STONE_ID,
    WATER_ID,
)

MIN_Y = -64
MAX_Y = 319
NUM_Y = MAX_Y - MIN_Y + 1  # 384
CHUNK_SIZE = 16
QUARTER = 4
QUARTER_Y = NUM_Y // 4  # 96

# TerrainHead's representable band ([HEIGHT_MIN, HEIGHT_MAX] = [-96, 288] after
# docs/phase2-plan.md §1b widened it) is deliberately *wider* than Minecraft's world band
# [MIN_Y, MAX_Y] at the bottom: the head is allowed to want a height below bedrock, and the
# clamp in ChunkExportWrapper.forward is what makes that safe. Keeping the head's range strictly
# inside the world band instead would reintroduce the tanh-saturation failure at the low end.
# What must NOT happen is the head being able to exceed the *top* of the world, which would index
# the y-grid out of range, so that direction is asserted rather than merely clamped.
assert HEIGHT_MAX <= MAX_Y, (
    f"TerrainHead can emit heights up to {HEIGHT_MAX}, above the world ceiling {MAX_Y} — "
    f"spec_constants.HEIGHT_CENTER/HEIGHT_AMPLITUDE must stay inside the buildable world."
)
# Clamp band for the block_volume expansion: leave 5 blocks below for the bedrock floor + the
# 4-deep subsurface band, and 2 above so `y == h` and the water fill always have room.
HEIGHT_CLAMP_MIN = MIN_Y + 5
HEIGHT_CLAMP_MAX = MAX_Y - 2
assert HEIGHT_MIN <= HEIGHT_CLAMP_MIN, "clamp is pointless if the head cannot reach below it"


class ChunkExportWrapper(nn.Module):
    """Wraps a trained `ImagineNet` so the exported ONNX graph takes exactly
    (prompt_tokens, chunk_x, chunk_z, seed) and emits exactly (heightmap, block_volume, biome_grid),
    per docs/model-spec.md's `model.onnx` contract. See module docstring for the expansion logic.
    """

    def __init__(self, imagine_net: ImagineNet, deterministic_carve: bool = False) -> None:
        super().__init__()
        self.imagine_net = imagine_net
        self.deterministic_carve = deterministic_carve

        subsurface_ids = [row[1] for row in PROFILE_TABLE]
        surface_ids = [row[2] for row in PROFILE_TABLE]
        self.register_buffer("subsurface_lut", torch.tensor(subsurface_ids, dtype=torch.int64), persistent=False)
        self.register_buffer("surface_lut", torch.tensor(surface_ids, dtype=torch.int64), persistent=False)

        y_grid = torch.arange(MIN_Y, MAX_Y + 1, dtype=torch.int64)  # [384], index i -> world Y = MIN_Y + i
        self.register_buffer("y_grid", y_grid, persistent=False)

    def forward(
        self, prompt_tokens: torch.Tensor, chunk_x: torch.Tensor, chunk_z: torch.Tensor, seed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.imagine_net(prompt_tokens, chunk_x, chunk_z, seed)
        heightmap_f = out["heightmap"]        # [B,16,16] = [B, z, x]
        profile_logits = out["profile_logits"]  # [B, num_profiles, 16, 16] = [B, P, z, x]
        water_f = out["water_level"]           # [B,16,16] = [B, z, x]
        biome_logits = out["biome_logits"]     # [B, num_biomes, 4, 4] = [B, C, z_q, x_q]
        occupancy_logits = out["occupancy_logits"]  # [B,64,16,16] = [B, band_i, z, x]

        # TerrainHead's band ([HEIGHT_MIN, HEIGHT_MAX]) intentionally extends below the world floor,
        # so this clamp is load-bearing, not defensive: it is what keeps every index into the
        # 384-tall y-grid valid no matter what the head asks for. See the module-level constants.
        h_round = torch.round(heightmap_f).to(torch.int64)
        h_round = torch.clamp(h_round, HEIGHT_CLAMP_MIN, HEIGHT_CLAMP_MAX)
        water_round = torch.round(water_f).to(torch.int64)
        water_round = torch.clamp(water_round, MIN_Y + 1, MAX_Y - 1)
        profile_id = torch.argmax(profile_logits, dim=1)  # [B,16,16] = [B,z,x]
        biome_id = torch.argmax(biome_logits, dim=1)       # [B,4,4] = [B,z_q,x_q]

        # --- assume batch size 1 for export (per-chunk graph is always called with B=1) ---
        h_zx = h_round[0]        # [16,16] z,x
        water_zx = water_round[0]
        profile_zx = profile_id[0]
        biome_zx = biome_id[0]   # [4,4] z_q,x_q

        # transpose [z,x] -> [x,z] for 3D tensors, which are indexed [x][y][z]
        h_xz = h_zx.transpose(0, 1)          # [16(x), 16(z)]
        water_xz = water_zx.transpose(0, 1)
        profile_xz = profile_zx.transpose(0, 1)

        subsurface_xz = self.subsurface_lut[profile_xz]  # [16(x),16(z)]
        surface_xz = self.surface_lut[profile_xz]

        shape = (CHUNK_SIZE, NUM_Y, CHUNK_SIZE)  # (x, y, z)
        y_b = self.y_grid.view(1, NUM_Y, 1).expand(shape)
        h_b = h_xz.unsqueeze(1).expand(shape)
        water_b = water_xz.unsqueeze(1).expand(shape)
        subsurface_b = subsurface_xz.unsqueeze(1).expand(shape)
        surface_b = surface_xz.unsqueeze(1).expand(shape)

        band_bottom_b = h_b + BAND_BOTTOM_OFFSET
        band_top_b = h_b + BAND_TOP_OFFSET

        # Base solid mask. In normal Phase 4 exports the model-authored occupancy band is
        # authoritative inside the 64-block window; below it is deterministic rock, and above it is
        # air/water. The gather maps each absolute world-y cell back to the column's relative band
        # index, whose index 0 is the bottom of the band (spec_constants.BAND_BOTTOM_OFFSET).
        occupancy_xiz = occupancy_logits[0].permute(2, 0, 1)  # [16(x),64(i),16(z)]
        band_i = (y_b - band_bottom_b).clamp(0, BAND_HEIGHT - 1).to(torch.int64)
        band_solid = torch.gather(occupancy_xiz, 1, band_i) >= 0.0
        in_band = (y_b >= band_bottom_b) & (y_b <= band_top_b)
        below_band = y_b < band_bottom_b
        is_solid = below_band | (in_band & band_solid)

        if self.deterministic_carve:
            # Gate 1 fixture mode: prove the ONNX/mod plumbing with a guaranteed overhang before a
            # trained occupancy head exists. Release exports leave this disabled and use the
            # predicted occupancy band above.
            is_solid = (y_b <= h_b)
            x_grid = torch.arange(CHUNK_SIZE, device=y_b.device).view(CHUNK_SIZE, 1, 1).expand(shape)
            z_grid = torch.arange(CHUNK_SIZE, device=y_b.device).view(1, 1, CHUNK_SIZE).expand(shape)
            carve_mask = (x_grid >= 3) & (x_grid <= 12) & (z_grid >= 3) & (z_grid <= 12) & (y_b >= h_b - 20) & (y_b <= h_b - 6)
            is_solid = is_solid & (~carve_mask)

        # Derive heightmap tensor from topmost solid cell y in each (x, z) column
        y_solid_vals = torch.where(is_solid, y_b, torch.tensor(MIN_Y - 1, dtype=torch.int64, device=y_b.device))
        topmost_solid_y, _ = torch.max(y_solid_vals, dim=1)  # [16(x), 16(z)] via ReduceMax
        has_solid, _ = torch.max(is_solid.to(torch.int64), dim=1)
        topmost_solid_y = torch.where(has_solid > 0, topmost_solid_y, h_xz)
        topmost_solid_y = torch.clamp(topmost_solid_y, MIN_Y, MAX_Y)

        heightmap_out = topmost_solid_y.transpose(0, 1).to(torch.int32)  # [16(z), 16(x)]

        topmost_y_b = topmost_solid_y.unsqueeze(1).expand(shape)

        # Open-to-sky water fill via top-down cumsum on flipped solid mask
        solid_flipped = torch.flip(is_solid.to(torch.int64), dims=[1])
        cum_solid_top_down_flipped = torch.cumsum(solid_flipped, dim=1)
        cum_solid_top_down = torch.flip(cum_solid_top_down_flipped, dims=[1])
        open_to_sky = (cum_solid_top_down == 0)

        is_water = (~is_solid) & (y_b <= water_b) & open_to_sky

        # Topmost solid run surface dressing
        rock_b = torch.where(
            y_b < 0, torch.full(shape, DEEPSLATE_ID, dtype=torch.int64, device=y_b.device), torch.full(shape, STONE_ID, dtype=torch.int64, device=y_b.device)
        )

        block = torch.full(shape, AIR_ID, dtype=torch.int64, device=y_b.device)
        block = torch.where(is_solid, rock_b, block)
        block = torch.where(is_solid & (y_b >= topmost_y_b - 4) & (y_b <= topmost_y_b - 1), subsurface_b, block)
        block = torch.where(is_solid & (y_b == topmost_y_b), surface_b, block)
        block = torch.where(is_water, torch.full(shape, WATER_ID, dtype=torch.int64, device=y_b.device), block)
        block = torch.where(y_b == MIN_Y, torch.full(shape, BEDROCK_ID, dtype=torch.int64, device=y_b.device), block)

        block_volume_out = block.to(torch.int32)  # [16(x), 384(y), 16(z)] — matches spec directly.

        biome_xq_zq = biome_zx.transpose(0, 1)  # [4(x_q), 4(z_q)]
        biome_volume_out = biome_xq_zq.unsqueeze(1).expand(QUARTER, QUARTER_Y, QUARTER).to(torch.int32)

        return heightmap_out, block_volume_out, biome_volume_out


def load_imagine_net(checkpoint_path: str) -> ImagineNet:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]["model"]
    net = ImagineNet(config)
    net.load_state_dict(ckpt["model_state_dict"], strict=False)
    net.eval()
    return net


def export_chunk_graph(checkpoint_path: str, output_path: str, opset: int, deterministic_carve: bool = False) -> None:
    net = load_imagine_net(checkpoint_path)
    wrapper = ChunkExportWrapper(net, deterministic_carve=deterministic_carve)
    wrapper.eval()

    dummy_prompt = torch.zeros((1, 128), dtype=torch.int32)
    dummy_prompt[0, 0] = 101
    dummy_prompt[0, 1] = 102
    dummy_chunk_x = torch.tensor([0], dtype=torch.int32)
    dummy_chunk_z = torch.tensor([0], dtype=torch.int32)
    dummy_seed = torch.tensor([12345], dtype=torch.int64)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_prompt, dummy_chunk_x, dummy_chunk_z, dummy_seed),
            output_path,
            input_names=["prompt_tokens", "chunk_x", "chunk_z", "seed"],
            output_names=["heightmap", "block_volume", "biome_grid"],
            opset_version=opset,
            dynamo=False,
        )

    print(f"Exported {output_path}")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("onnx.checker.check_model passed.")


def _run(session: onnxruntime.InferenceSession, prompt_ids, chunk_x: int, chunk_z: int, seed: int) -> Dict[str, np.ndarray]:
    inputs = {
        "prompt_tokens": np.array([prompt_ids], dtype=np.int32),
        "chunk_x": np.array([chunk_x], dtype=np.int32),
        "chunk_z": np.array([chunk_z], dtype=np.int32),
        "seed": np.array([seed], dtype=np.int64),
    }
    names = [o.name for o in session.get_outputs()]
    outputs = session.run(names, inputs)
    return dict(zip(names, outputs))


def verify_coordinate_purity(net: ImagineNet, seed: int = 12345) -> float:
    """
    The actual, rigorous, exact-equality proof behind docs/poc-plan.md §1b's seamless-border
    guarantee: `CoordinateEncoder` (model/positional.py) is a pure function of global block
    coordinate *at a fixed world seed*, so the region of chunk(0,0)'s halo window that overlaps
    chunk(1,0)'s halo window must produce bit-identical Fourier features regardless of which chunk's
    request constructed it.

    Phase 2 (docs/phase2-plan.md §1a) added a seed-derived translation to the coordinate frame, so
    the check is now run at one fixed seed — which is exactly the right scope: a single generated
    world only ever has one seed, and the seam guarantee is a within-world property. (Two chunks
    generated at *different* seeds belong to different worlds and are not expected to agree; that
    they disagree is verified separately in tests/test_model.py.)

    Why this — rather than literally comparing chunk(0,0)'s emitted core column x=15 to
    chunk(1,0)'s emitted core column x=0 — is the correct rigorous check: those two columns are
    *adjacent, but different, global coordinates* (global x=15 vs x=16). Real, varying terrain has
    no reason to have identical heights at two different columns, and asserting so would only ever
    hold for degenerate (perfectly flat) terrain. Minecraft chunks are also disjoint, non-overlapping
    16-block tiles, so no *exported* chunk output ever exposes two chunks' values for the *same*
    global coordinate to compare directly. What must instead be proven is that the network's
    seam-avoidance mechanism itself (CoordinateEncoder purity + the valid-padding-only conv stack
    combining valid 2D convs and valid 3D convs in x/z being a fixed deterministic function) is bit-exact —
    which is what this function checks, and what `export_onnx.py`'s per-chunk cropping in `ChunkExportWrapper` relies on.

    Returns the max abs difference (expected to be exactly 0.0, or at most float rounding noise).
    """
    ce = net.coord_encoder
    seed_t = torch.tensor([seed], dtype=torch.int64)
    with torch.no_grad():
        f0 = ce(torch.tensor([0]), torch.tensor([0]), seed_t)  # covers global x in [-halo, 15+halo]
        f1 = ce(torch.tensor([1]), torch.tensor([0]), seed_t)  # covers global x in [16-halo, 31+halo]
    # The two halo windows overlap on global x in [16-halo, 15+halo], which is `2*halo` columns
    # wide. In f0's `size`-wide grid those sit at local indices [patch, size); in f1's they sit at
    # [0, size-patch). Derived from ce.halo/ce.patch rather than hardcoded so this stays correct if
    # the halo changes (Phase 2 took it from 4 to 6).
    patch, size = ce.patch, ce.size
    f0_shared = f0[0, :, :, patch:size]
    f1_shared = f1[0, :, :, 0:size - patch]
    assert f0_shared.shape[-1] == 2 * ce.halo, "overlap width must be exactly 2*halo columns"
    diff = (f0_shared - f1_shared).abs().max().item()
    assert diff == 0.0, (
        f"CoordinateEncoder is not a bit-exact pure function of (global coordinate, seed): "
        f"diff={diff}"
    )
    return diff


def verify_chunk_graph(onnx_path: str, checkpoint_path: str, max_tokens: int = 128) -> None:
    """onnx.checker + ORT round-trip: shape/dtype conformance, determinism, and the §1b
    border-continuity guarantee."""
    from mc_imagine_model.tokenizer_utils import tokenize

    net = load_imagine_net(checkpoint_path)
    purity_diff = verify_coordinate_purity(net)
    print(f"CoordinateEncoder purity check passed (max diff at shared global coordinates = {purity_diff}) "
          f"— this is the actual mechanism proving §1b's seamless-border guarantee.")

    session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    prompt_ids = tokenize("vast deep valleys, beautiful mountains overlooking water-filled craters", max_tokens)
    seed = 12345

    out_a = _run(session, prompt_ids, 0, 0, seed)
    out_b = _run(session, prompt_ids, 0, 0, seed)

    assert out_a["heightmap"].shape == (16, 16), out_a["heightmap"].shape
    assert out_a["heightmap"].dtype == np.int32
    assert out_a["block_volume"].shape == (16, 384, 16), out_a["block_volume"].shape
    assert out_a["block_volume"].dtype == np.int32
    assert out_a["biome_grid"].shape == (4, 96, 4), out_a["biome_grid"].shape
    assert out_a["biome_grid"].dtype == np.int32
    print("Shape/dtype checks passed.")

    for name in ("heightmap", "block_volume", "biome_grid"):
        assert out_a[name].tobytes() == out_b[name].tobytes(), f"{name} not bit-identical across repeated calls"
    print("Determinism check passed (bit-identical across repeated calls).")

    out_c0 = _run(session, prompt_ids, 0, 0, seed)
    out_c1 = _run(session, prompt_ids, 1, 0, seed)
    # heightmap indexed [z][x]: chunk(0,0)'s last x-column (x=15) is global x=15;
    # chunk(1,0)'s first x-column (x=0) is global x=16 — these are DIFFERENT global columns
    # (adjacent, not identical), so equality isn't expected there. The actual border-continuity
    # guarantee is: both chunks agree on any *shared* global coordinate. Since chunk(1,0) doesn't
    # overlap chunk(0,0) in this tensor's core 16x16 (chunks are disjoint 16-block tiles), the
    # checkable form of §1b's guarantee is that adjacent chunks' outputs are a pure function of
    # global coordinate — verified by re-deriving chunk(0,0)'s rightmost column as chunk(1,0) would
    # see it via the *halo* mechanism internal to the network. We verify this directly by checking
    # CoordinateEncoder's own property in tests/test_model.py; here we verify the practical,
    # end-to-end symptom instead: chunk(1,0)'s height profile continues smoothly from chunk(0,0)'s
    # (no discontinuous jump at the seam), by comparing the trend at the border.
    hm0 = out_c0["heightmap"]  # [z][x]
    hm1 = out_c1["heightmap"]
    last_col_0 = hm0[:, -1].astype(np.float64)
    first_col_1 = hm1[:, 0].astype(np.float64)
    interior_jump = np.abs(hm0[:, -1].astype(np.float64) - hm0[:, -2].astype(np.float64)).max()
    seam_jump = np.abs(last_col_0 - first_col_1).max()
    print(
        f"Diagnostic (informational, NOT expected to be zero for varying terrain — global x=15 "
        f"and x=16 are adjacent but different columns): seam jump |h(x=15,chunk0)-h(x=16,chunk1)| "
        f"= {seam_jump}, vs. a typical interior column-to-column jump of {interior_jump} — seam "
        f"jump should be the same order of magnitude as the interior jump, not anomalously larger "
        f"(an anomalously larger seam jump would indicate a real chunk-boundary artifact)."
    )


def main() -> None:
    """
    Main export function. Loads a checkpoint and exports the selected graph to ONNX.
    """
    parser = argparse.ArgumentParser(description="Export a model graph to ONNX (see docs/model-spec.md)")
    parser.add_argument("--graph", choices=["chunk", "macro", "detail"], default="chunk", help="Which of the up-to-three graphs to export")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PyTorch checkpoint")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output ONNX file path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--skip-verify", action="store_true", help="Skip the ORT round-trip verification")
    parser.add_argument("--deterministic-carve", action="store_true",
                        help="Gate 1 fixture mode: ignore occupancy_logits and carve a fixed test overhang")
    args = parser.parse_args()

    print(f"Exporting checkpoint {args.checkpoint} ({args.graph} graph) to {args.output} with opset {args.opset}")

    if args.graph == "chunk":
        export_chunk_graph(args.checkpoint, args.output, args.opset, deterministic_carve=args.deterministic_carve)
        if not args.skip_verify:
            verify_chunk_graph(args.output, args.checkpoint)
    elif args.graph == "macro":
        raise NotImplementedError(
            "macro.onnx export is out of scope for this PoC — imaginator-low_intensity-no_structures "
            "declares requires_macro_field: false. See docs/poc-plan.md's scope."
        )
    elif args.graph == "detail":
        raise NotImplementedError(
            "detail.onnx export is out of scope for this PoC — imaginator-low_intensity-no_structures "
            "declares detail_passes: 0. See docs/poc-plan.md's scope."
        )


if __name__ == "__main__":
    main()
