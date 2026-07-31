#!/usr/bin/env python3
"""
Speckle Diagnosis Script (Mc-Imagine Phase 3.1 Section 2.2).

Loads an ImagineNet checkpoint model and analyzes predicted biome_grid output across chunks
for spatial coherence vs high-frequency speckled noise. Computes neighbor identity ratio,
spatial transition frequency, isolated biome cell percentage, contiguous patch size distribution,
and seam vs interior transition stats.

Usage:
    PYTHONPATH=model/src python model/src/mc_imagine_model/scripts/diagnose_speckle.py \
        --checkpoint model/training_runs/run1/best.pt \
        --prompt "mountainous terrain with deep valleys and sharp peaks" \
        --chunk-span 8 \
        --output speckle_report.json
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import label, generate_binary_structure
import torch

from mc_imagine_model.export.export_onnx import load_imagine_net
from mc_imagine_model.spec_constants import BIOME_PALETTE
from mc_imagine_model.tokenizer_utils import tokenize

DEFAULT_CHECKPOINT = "model/training_runs/run1/best.pt"
DEFAULT_PROMPT = "mountainous terrain with deep valleys and sharp peaks"


@torch.no_grad()
def render_biome_area(
    model: torch.nn.Module,
    prompt: str,
    seed: int,
    chunk_span: int = 8,
    origin_chunk: Tuple[int, int] = (0, 0),
    device: str = "cpu",
    max_tokens: int = 128,
) -> np.ndarray:
    """
    Runs model over chunk_span x chunk_span tile of chunks and extracts the
    (chunk_span*4, chunk_span*4) biome ID grid.
    """
    model.eval()
    tokens = tokenize(prompt, max_tokens=max_tokens)
    b = chunk_span * chunk_span

    ox, oz = origin_chunk
    chunk_x_list = [ox + (idx % chunk_span) for idx in range(b)]
    chunk_z_list = [oz + (idx // chunk_span) for idx in range(b)]

    prompt_tokens = torch.tensor([tokens] * b, dtype=torch.int32, device=device)
    chunk_x = torch.tensor(chunk_x_list, dtype=torch.int32, device=device)
    chunk_z = torch.tensor(chunk_z_list, dtype=torch.int32, device=device)
    seeds = torch.tensor([seed] * b, dtype=torch.int64, device=device)

    out = model(prompt_tokens, chunk_x, chunk_z, seeds)
    biome_logits = out["biome_logits"].detach().to("cpu").numpy()  # [B, num_biomes, 4, 4]
    biome_id = biome_logits.argmax(axis=1)  # [B, 4, 4]

    full_biome = np.zeros((chunk_span * 4, chunk_span * 4), dtype=np.int64)
    for idx in range(b):
        row = idx // chunk_span
        col = idx % chunk_span
        full_biome[row * 4:(row + 1) * 4, col * 4:(col + 1) * 4] = biome_id[idx]

    return full_biome


def compute_speckle_metrics(
    biome_grid: np.ndarray,
    chunk_span: int = 8,
) -> Dict[str, Any]:
    """
    Computes quantitative spatial coherence and speckle noise metrics on a 2D biome grid (RxC).

    Args:
        biome_grid: 2D numpy array of biome IDs (int64).
        chunk_span: Number of 16x16 chunks per dimension (grid size is chunk_span*4 x chunk_span*4).

    Returns:
        Dict containing neighbor identity ratio, isolated cell %, patch distribution, and seam analysis.
    """
    grid = biome_grid.astype(np.int64)
    R, C = grid.shape
    total_cells = R * C

    # 1. Neighbor Identity Ratio (Horizontal & Vertical 4-neighbor edges)
    h_match = np.sum(grid[:, :-1] == grid[:, 1:])
    h_total = R * (C - 1)

    v_match = np.sum(grid[:-1, :] == grid[1:, :])
    v_total = (R - 1) * C

    total_edges = h_total + v_total
    matching_edges = int(h_match + v_match)
    neighbor_identity_ratio = float(matching_edges / total_edges) if total_edges > 0 else 1.0
    transition_frequency = float(1.0 - neighbor_identity_ratio)

    # 2. Isolated Biome Cell Analysis (Single-cell speckles with zero matching 4-neighbors)
    isolated_cells = 0
    weakly_isolated_cells = 0  # <2 matching neighbors out of 4

    for r in range(R):
        for c in range(C):
            val = grid[r, c]
            neighbors = []
            if r > 0:
                neighbors.append(grid[r - 1, c])
            if r < R - 1:
                neighbors.append(grid[r + 1, c])
            if c > 0:
                neighbors.append(grid[r, c - 1])
            if c < C - 1:
                neighbors.append(grid[r, c + 1])

            matches = sum(1 for n in neighbors if n == val)
            if matches == 0:
                isolated_cells += 1
            if matches < 2:
                weakly_isolated_cells += 1

    isolated_cell_pct = float(isolated_cells / total_cells * 100.0)
    weakly_isolated_cell_pct = float(weakly_isolated_cells / total_cells * 100.0)

    # 3. Contiguous Biome Patch Analysis (4-connected components per biome)
    unique_biomes = np.unique(grid)
    all_patch_sizes: List[int] = []
    structure_4conn = generate_binary_structure(2, 1)  # 4-connectivity cross

    for b_id in unique_biomes:
        mask = (grid == b_id)
        labeled_array, num_features = label(mask, structure=structure_4conn)
        if num_features > 0:
            sizes = np.bincount(labeled_array.ravel())[1:]
            all_patch_sizes.extend(sizes.tolist())

    total_patches = len(all_patch_sizes)
    mean_patch_size = float(np.mean(all_patch_sizes)) if total_patches > 0 else 0.0
    median_patch_size = float(np.median(all_patch_sizes)) if total_patches > 0 else 0.0
    max_patch_size = int(np.max(all_patch_sizes)) if total_patches > 0 else 0

    patch_size_buckets = {
        "singleton_1_cell": int(sum(1 for s in all_patch_sizes if s == 1)),
        "small_2_to_4_cells": int(sum(1 for s in all_patch_sizes if 2 <= s <= 4)),
        "medium_5_to_16_cells": int(sum(1 for s in all_patch_sizes if 5 <= s <= 16)),
        "large_17_plus_cells": int(sum(1 for s in all_patch_sizes if s >= 17)),
    }

    # 4. Biome Frequency Composition
    biome_counts = {}
    biome_percentages = {}
    for b_id in range(len(BIOME_PALETTE)):
        cnt = int(np.sum(grid == b_id))
        if cnt > 0 or b_id in unique_biomes:
            name = BIOME_PALETTE[b_id] if b_id < len(BIOME_PALETTE) else f"biome_{b_id}"
            biome_counts[name] = cnt
            biome_percentages[name] = float(cnt / total_cells * 100.0)

    # 5. Seam vs Interior Spatial Coherence
    seam_matches = 0
    seam_edges = 0
    interior_matches = 0
    interior_edges = 0

    # Horizontal transitions
    for c in range(C - 1):
        is_seam = ((c + 1) % 4 == 0)
        col_matches = np.sum(grid[:, c] == grid[:, c + 1])
        if is_seam:
            seam_matches += int(col_matches)
            seam_edges += R
        else:
            interior_matches += int(col_matches)
            interior_edges += R

    # Vertical transitions
    for r in range(R - 1):
        is_seam = ((r + 1) % 4 == 0)
        row_matches = np.sum(grid[r, :] == grid[r + 1, :])
        if is_seam:
            seam_matches += int(row_matches)
            seam_edges += C
        else:
            interior_matches += int(row_matches)
            interior_edges += C

    seam_identity_ratio = float(seam_matches / seam_edges) if seam_edges > 0 else 1.0
    interior_identity_ratio = float(interior_matches / interior_edges) if interior_edges > 0 else 1.0

    # 6. Overall Coherence Classification
    if neighbor_identity_ratio >= 0.75 and isolated_cell_pct <= 3.0:
        coherence_classification = "SPATIALLY_COHERENT"
    elif neighbor_identity_ratio >= 0.60 and isolated_cell_pct <= 8.0:
        coherence_classification = "MODERATELY_COHERENT"
    else:
        coherence_classification = "HIGH_FREQUENCY_SPECKLED_NOISE"

    return {
        "total_cells": total_cells,
        "grid_shape": [R, C],
        "neighbor_identity_ratio": neighbor_identity_ratio,
        "transition_frequency": transition_frequency,
        "isolated_cell_count": isolated_cells,
        "isolated_cell_percentage": isolated_cell_pct,
        "weakly_isolated_cell_percentage": weakly_isolated_cell_pct,
        "patch_analysis": {
            "total_patches": total_patches,
            "mean_patch_size_cells": mean_patch_size,
            "median_patch_size_cells": median_patch_size,
            "max_patch_size_cells": max_patch_size,
            "patch_size_buckets": patch_size_buckets,
        },
        "biome_distribution": {
            "unique_biome_count": int(len(unique_biomes)),
            "counts": biome_counts,
            "percentages": biome_percentages,
        },
        "seam_analysis": {
            "seam_identity_ratio": seam_identity_ratio,
            "interior_identity_ratio": interior_identity_ratio,
        },
        "coherence_classification": coherence_classification,
    }


def save_diagnostic_plot(
    biome_grid: np.ndarray,
    metrics: Dict[str, Any],
    prompt: str,
    output_path: str,
) -> None:
    """Saves a diagnostic plot visualizing the biome grid map and patch distribution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mc_imagine_model.viz import _BIOME_COLORS

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Biome Map
    grid = biome_grid.astype(np.int64)
    rgb = np.zeros((*grid.shape, 3), dtype=float)
    for b_id in range(len(_BIOME_COLORS)):
        rgb[grid == b_id] = _BIOME_COLORS[b_id]

    axes[0].imshow(rgb, origin="upper")
    axes[0].set_title(f"Predicted Biome Grid (4x4 cells)\nPrompt: {prompt[:30]}...")
    axes[0].set_xlabel("X Cell")
    axes[0].set_ylabel("Z Cell")

    # Patch size buckets histogram
    buckets = metrics["patch_analysis"]["patch_size_buckets"]
    labels = list(buckets.keys())
    counts = list(buckets.values())
    short_labels = ["1 Cell\n(Singleton)", "2-4 Cells\n(Small)", "5-16 Cells\n(Medium)", "17+ Cells\n(Large)"]

    axes[1].bar(short_labels, counts, color="forestgreen", edgecolor="black", alpha=0.7)
    axes[1].set_ylabel("Number of Contiguous Patches")
    axes[1].set_title("Biome Patch Size Distribution")
    axes[1].grid(True, which="both", linestyle="--", alpha=0.5)

    classification = metrics["coherence_classification"]
    plt.suptitle(f"Speckle Diagnostic Report — [{classification}]", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Speckle Diagnosis Tool — measures spatial coherence and biome noise density."
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"Path to model checkpoint (.pt) (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"Text prompt for biome prediction (default: '{DEFAULT_PROMPT}')",
    )
    parser.add_argument(
        "--chunk-span",
        type=int,
        default=8,
        help="Chunk grid size (default: 8 -> 8x8 chunks = 32x32 4x4 cells)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="World seed for coordinate encoding (default: 12345)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch execution device (cpu, mps, cuda) (default: cpu)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to output file (.json for metrics data, .png for plot visualization)",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: Checkpoint file not found at {args.checkpoint}", file=sys.stderr)
        return 1

    print(f"=== Speckle Diagnosis Script ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Prompt:     '{args.prompt}'")
    print(f"Chunk span: {args.chunk_span}x{args.chunk_span} ({args.chunk_span * 4}x{args.chunk_span * 4} 4x4 cells)")
    print(f"Seed:       {args.seed}")
    print(f"Device:     {args.device}")

    net = load_imagine_net(args.checkpoint)
    net.to(args.device)

    print("Running inference to predict biome grid...")
    biome_grid = render_biome_area(
        model=net,
        prompt=args.prompt,
        seed=args.seed,
        chunk_span=args.chunk_span,
        device=args.device,
    )

    print("Computing speckle and spatial coherence metrics...")
    metrics = compute_speckle_metrics(biome_grid, chunk_span=args.chunk_span)

    # Display Quantitative Report
    print("\n" + "=" * 55)
    print("             QUANTITATIVE SPECKLE METRICS           ")
    print("=" * 55)
    print(f"Coherence Classification:    {metrics['coherence_classification']}")
    print(f"Total Biome Cells Analyzed: {metrics['total_cells']:,}")
    print(f"Neighbor Identity Ratio:     {metrics['neighbor_identity_ratio'] * 100:.2f}% (higher = more coherent)")
    print(f"Transition Frequency:        {metrics['transition_frequency'] * 100:.2f}% (lower = cleaner boundaries)")
    print(f"Isolated Cell Percentage:    {metrics['isolated_cell_percentage']:.2f}% (singletons)")
    print(f"Weakly Isolated Cell Pct:   {metrics['weakly_isolated_cell_percentage']:.2f}% (<2 matching neighbors)")

    patch = metrics["patch_analysis"]
    print("\n--- BIOME PATCH SIZE ANALYSIS ---")
    print(f"  Total Contiguous Patches:  {patch['total_patches']}")
    print(f"  Mean Patch Size:           {patch['mean_patch_size_cells']:.2f} cells")
    print(f"  Median Patch Size:         {patch['median_patch_size_cells']:.2f} cells")
    print(f"  Max Patch Size:            {patch['max_patch_size_cells']} cells")
    for bucket_name, count in patch["patch_size_buckets"].items():
        print(f"    {bucket_name:<20}: {count:>4} patches")

    dist = metrics["biome_distribution"]
    print("\n--- BIOME COMPOSITION ---")
    print(f"  Unique Biomes Present: {dist['unique_biome_count']}")
    for name, pct in dist["percentages"].items():
        cnt = dist["counts"][name]
        print(f"    {name:<18}: {cnt:>5} cells ({pct:6.2f}%)")

    seam = metrics["seam_analysis"]
    print("\n--- SEAM VS INTERIOR SPATIAL COHERENCE ---")
    print(f"  Seam Identity Ratio (Chunk Boundary): {seam['seam_identity_ratio'] * 100:.2f}%")
    print(f"  Interior Identity Ratio:               {seam['interior_identity_ratio'] * 100:.2f}%")
    print("=" * 55)

    # Handle Output saving
    if args.output:
        out_path = args.output
        if out_path.endswith(".png"):
            save_diagnostic_plot(biome_grid, metrics, args.prompt, out_path)
            print(f"\nSaved diagnostic plot image to: {out_path}")
        else:
            # Default to JSON export
            out_data = {
                "checkpoint": args.checkpoint,
                "prompt": args.prompt,
                "chunk_span": args.chunk_span,
                "seed": args.seed,
                "metrics": metrics,
            }
            with open(out_path, "w") as f:
                json.dump(out_data, f, indent=2)
            print(f"\nSaved diagnostic metrics JSON report to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
