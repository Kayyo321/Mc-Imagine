#!/usr/bin/env python3
"""
Terracing Diagnosis Script (Mc-Imagine Phase 3.1 Section 2.1).

Loads an ImagineNet checkpoint model and analyzes heightmap output across multiple chunks
for terracing / staircasing artifacts. Computes height-delta statistics, step distributions,
integer cliff spike ratios, mean grade, and seam vs interior continuity.

Usage:
    PYTHONPATH=model/src python model/src/mc_imagine_model/scripts/diagnose_terracing.py \
        --checkpoint model/training_runs/run1/best.pt \
        --prompt "mountainous terrain with deep valleys and sharp peaks" \
        --chunk-span 8 \
        --output terracing_report.json
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch

from mc_imagine_model.export.export_onnx import load_imagine_net
from mc_imagine_model.inference_utils import render_area

DEFAULT_CHECKPOINT = "model/training_runs/run1/best.pt"
DEFAULT_PROMPT = "mountainous terrain with deep valleys and sharp peaks"


def compute_terracing_metrics(
    heightfield: np.ndarray,
    chunk_span: int = 8,
) -> Dict[str, Any]:
    """
    Computes quantitative terracing metrics on a 2D float heightfield (HxW).

    Args:
        heightfield: 2D numpy array of predicted heights (float).
        chunk_span: Number of 16x16 chunks across each dimension.

    Returns:
        Dict containing histogram, step distribution, and quantitative metrics.
    """
    h_float = heightfield.astype(np.float64)
    h_int = np.round(h_float).astype(np.int64)

    # Calculate horizontal adjacent deltas: |H[z, x+1] - H[z, x]|
    dx_float = np.abs(np.diff(h_float, axis=1))
    dz_float = np.abs(np.diff(h_float, axis=0))
    all_deltas_float = np.concatenate([dx_float.ravel(), dz_float.ravel()])

    dx_int = np.abs(np.diff(h_int, axis=1))
    dz_int = np.abs(np.diff(h_int, axis=0))
    all_deltas_int = np.concatenate([dx_int.ravel(), dz_int.ravel()])

    total_edges = len(all_deltas_float)
    mean_grade = float(np.mean(all_deltas_float))
    std_grade = float(np.std(all_deltas_float))
    max_delta = float(np.max(all_deltas_float))
    min_delta = float(np.min(all_deltas_float))

    # Step distribution (binned integer steps: 0, 1, 2, 3, 4, 5+)
    step_counts = {
        "step_0_flat": int(np.sum(all_deltas_int == 0)),
        "step_1": int(np.sum(all_deltas_int == 1)),
        "step_2": int(np.sum(all_deltas_int == 2)),
        "step_3": int(np.sum(all_deltas_int == 3)),
        "step_4": int(np.sum(all_deltas_int == 4)),
        "step_5_plus": int(np.sum(all_deltas_int >= 5)),
    }
    step_proportions = {
        k: float(v / total_edges) for k, v in step_counts.items()
    }

    # Integer cliff spike ratio:
    # For continuous float deltas, calculate residual r = |delta - round(delta)|
    # Smooth slopes have uniformly distributed residuals r in [0, 0.5].
    # Terracing causes deltas to cluster near exact integer values (r < 0.15).
    non_zero_mask = all_deltas_float >= 0.5
    num_non_zero = int(np.sum(non_zero_mask))
    if num_non_zero > 0:
        residuals = np.abs(all_deltas_float[non_zero_mask] - np.round(all_deltas_float[non_zero_mask]))
        near_integer_count = int(np.sum(residuals < 0.15))
        integer_spike_ratio = float(near_integer_count / num_non_zero)
    else:
        integer_spike_ratio = 0.0

    # Fine-grained Delta Histogram (binned 0 to max_delta)
    max_bin = int(np.ceil(max_delta)) + 1
    bins = np.linspace(0, max_bin, min(max_bin * 2 + 1, 51))
    hist, bin_edges = np.histogram(all_deltas_float, bins=bins)
    histogram_data = {
        "bin_edges": [float(b) for b in bin_edges],
        "counts": [int(c) for c in hist],
        "proportions": [float(c / total_edges) for c in hist],
    }

    # Seam vs Interior gradient analysis
    # Chunk boundaries occur at x = 16, 32, 48, ... and z = 16, 32, 48, ...
    H, W = h_float.shape
    seam_deltas_x = []
    interior_deltas_x = []
    for x in range(W - 1):
        is_seam = ((x + 1) % 16 == 0)
        col_deltas = dx_float[:, x]
        if is_seam:
            seam_deltas_x.extend(col_deltas)
        else:
            interior_deltas_x.extend(col_deltas)

    seam_deltas_z = []
    interior_deltas_z = []
    for z in range(H - 1):
        is_seam = ((z + 1) % 16 == 0)
        row_deltas = dz_float[z, :]
        if is_seam:
            seam_deltas_z.extend(row_deltas)
        else:
            interior_deltas_z.extend(row_deltas)

    all_seam_deltas = np.array(seam_deltas_x + seam_deltas_z)
    all_interior_deltas = np.array(interior_deltas_x + interior_deltas_z)

    mean_seam_delta = float(np.mean(all_seam_deltas)) if len(all_seam_deltas) > 0 else 0.0
    mean_interior_delta = float(np.mean(all_interior_deltas)) if len(all_interior_deltas) > 0 else 0.0
    seam_jump_ratio = float(mean_seam_delta / mean_interior_delta) if mean_interior_delta > 1e-6 else 1.0

    return {
        "total_sampled_edges": total_edges,
        "mean_grade": mean_grade,
        "std_grade": std_grade,
        "min_delta": min_delta,
        "max_delta": max_delta,
        "step_distribution_counts": step_counts,
        "step_distribution_proportions": step_proportions,
        "integer_cliff_spike_ratio": integer_spike_ratio,
        "flat_shelf_ratio": step_proportions["step_0_flat"],
        "steep_cliff_ratio": float(step_proportions["step_3"] + step_proportions["step_4"] + step_proportions["step_5_plus"]),
        "seam_analysis": {
            "mean_seam_delta": mean_seam_delta,
            "mean_interior_delta": mean_interior_delta,
            "seam_jump_ratio": seam_jump_ratio,
        },
        "histogram": histogram_data,
    }


def save_diagnostic_plot(
    heightfield: np.ndarray,
    metrics: Dict[str, Any],
    prompt: str,
    output_path: str,
) -> None:
    """Saves a diagnostic plot combining heightmap visualization and delta histogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Heightfield map
    im0 = axes[0].imshow(heightfield, cmap="terrain", origin="upper")
    axes[0].set_title(f"Heightmap (Prompt: {prompt[:30]}...)")
    plt.colorbar(im0, ax=axes[0], label="Height (y)")

    # Height Delta Histogram
    hist = metrics["histogram"]
    bin_edges = hist["bin_edges"]
    counts = hist["counts"]
    centers = 0.5 * (np.array(bin_edges[:-1]) + np.array(bin_edges[1:]))
    widths = np.diff(bin_edges)

    axes[1].bar(centers, counts, width=widths, color="steelblue", edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Height Delta (|Δh| blocks)")
    axes[1].set_ylabel("Edge Count")
    axes[1].set_title("Column-to-Column Height Delta Histogram")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", linestyle="--", alpha=0.5)

    plt.suptitle("Terracing Diagnostic Report", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Terracing Diagnosis Tool — measures staircasing artifacts and height delta distributions."
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"Path to model checkpoint (.pt) (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"Text prompt for terrain generation (default: '{DEFAULT_PROMPT}')",
    )
    parser.add_argument(
        "--chunk-span",
        type=int,
        default=8,
        help="Chunk grid size (default: 8 -> 8x8 = 64 chunks, 128x128 columns)",
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

    print(f"=== Terracing Diagnosis Script ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Prompt:     '{args.prompt}'")
    print(f"Chunk span: {args.chunk_span}x{args.chunk_span} ({args.chunk_span * 16}x{args.chunk_span * 16} columns)")
    print(f"Seed:       {args.seed}")
    print(f"Device:     {args.device}")

    net = load_imagine_net(args.checkpoint)
    net.to(args.device)

    print("Running inference to generate heightfield...")
    full_h, full_p, water_level = render_area(
        model=net,
        prompt=args.prompt,
        seed=args.seed,
        chunk_span=args.chunk_span,
        device=args.device,
    )

    print("Computing terracing metrics...")
    metrics = compute_terracing_metrics(full_h, chunk_span=args.chunk_span)

    # Display Quantitative Report
    print("\n" + "=" * 55)
    print("           QUANTITATIVE TERRACING METRICS           ")
    print("=" * 55)
    print(f"Height Range:               [{full_h.min():.2f}, {full_h.max():.2f}] blocks")
    print(f"Total Column Edges Sampled: {metrics['total_sampled_edges']:,}")
    print(f"Mean Grade (Avg |Δh|):      {metrics['mean_grade']:.4f} blocks")
    print(f"Std Grade:                  {metrics['std_grade']:.4f} blocks")
    print(f"Max Height Delta:           {metrics['max_delta']:.4f} blocks")
    print(f"Integer Cliff Spike Ratio:  {metrics['integer_cliff_spike_ratio'] * 100:.2f}%")
    print(f"Flat Shelf Ratio (Step 0):  {metrics['flat_shelf_ratio'] * 100:.2f}%")
    print(f"Steep Cliff Ratio (Step ≥3):{metrics['steep_cliff_ratio'] * 100:.2f}%")

    print("\n--- STEP FREQUENCY DISTRIBUTION ---")
    for step_name, count in metrics["step_distribution_counts"].items():
        prop = metrics["step_distribution_proportions"][step_name]
        print(f"  {step_name:<16}: {count:>6} edges ({prop * 100:6.2f}%)")

    seam = metrics["seam_analysis"]
    print("\n--- SEAM VS INTERIOR GRADIENT CONTINUITY ---")
    print(f"  Mean Seam Delta (Chunk Boundary): {seam['mean_seam_delta']:.4f}")
    print(f"  Mean Interior Delta:               {seam['mean_interior_delta']:.4f}")
    print(f"  Seam Jump Ratio (Boundary/Interior):{seam['seam_jump_ratio']:.4f}")
    print("=" * 55)

    # Handle Output saving
    if args.output:
        out_path = args.output
        if out_path.endswith(".png"):
            save_diagnostic_plot(full_h, metrics, args.prompt, out_path)
            print(f"\nSaved diagnostic plot image to: {out_path}")
        else:
            # Default to JSON export
            out_data = {
                "checkpoint": args.checkpoint,
                "prompt": args.prompt,
                "chunk_span": args.chunk_span,
                "seed": args.seed,
                "height_min": float(full_h.min()),
                "height_max": float(full_h.max()),
                "metrics": metrics,
            }
            with open(out_path, "w") as f:
                json.dump(out_data, f, indent=2)
            print(f"\nSaved diagnostic metrics JSON report to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
