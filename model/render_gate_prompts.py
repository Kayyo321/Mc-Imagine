#!/usr/bin/env python3
"""
Renders docs/poc-plan.md's Phase 5 gate: the 6 held-out prompts, each at a fixed seed, from a
trained checkpoint. Produces one combined grid PNG and one PNG per prompt, plus a small text
summary of per-prompt height statistics (useful for judging "visibly, unmistakably different from
each other" quantitatively as well as visually).

Usage: python3 render_gate_prompts.py --checkpoint training_runs/run1/best.pt --out-dir training_runs/run1/gate
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import torch

from mc_imagine_model.export.export_onnx import load_imagine_net
from mc_imagine_model.inference_utils import render_area
from mc_imagine_model.spec_constants import GATE_PROMPTS, GATE_SEED
from mc_imagine_model.viz import save_prompt_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--chunk-span", type=int, default=12)
    parser.add_argument("--seed", type=int, default=GATE_SEED)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    net = load_imagine_net(args.checkpoint)
    net.eval()

    entries = []
    stats_lines = []
    for prompt in GATE_PROMPTS:
        h, p, water = render_area(net, prompt, args.seed, chunk_span=args.chunk_span, device="cpu")
        entries.append({"caption": prompt, "heightfield": h, "profile_map": p, "water_level": water})
        stats_lines.append(
            f"{prompt!r}: height min={h.min():.1f} max={h.max():.1f} mean={h.mean():.1f} std={h.std():.1f} "
            f"water_level={water:.1f} profile_ids_used={sorted(set(np.unique(p).tolist()))}"
        )
        # individual PNG too
        save_prompt_grid([entries[-1]], os.path.join(args.out_dir, f"{GATE_PROMPTS.index(prompt):02d}.png"))

    save_prompt_grid(entries, os.path.join(args.out_dir, "gate_grid.png"), title=f"Phase 5 gate @ seed {args.seed}")

    # Cross-prompt discriminability: pairwise mean-height and correlation, to quantify "visibly
    # different" beyond eyeballing.
    stats_lines.append("")
    stats_lines.append("Pairwise mean|height_i - height_j| (flattened, same seed/area):")
    n = len(entries)
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.abs(entries[i]["heightfield"] - entries[j]["heightfield"]).mean()
            corr = np.corrcoef(entries[i]["heightfield"].ravel(), entries[j]["heightfield"].ravel())[0, 1]
            stats_lines.append(
                f"  [{i}] {GATE_PROMPTS[i][:30]!r} vs [{j}] {GATE_PROMPTS[j][:30]!r}: "
                f"mean_abs_diff={diff:.2f} corr={corr:.3f}"
            )

    summary_path = os.path.join(args.out_dir, "gate_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(stats_lines) + "\n")
    print("\n".join(stats_lines))
    print(f"\nWrote grid PNG + per-prompt PNGs + summary to {args.out_dir}")


if __name__ == "__main__":
    main()
