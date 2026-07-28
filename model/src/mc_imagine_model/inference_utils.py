"""
Shared helper for running a trained ImagineNet over a tiled area of chunks and assembling the
per-chunk outputs into one contiguous heightfield/profile grid — used by `training/train.py`'s
per-epoch qualitative PNG dump and by the Phase 5 gate script that renders the 6 held-out prompts.

Rendering more than one chunk here is purely for a legible PNG (a single 16x16 chunk is a tiny
image); it also happens to be a convenient way to eyeball docs/poc-plan.md §1b's border-continuity
property before ever touching ONNX export.

Note that every chunk in a rendered tile shares one `seed`, which is what makes the assembled image
meaningful: since Phase 2 the seed selects a *translation of the canonical noise field*
(docs/phase2-plan.md §1a, applied in `model/positional.py`'s CoordinateEncoder), so mixing seeds
across the tile would stitch together fragments of different worlds. Rendering the same prompt at
two seeds should produce two different-but-similar-in-character heightfields; rendering it twice at
one seed must produce bit-identical output.
"""

from typing import Tuple

import numpy as np
import torch

from mc_imagine_model.tokenizer_utils import tokenize


@torch.no_grad()
def render_area(
    model: torch.nn.Module,
    prompt: str,
    seed: int,
    chunk_span: int = 8,
    origin_chunk: Tuple[int, int] = (0, 0),
    device: str = "cpu",
    max_tokens: int = 128,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Runs `model` over a `chunk_span x chunk_span` tile of chunks (all sharing one prompt/seed, batched
    into a single forward call) and assembles the results into one contiguous
    `(chunk_span*16, chunk_span*16)` heightfield + profile-id grid.

    Returns:
        (heightfield, profile_id_map, mean_water_level)
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
    heightmap = out["heightmap"].detach().to("cpu").float().numpy()  # [B,16,16]
    profile_logits = out["profile_logits"].detach().to("cpu").float().numpy()  # [B,P,16,16]
    profile_id = profile_logits.argmax(axis=1)  # [B,16,16]
    water_level = out["water_level"].detach().to("cpu").float().numpy()

    full_h = np.zeros((chunk_span * 16, chunk_span * 16), dtype=np.float32)
    full_p = np.zeros((chunk_span * 16, chunk_span * 16), dtype=np.uint8)
    for idx in range(b):
        row = idx // chunk_span
        col = idx % chunk_span
        full_h[row * 16:(row + 1) * 16, col * 16:(col + 1) * 16] = heightmap[idx]
        full_p[row * 16:(row + 1) * 16, col * 16:(col + 1) * 16] = profile_id[idx]

    return full_h, full_p, float(water_level.mean())
