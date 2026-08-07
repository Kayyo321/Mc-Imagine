"""
Occupancy3DHead capacity test (Mc-Imagine Phase 4.2 — docs/phase4.2-plan.md §2.1, §3.2).

`docs/phase4.2-plan.md` §1: Gate 2 failed because the model found the exact optimum of the loss it
was given (a heightfield), not because it couldn't represent an overhang. §2.1 proved that by
overfitting `Occupancy3DHead` against 8 real overhang-rich chunks with random frozen features and no
regularization: it reaches 100% cell accuracy and reproduces the multi-run column rate exactly in a
few hundred steps. This test pins that result so that if a future architecture change breaks it, the
regression is caught here rather than three gates downstream, where "the run found a heightfield
again" would look identical to a loss-function regression.

Bounded to <=400 steps / CPU / <60s per §3.2 so it runs in the normal `pytest tests/ -q` suite, not
just as an operator-run diagnostic (that's `loss_mass.py`'s job, for the loss side of this).

Requires real shards under `model/data/` (gitignored, not part of a fresh checkout) with at least
one overhang-rich chunk; skips with a clear reason if that data isn't present rather than failing.
"""

import glob
import os

import numpy as np
import pytest
import torch

from mc_imagine_model.model.heads import Occupancy3DHead

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MAX_STEPS = 400
LOG_EVERY = 100


def _load_overhang_rich_chunks(data_dir: str, num_chunks: int = 8, num_regions: int = 40):
    """Same windowing/selection as the §2.1 diagnosis: 16x16 chunks with >25% multi-run columns."""
    paths = sorted(glob.glob(os.path.join(data_dir, "region_*.npz")))[:num_regions]
    chunks = []
    for p in paths:
        with np.load(p) as z:
            b = z["band"]
            c = b.shape[0]
            for zz in range(0, c - 15, 16):
                for xx in range(0, c - 15, 16):
                    chunk = np.transpose(b[zz:zz + 16, xx:xx + 16, :], (2, 0, 1))
                    dy = np.abs(np.diff(chunk.astype(np.int8), axis=0)).sum(axis=0)
                    if (dy >= 2).mean() > 0.25:
                        chunks.append(chunk)
        if len(chunks) >= num_chunks:
            break
    return chunks[:num_chunks]


_chunks = _load_overhang_rich_chunks(DATA_DIR) if os.path.isdir(DATA_DIR) else []


@pytest.mark.skipif(
    not os.path.isdir(DATA_DIR) or len(_chunks) == 0,
    reason=f"no overhang-rich real shards found under {DATA_DIR} (gitignored dataset; "
           f"generate it with generate_data.py before running this test)",
)
def test_occupancy_3d_head_overfits_real_overhang_rich_chunks() -> None:
    """`Occupancy3DHead` must be able to drive real overhang-bearing bands to ~0 loss.

    If this regresses, the architecture — not just the objective §3.1/§3.2 fix — is the
    constraint, and nothing downstream (Stage 3's reweighted losses, Stage 5's confirmatory run) is
    worth debugging until this is green again.
    """
    torch.manual_seed(0)
    band = torch.from_numpy(np.stack(_chunks)).float()
    dy = np.abs(np.diff(band.numpy().astype(np.int8), axis=1)).sum(axis=1)
    target_multirun_pct = (dy >= 2).mean() * 100

    head = Occupancy3DHead(in_channels=192)
    feats = torch.randn(len(_chunks), 192, 22, 22) * 0.5
    feats.requires_grad_(False)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3)

    for step in range(MAX_STEPS):
        logits = head(feats)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, band)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = (torch.sigmoid(logits) > 0.5).float()
        pred_dy = (pred[:, 1:] - pred[:, :-1]).abs().sum(dim=1)
        acc = (pred == band).float().mean().item() * 100
        pred_multirun_pct = (pred_dy >= 2).float().mean().item() * 100

    assert acc >= 99.0, (
        f"Occupancy3DHead overfit only to {acc:.2f}% cell accuracy after {MAX_STEPS} steps on "
        f"{len(_chunks)} real overhang-rich chunks (target multi-run {target_multirun_pct:.1f}%); "
        f"docs/phase4.2-plan.md §2.1 measured 100% — the architecture may have regressed."
    )
    assert abs(pred_multirun_pct - target_multirun_pct) <= 5.0, (
        f"predicted multi-run columns {pred_multirun_pct:.1f}% vs target {target_multirun_pct:.1f}% "
        f"(>5pt gap) after {MAX_STEPS} steps — the head fit cell accuracy without reproducing the "
        f"structure §2.1 requires."
    )
