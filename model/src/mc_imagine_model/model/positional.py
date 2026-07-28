"""
Coordinate/positional encoding module.

`CoordinateEncoder` is the concrete mechanism behind docs/poc-plan.md §1b's seamless-chunk-border
guarantee: it is a **pure, deterministic function of (global block coordinates, world seed)** (no
learned parameters at all), evaluated over a `(16+2R)x(16+2R)` grid built from `chunk_x`/`chunk_z`.
Because two neighboring chunks' overlapping/adjacent global coordinates always produce bit-identical
Fourier features regardless of which chunk "owns" them (at a fixed seed), and the downstream conv
stack (`model/imagine_net.py`) uses valid-padding only (no zero-padding), the network's output for a
border column is a pure function of that column's global coordinate — never a function of which
16x16 tile it happened to be requested as part of. That is what makes chunk borders seamless by
construction rather than by hoping a shared macro-region graph patches things up.

Phase 2 (docs/phase2-plan.md §1a) added the *seed offset* to that pure function. The data generator
no longer reseeds its Perlin gradient table per macro-region — it samples one canonical noise field
at `global_coord + seed_to_offset(world_seed)`. The offset therefore has to be applied to the
coordinates *before* Fourier encoding here too, otherwise the network would have to invert a hash to
recover which translation of the field it is being asked for (exactly the unlearnable situation §0
diagnosed). `seed_to_offset_torch` below is the torch mirror of
`spec_constants.seed_to_offset`, and MUST stay bit-identical to it: it is what ties the exported
ONNX graph to the coordinate frame the training targets were rendered in.

`SeedEncoder` maps the world seed to a fixed sinusoidal feature bank (again fully deterministic,
no learned frequencies/phases) followed by a single learned `Linear(32 -> 64)` projection. Unlike
`PromptEncoder`, this module is *not* frozen — the projection is trained — but the sinusoidal
features it's built from are fixed and genuinely vary with `seed`, so a collapsed/zeroed linear
projection is a training failure, not a structural inevitability. See `verify_seed_encoder_varies`
at the bottom (also exercised by `tests/test_model.py`) for the explicit check
docs/poc-plan.md calls for.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn

from mc_imagine_model.spec_constants import (
    OFFSET_RANGE,
    SEED_FOLD,
    SEED_HASH_MOD,
    SEED_MIX,
)

# Wavelengths lambda in {2048 ... 8} halving each octave -> 9 octaves -> 4 features/octave
# (sin_x, cos_x, sin_z, cos_z) = 36 channels total, per docs/poc-plan.md Phase 5.
DEFAULT_WAVELENGTHS = (2048.0, 1024.0, 512.0, 256.0, 128.0, 64.0, 32.0, 16.0, 8.0)


def seed_to_offset_torch(seed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Torch mirror of `spec_constants.seed_to_offset`, batched and ONNX-exportable.

    Args:
        seed (torch.Tensor): [B] world seeds (any integer dtype; cast to int64 internally).

    Returns:
        tuple: `(offset_x, offset_z)`, each [B] float32, each an exact integer in
        [0, `spec_constants.OFFSET_RANGE`) — currently [0, 2048), which is `WORLD_PERIOD` and hence
        exactly one period of this encoder's Fourier features. Offsets beyond one period alias onto
        positions the encoder cannot distinguish, so a wider range would add no variety while
        creating conflicting training targets (see spec_constants' WORLD_PERIOD note).

    This MUST agree bit-for-bit with the Python reference for every int64 seed, because
    `data/world_generator.py` uses the Python version to decide *where in the canonical noise field*
    a world's training targets were rendered. Any disagreement here would silently train the network
    against a translated copy of its own input coordinates — the worst kind of bug, since nothing
    would crash and the loss would merely fail to go down as far as it should.

    Three implementation details are load-bearing and deliberately not "simplified":

    1. **Everything stays int64 until the very last cast.** The largest intermediate is
       `(SEED_FOLD - 1) * SEED_MIX` ~= 2.66e15, comfortably inside int64's 9.22e18 and — critically —
       *outside* float32's 2**24 exact-integer range. Doing any step of this in float32 rounds the
       product and produces a different, plausible-looking offset. The final results are
       < `OFFSET_RANGE` (2048), far inside 2**24, so the float32 cast at the end is exact.
    2. **`torch.remainder`, never `torch.fmod`.** `remainder` follows Python/numpy sign semantics
       (result takes the divisor's sign); `fmod` follows C (result takes the dividend's sign). World
       seeds are routinely negative, and `-7 % 5` is `3` in Python but `-2` in C — so `fmod` would
       diverge from the reference on exactly the seeds users type in.
    3. **`torch.div(..., rounding_mode="floor")`** for the `//`, matching Python's floor division.
       (`h` is non-negative here so truncation would coincide, but spelling it out keeps the mirror
       obviously correct rather than incidentally correct.)
    """
    s64 = seed.to(torch.int64)
    s = torch.remainder(s64, SEED_FOLD)
    h = torch.remainder(s * SEED_MIX, SEED_HASH_MOD)
    ox = torch.remainder(h, OFFSET_RANGE)
    oz = torch.remainder(torch.div(h, OFFSET_RANGE, rounding_mode="floor"), OFFSET_RANGE)
    return ox.to(torch.float32), oz.to(torch.float32)


class CoordinateEncoder(nn.Module):
    """
    Encodes a `(16+2*halo) x (16+2*halo)` grid of **global block** `(X, Z)` coordinates — built from
    `chunk_x`/`chunk_z` (chunk coordinates, each block-scale = chunk coordinate * 16) and translated
    by the per-world seed offset (`seed_to_offset_torch`) — into multi-octave Fourier features.
    Output channel order is `[sin_x, cos_x, sin_z, cos_z]` repeated once per wavelength octave
    (36 channels for the default 9 octaves).

    Still parameter-free and still a pure function of its inputs; the seed simply widened those
    inputs from "global coordinate" to "global coordinate, world seed". Seamlessness is unaffected:
    at a fixed seed, two chunks asking about the same global coordinate get bit-identical features,
    which is the property `export/export_onnx.py:verify_coordinate_purity` asserts exactly.
    """

    def __init__(self, halo: int = 4, patch: int = 16, wavelengths=DEFAULT_WAVELENGTHS) -> None:
        super().__init__()
        self.halo = halo
        self.patch = patch
        self.size = patch + 2 * halo  # 28 for halo=6, patch=16 (Phase 2); 24 for halo=4 (Day 1)
        freqs = torch.tensor([2.0 * math.pi / w for w in wavelengths], dtype=torch.float32)
        self.register_buffer("freqs", freqs, persistent=False)  # [num_octaves]
        # Local block offsets within/around the chunk: -halo .. patch+halo-1 (28 values for halo=6).
        offsets = torch.arange(-halo, patch + halo, dtype=torch.float32)
        self.register_buffer("offsets", offsets, persistent=False)  # [size]
        self.out_channels = 4 * len(wavelengths)  # 36

    def _fourier(self, global_x: torch.Tensor, global_z: torch.Tensor) -> torch.Tensor:
        """Multi-octave Fourier features of already-offset global coordinates.

        Split out from `forward` so the coordinate frame (chunk -> global -> seed-translated) and
        the encoding itself can be tested independently: a test can hand this raw global
        coordinates and check that `forward` agrees for the coordinates it claims to be encoding.

        Args:
            global_x/global_z: broadcastable tensors of global block coordinates, `[B, Z, X]`.

        Returns:
            torch.Tensor: `[B, 4*num_octaves, Z, X]`, channels ordered
                `[sin_x, cos_x, sin_z, cos_z]` per octave.
        """
        feats = []
        num_octaves = self.freqs.shape[0]
        for i in range(num_octaves):
            f = self.freqs[i]
            feats.append(torch.sin(global_x * f))
            feats.append(torch.cos(global_x * f))
            feats.append(torch.sin(global_z * f))
            feats.append(torch.cos(global_z * f))
        return torch.stack(feats, dim=1)  # [B, 4*num_octaves, Z, X]

    def forward(self, chunk_x: torch.Tensor, chunk_z: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            chunk_x (torch.Tensor): [B] chunk X coordinates (int or float).
            chunk_z (torch.Tensor): [B] chunk Z coordinates.
            seed (torch.Tensor): [B] world seeds (int64) — translates the canonical field per
                docs/phase2-plan.md §1a.

        Returns:
            torch.Tensor: [B, out_channels, size, size] Fourier feature grid, `[z][x]` spatial
                axis order (dim -2 is Z, dim -1 is X) matching docs/model-spec.md's pinned
                `heightmap`/`block_volume` axis order.
        """
        b = chunk_x.shape[0]
        # The int64 hash is evaluated on CPU for the same reason SeedEncoder evaluates its float64
        # phase on CPU: MPS's int64 support is partial, and this is B scalars of work, so keeping it
        # off the accelerator costs nothing. Device moves are traced away by the ONNX exporter, so
        # the exported graph still computes the offset in-graph.
        off_x, off_z = seed_to_offset_torch(seed.detach().to("cpu"))
        off_x = off_x.to(self.freqs.device).view(b, 1, 1)
        off_z = off_z.to(self.freqs.device).view(b, 1, 1)

        gx0 = (chunk_x.to(torch.float32) * self.patch).view(b, 1, 1) + off_x
        gz0 = (chunk_z.to(torch.float32) * self.patch).view(b, 1, 1) + off_z

        x_row = self.offsets.view(1, 1, self.size)  # varies along last dim (X)
        z_col = self.offsets.view(1, self.size, 1)  # varies along middle dim (Z)
        global_x = (gx0 + x_row).expand(b, self.size, self.size)  # [B, size(z), size(x)]
        global_z = (gz0 + z_col).expand(b, self.size, self.size)
        return self._fourier(global_x, global_z)


class SeedEncoder(nn.Module):
    """
    Encodes a world seed into a latent noise vector: fixed sinusoidal features
    `sin(seed * a_i + b_i)` over deterministic (non-learned) constants `a`/`b`, followed by a
    trained `Linear(num_freqs -> embed_dim)`.

    Numerical note: Minecraft world seeds are arbitrary int64 values (up to ~9.2e18 in magnitude).
    float32 only represents integers exactly up to 2**24, so the seed*a_i multiply is done in
    float64 (exact up to 2**53) before casting down to float32 for the linear layer — this keeps
    the sinusoidal phase meaningfully distinct across the seed magnitudes this PoC actually
    exercises (small-to-moderate user-entered seeds, and the ~2**62-magnitude synthetic training
    seeds `data/world_generator.py` draws), even though it does not guarantee zero collisions
    across the *entire* 64-bit range (seeds differing only in their lowest ~10 bits, both near
    2**63, could alias) — an acceptable, explicitly-noted PoC-level tradeoff rather than a silent one.
    """

    def __init__(self, num_freqs: int = 32, embed_dim: int = 64) -> None:
        super().__init__()
        i = torch.arange(num_freqs, dtype=torch.float64)
        # Geometrically spaced frequencies (transformer-positional-encoding style) plus a
        # golden-angle phase offset per index for good spectral spread — fixed, not learned.
        # Deliberately NOT registered as buffers: kept as plain float64 CPU tensors so
        # `module.to(device)` (e.g. "mps") never touches them — MPS has no float64 tensor support
        # at all (even just holding one on-device raises), and this computation is tiny (B x
        # num_freqs scalars) so doing it on CPU is free regardless of what device the rest of the
        # model runs on. See forward()'s explicit CPU roundtrip.
        self._a_cpu = 1.0 / (10000.0 ** (i / num_freqs)) * 1.0e-3
        self._b_cpu = (i * 2.399963229728653) % (2.0 * math.pi)
        self.linear = nn.Linear(num_freqs, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, seed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seed (torch.Tensor): [B] world seeds (int64).

        Returns:
            torch.Tensor: [B, embed_dim] seed noise vectors.
        """
        seed_f64 = seed.detach().to("cpu").to(torch.float64).unsqueeze(-1)  # [B, 1]
        phase = seed_f64 * self._a_cpu.unsqueeze(0) + self._b_cpu.unsqueeze(0)  # [B, num_freqs]
        features = torch.sin(phase).to(torch.float32)
        # Move back to wherever the (learned, device-resident) linear layer's weights live.
        features = features.to(self.linear.weight.device)
        return self.linear(features)


def verify_seed_encoder_varies(embed_dim: int = 64, atol: float = 1e-5) -> bool:
    """Explicit runtime check that two different seeds produce different SeedEncoder outputs for
    the same (irrelevant, since SeedEncoder doesn't take coordinates) module state — the property
    docs/poc-plan.md's Phase 5 section requires be verified with a unit test. Returns True if the
    property holds; raises AssertionError otherwise. Mirrored by tests/test_model.py."""
    enc = SeedEncoder(embed_dim=embed_dim)
    enc.eval()
    seeds = torch.tensor([12345, 67890, -8756019554883095434, 1, 2**61 + 17], dtype=torch.int64)
    with torch.no_grad():
        out = enc(seeds)
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            diff = (out[i] - out[j]).abs().max().item()
            assert diff > atol, (
                f"SeedEncoder produced near-identical output for seeds {seeds[i].item()} and "
                f"{seeds[j].item()} (max abs diff {diff}) — seed dependency must be non-ignorable."
            )
    return True


if __name__ == "__main__":
    verify_seed_encoder_varies()
    print("SeedEncoder varies meaningfully across distinct seeds — OK")
