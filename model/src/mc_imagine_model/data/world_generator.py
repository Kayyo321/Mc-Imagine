"""
Module for procedural training-data generation ("procedural bootstrap" — PROJECT.md's data
sourcing strategy 1) and (separately, Phase 8, not implemented here) headless Minecraft world
generation for the real-`.mca` data source.

`WorldGenerator` below is the Phase-8 stub (headless-server sourcing) and is intentionally left
untouched — it isn't needed for the procedural bootstrap this phase implements.

`ProceduralWorldSource` is the Phase 4 deliverable: it samples a physically-motivated parameter
vector per macro-region (512x512 blocks, aligned to the `.mca` region unit per docs/model-spec.md
"Runtime Architecture", even though nothing here reads a real `.mca` file), and renders a
heightfield + per-column surface-profile map + per-column biome map for that region using
deterministic fBm / ridged-multifractal / domain-warped noise, implemented directly on top of
numpy (a from-scratch vectorized Perlin-noise kernel — no external noise library), plus a light
`scipy.ndimage.gaussian_filter` smoothing pass on the derived rock-exposure mask.

The noise is **exactly tileable with period `spec_constants.WORLD_PERIOD` (2048 blocks) in both
axes** — every octave's lattice wraps, and every octave's frequency is quantized so its period is
an exact integer number of cells. That number is not arbitrary: it is the largest wavelength in
`model/positional.py`'s `CoordinateEncoder`, whose Fourier features are therefore 2048-periodic
too. Terrain that was *not* 2048-periodic would hand the network identical inputs with different
targets, and MSE answers conflicting targets with their mean — flat terrain. See
`spec_constants.WORLD_PERIOD` for the full argument and the accepted trade-off.

Each region is rendered on a (512 + 2*HALO_MARGIN) canvas so that every one of the 32x32 chunks in
the region — including edge chunks — can be sliced with its full 4-cell halo (see
docs/model-spec.md's coordinate-conditioning design, `model/positional.py`'s `CoordinateEncoder`)
without any extra padding logic at slice time.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from mc_imagine_model.spec_constants import (
    BIOME_DEFAULT_PROFILE,
    NUM_BIOMES,
    NUM_PROFILES,
    TERRAIN_CLIP_MAX,
    TERRAIN_CLIP_MIN,
    WORLD_PERIOD,
    seed_to_offset,
)

REGION_BLOCKS = 512  # 32 chunks x 16 blocks, matches vanilla .mca region granularity
HALO_MARGIN = 4  # R from docs/poc-plan.md §1b
CANVAS = REGION_BLOCKS + 2 * HALO_MARGIN  # 520

# One fixed gradient-table id for the entire project — see docs/phase2-plan.md §1a. Every octave
# offsets deterministically from this, so the noise field is identical everywhere and the only thing
# a world seed changes is *where in that field* you are standing.
CANONICAL_NOISE_SEED = 20260728

# Negative excursion of `shaped` (see render_region) at which the erosion carve reaches full
# strength. This constant is what makes the Phase-2 valley carving actually *do* something.
#
# Measured over 160 sampled regions: `shaped` is strongly right-skewed, because the ridged
# component `1 - |perlin|` has mean +0.50 once remapped to [-1, 1] (measured +0.5017, stable across
# frequency) while the plain fBm component has mean 0. For the ridge-heavy archetypes that are
# supposed to produce canyons, `shaped` therefore spans roughly [-0.22, +0.86] rather than
# [-1, +1]. An unnormalized `clip(-shaped, 0, 1)**2` mask consequently peaked at 0.22**2 = 0.048,
# so the carve term delivered ~3 blocks instead of the ~55 it was written to deliver, and the
# global minimum height over 160 regions was 34.4 — *no better than the Day-1 baseline of 27.7*
# (docs/phase2-plan.md §0). Dividing by the realistic negative excursion lets the mask reach 1.0 at
# the true lows: global min drops to 5.6, 56% of regions dip below sea level and 4% below y=20,
# while peaks are untouched (the mask is zero for positive `shaped`).
#
# It must stay a FIXED constant, never a per-region statistic such as `-shaped.min()`: a
# region-dependent normalization would reintroduce exactly the region-to-region discontinuity that
# docs/phase2-plan.md §1a exists to remove.
VALLEY_MASK_SCALE = 0.25

# Slope (|grad h|, blocks of height per block of horizontal distance) at which `render_region`'s
# rock-exposure term saturates — i.e. "as steep as this generator's terrain ever gets".
#
# This replaces a per-region `slope / (slope.max() + 1e-6)`. That normalizer was a *per-region
# statistic*, and it fails for the same reason `-shaped.min()` would (see VALLEY_MASK_SCALE above):
# two regions whose terrain meets at a shared boundary can have different `slope.max()`, so the
# identical boundary column gets divided by different numbers, crosses the 0.35 stone threshold on
# one side and not the other, and leaves a visible material seam exactly at the region edge. It also
# inverted the intended physics: the steeper a region was overall, the *smaller* every column's
# normalized slope became, so the most mountainous regions were the least likely to expose rock.
#
# Measured over 200 regions (only those with relief_amplitude > 25, the ones the branch runs on),
# pooled across all columns rather than per region: p50 0.57, p90 2.78, p99 5.40, p99.5 6.28,
# p99.9 9.38, pooled max 31.3. 6.0 sits at roughly the 99.4th percentile, so the term reads ~1.0 on
# genuine cliff faces and stays proportional everywhere below that. Paired with an explicit clip to
# [0, 1] so the rare steeper-than-6 column saturates instead of overshooting the score.
#
# Calibration, measured on the same 200 regions: the old per-region normalizer labelled 0.061% of
# columns `stone`, i.e. the rock-exposure override was very nearly inert (dividing by the region's
# own maximum meant only columns steeper than ~58% of that maximum could clear the 0.35 threshold,
# and the sigma-1.5 blur then erased most of those). At a fixed 6.0 the share is 0.885% — a real
# but still-sparse dusting of exposed rock on cliffs and summits, which is what profile id 4 is for
# and what makes it learnable at all. For reference: 4.0 -> 4.9%, 3.0 -> 10.3%, 2.0 -> 18.4%, which
# would repaint whole mountainsides in stone.
SLOPE_NORM_SCALE = 6.0


class WorldGenerator:
    """
    Handles headless Minecraft server execution to generate world data (Phase 8 — real `.mca`
    sourcing). Not implemented in this pass; `ProceduralWorldSource` below is the Phase 4 stand-in.
    """

    def __init__(self) -> None:
        pass

    def generate_world(self, seed: int, output_dir: str) -> str:
        """
        Generates a Minecraft world for a specific seed.

        Args:
            seed (int): The world generation seed.
            output_dir (str): The directory to save the world data in.

        Returns:
            str: Path to the generated world directory.
        """
        # TODO (Phase 8): Implement headless server launch and generation logic
        pass

    def batch_generate(self, seeds: List[int], output_dir: str) -> List[str]:
        """
        Generates multiple Minecraft worlds in batch.

        Args:
            seeds (List[int]): List of world generation seeds.
            output_dir (str): Base directory to save the worlds in.

        Returns:
            List[str]: List of paths to the generated world directories.
        """
        # TODO (Phase 8): Implement batch generation
        pass


@dataclass
class RegionParams:
    """One macro-region's generation parameter vector — the "known ground truth" that
    `data/labeler.py` turns into a caption and `ProceduralWorldSource.render_region` turns into
    actual heightfield/profile/biome rasters. Field names match docs/poc-plan.md's Phase 4 spec
    verbatim."""

    seed: int
    region_x: int
    region_z: int
    archetype: str
    base_height: float
    relief_amplitude: float
    relief_frequency: float
    ridge_sharpness: float
    plateau_quantization: float
    water_level: float
    erosion: float
    surface_profile_id: int
    biome_class: int


# --- Archetypes -----------------------------------------------------------------------------
# Each archetype is a named region of parameter space, deliberately constructed so the sampled
# parameter vectors are *discriminative* per docs/poc-plan.md's Phase 5 gate: the 6 held-out
# benchmark prompts each map onto (or very near) one of these, so the labeler's caption vocabulary
# and the world_generator's actual rendered terrain stay in lockstep. See data/labeler.py for the
# caption templates keyed to the same archetype names.
#
# `relief_frequency` is PINNED per archetype (lo == hi), and that is load-bearing — see
# `sample_params` for the full argument. Briefly: it is the one sampled parameter the caption cannot
# describe and the seed cannot reveal, and averaging a fixed-phase noise field over a range of
# frequencies annihilates its structure, so a randomized frequency makes flat terrain the
# MSE-optimal answer. `tests/test_model.py` asserts every range here stays degenerate.
ARCHETYPES: Dict[str, Dict[str, Any]] = {
    "valley_mountains_craters": dict(
        biome_choices=[0, 2], profile=4, base_profile=0,
        base_height=(84, 118), relief_amplitude=(64, 104), relief_frequency=(4.75, 4.75),
        ridge_sharpness=(0.60, 0.95), plateau_quantization=(0.0, 0.2),
        water_level=(55, 74), erosion=(0.65, 0.95),
    ),
    # Phase 2: the archetype that actually answers "deep valleys going underground". High base with
    # very large amplitude and near-max erosion, so the carving term drives floors far below sea
    # level (63) while the ridges stay tall — the biggest vertical range of any archetype.
    "deep_canyon": dict(
        biome_choices=[8, 9, 5], profile=4, base_profile=3,
        base_height=(96, 130), relief_amplitude=(92, 132), relief_frequency=(3.35, 3.35),
        ridge_sharpness=(0.45, 0.85), plateau_quantization=(0.25, 0.65),
        water_level=(18, 44), erosion=(0.75, 1.0),
    ),
    # Phase 2: drowned glacial valleys — steep walls plunging straight into deep water.
    "fjord_rift": dict(
        biome_choices=[7, 6, 10], profile=4, base_profile=2,
        base_height=(78, 104), relief_amplitude=(74, 116), relief_frequency=(3.9, 3.9),
        ridge_sharpness=(0.65, 1.0), plateau_quantization=(0.0, 0.15),
        water_level=(56, 68), erosion=(0.7, 1.0),
    ),
    "desert_dunes": dict(
        biome_choices=[1], profile=1, base_profile=1,
        base_height=(58, 74), relief_amplitude=(3, 12), relief_frequency=(3.0, 3.0),
        ridge_sharpness=(0.0, 0.15), plateau_quantization=(0.0, 0.25),
        water_level=(-20, 20), erosion=(0.0, 0.15),
    ),
    # Ranges are bounded so `max(base_height) + max(relief_amplitude)` = 240 sits clear of
    # TERRAIN_CLIP_MAX (252). They previously summed to 280, and 0.008% of columns across 7/600
    # regions pinned flat against the clip — the same flat-topped-peak failure mode as Day 1's tanh
    # saturation, just arriving via np.clip instead. A ceiling-pinned plateau teaches the model that
    # "towering peaks" means "a table at y=252"; 12 blocks of structural headroom costs ~20 blocks
    # of peak height and buys a summit that actually has relief on it.
    "snow_peaks": dict(
        biome_choices=[3, 10], profile=4, base_profile=2,
        base_height=(100, 140), relief_amplitude=(72, 100), relief_frequency=(5.25, 5.25),
        ridge_sharpness=(0.70, 1.0), plateau_quantization=(0.0, 0.15),
        water_level=(40, 56), erosion=(0.3, 0.55),
    ),
    "tropical_archipelago": dict(
        biome_choices=[7, 11], profile=1, base_profile=1,
        base_height=(46, 60), relief_amplitude=(20, 38), relief_frequency=(4.5, 4.5),
        ridge_sharpness=(0.0, 0.25), plateau_quantization=(0.0, 0.1),
        water_level=(50, 62), erosion=(0.1, 0.3),
    ),
    # biome_choices are badlands-family as of Phase 2 (was [desert, savanna], which is what made the
    # Day-1 red-mesa world sprout vanilla savanna lakes — see docs/phase2-plan.md §0).
    "mesa_plateaus": dict(
        biome_choices=[8, 9], profile=3, base_profile=3,
        base_height=(72, 104), relief_amplitude=(28, 62), relief_frequency=(3.75, 3.75),
        ridge_sharpness=(0.3, 0.6), plateau_quantization=(0.6, 1.0),
        water_level=(28, 46), erosion=(0.5, 0.8),
    ),
    "rolling_grassland": dict(
        biome_choices=[0], profile=0, base_profile=0,
        base_height=(58, 76), relief_amplitude=(5, 18), relief_frequency=(2.25, 2.25),
        ridge_sharpness=(0.0, 0.2), plateau_quantization=(0.0, 0.1),
        water_level=(44, 60), erosion=(0.1, 0.3),
    ),
    "swamp": dict(
        biome_choices=[4], profile=5, base_profile=5,
        base_height=(58, 70), relief_amplitude=(3, 9), relief_frequency=(2.25, 2.25),
        ridge_sharpness=(0.0, 0.15), plateau_quantization=(0.0, 0.05),
        water_level=(48, 60), erosion=(0.2, 0.4),
    ),
    "taiga_forest": dict(
        biome_choices=[6], profile=7, base_profile=7,
        base_height=(64, 86), relief_amplitude=(14, 36), relief_frequency=(3.0, 3.0),
        ridge_sharpness=(0.2, 0.5), plateau_quantization=(0.0, 0.2),
        water_level=(48, 60), erosion=(0.2, 0.45),
    ),
    "savanna_plains": dict(
        biome_choices=[5], profile=0, base_profile=0,
        base_height=(60, 80), relief_amplitude=(8, 22), relief_frequency=(2.65, 2.65),
        ridge_sharpness=(0.0, 0.25), plateau_quantization=(0.0, 0.15),
        water_level=(40, 55), erosion=(0.15, 0.35),
    ),
}
ARCHETYPE_NAMES = list(ARCHETYPES.keys())
# Slightly upweight the 6 archetypes the Phase 5 gate benchmarks against, without starving the rest.
_GATE_ARCHETYPES = {
    "valley_mountains_craters", "desert_dunes", "snow_peaks",
    "tropical_archipelago", "mesa_plateaus", "rolling_grassland",
}
ARCHETYPE_WEIGHTS = np.array(
    [1.6 if name in _GATE_ARCHETYPES else 1.0 for name in ARCHETYPE_NAMES], dtype=np.float64
)
ARCHETYPE_WEIGHTS /= ARCHETYPE_WEIGHTS.sum()


def _region_rng(seed: int, region_x: int, region_z: int, stream: int = 0) -> np.random.Generator:
    """Deterministic per-region RNG — any macro-region can be resampled independently given
    (seed, region_x, region_z), mirroring the "lazily evaluable at any region coordinate" property
    docs/model-spec.md requires of macro.onnx, even though nothing here ships a macro.onnx.

    `stream` selects an independent RNG sub-stream (see `sample_params`, which deliberately draws
    the archetype/parameter choices and the region's *world seed* from two decorrelated streams —
    so `seed` never leaks information about which archetype/caption a region got, and only ever
    drives the specific noise realization. That decorrelation is what gives `SeedEncoder`
    something real to learn: "same caption, different seed -> different layout, same character"."""
    mix = (int(seed) * 1_000_003) ^ (region_x * 20_113) ^ (region_z * 40_213 + 12345) ^ (stream * 99_991 + 7)
    return np.random.default_rng(np.array([mix & 0xFFFFFFFF, (mix >> 32) & 0xFFFFFFFF], dtype=np.uint64))


def _sample_range(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


class ProceduralWorldSource:
    """
    Synthesizes macro-region terrain directly in numpy, standing in for a headless-server +
    `.mca`-extraction pipeline this phase (Phase 4) doesn't implement. See module docstring.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def sample_params(self, region_x: int, region_z: int) -> RegionParams:
        """Samples one macro-region's parameter vector, deterministic given (seed, region_x,
        region_z). Picks an archetype, jitters within its ranges, and — 25% of the time — blends
        toward a second archetype for extra diversity so the model doesn't just memorize 9
        discrete clusters.

        **`relief_frequency` is pinned per archetype and excluded from the blend.** This is the
        third time this project has shipped flat terrain for the same underlying reason, so the
        argument is written out in full:

        The network only ever sees `(caption, chunk coords, world seed)`. The seed enters
        `render_region` solely as `seed_to_offset(seed)` — a *translation* of one canonical noise
        field — so the phase is recoverable. The parameters are not: they come from `stream=0`
        while the world seed comes from `stream=1`, decorrelated on purpose (see `_region_rng`).
        The caption names an archetype and coarse amplitude/erosion buckets, nothing more.

        Under MSE the network's optimal output is therefore `E[height | caption, coords, seed]`,
        averaged over whatever it cannot determine. Measured, per parameter, as the fraction of
        within-chunk relief surviving that average:

            amplitude only        ~100%      erosion only          ~100%
            ridge + plateau     83-100%      frequency only       20-37%   <-- annihilates it

        Amplitude, erosion, ridge and plateau all *rescale or reshape a fixed pattern*, so the
        average keeps its structure. Frequency changes *which* pattern is drawn, so averaging over
        a 2x frequency range decorrelates the fields being averaged and the mean flattens out.
        With frequency randomized the ceiling on achievable relief is ~25% of target; with it
        pinned, ~85-100%.

        Day 1 hit this with a per-region reseeded gradient table (fixed by the canonical field +
        seed offset, `spec_constants.seed_to_offset`). Phase 2 hit it again through a loss-scale
        imbalance that made the slope term a no-op. This is the same failure's third channel.

        Diversity is unaffected in the way that matters: a different seed still translates the
        field to a genuinely different landscape, and amplitude/erosion/ridge/plateau/base_height
        all still vary freely. Only feature *scale* is now fixed per archetype — which is arguably
        correct, since "mesa" and "snow peaks" describe characteristic feature sizes.
        """
        rng = _region_rng(self.seed, region_x, region_z, stream=0)  # archetype + params only
        seed_rng = _region_rng(self.seed, region_x, region_z, stream=1)  # world seed only
        world_seed = int(seed_rng.integers(1, 2**62 - 1))

        primary = rng.choice(ARCHETYPE_NAMES, p=ARCHETYPE_WEIGHTS)
        spec = ARCHETYPES[primary]

        def draw(spec_: Dict[str, Any]) -> Dict[str, float]:
            return {
                key: _sample_range(rng, *spec_[key])
                for key in (
                    "base_height", "relief_amplitude", "relief_frequency",
                    "ridge_sharpness", "plateau_quantization", "water_level", "erosion",
                )
            }

        values = draw(spec)
        blend_weight = 0.0
        if rng.uniform() < 0.25:
            secondary = rng.choice(ARCHETYPE_NAMES)
            if secondary != primary:
                other = draw(ARCHETYPES[secondary])
                blend_weight = float(rng.uniform(0.15, 0.45))
                for key in values:
                    # `relief_frequency` is deliberately NOT blended — it keeps the primary
                    # archetype's pinned value. See the note below on why it is the one parameter
                    # that must stay a function of the caption alone: blending it here would
                    # reintroduce a caption-invisible frequency for ~25% of regions, which is
                    # exactly the hole this pinning exists to close.
                    if key == "relief_frequency":
                        continue
                    values[key] = (1 - blend_weight) * values[key] + blend_weight * other[key]

        biome_class = int(rng.choice(spec["biome_choices"]))
        profile_id = int(spec["base_profile"])

        return RegionParams(
            seed=world_seed,
            region_x=region_x,
            region_z=region_z,
            archetype=primary,
            surface_profile_id=profile_id,
            biome_class=biome_class,
            **values,
        )

    # --- noise kernels ------------------------------------------------------------------------
    @staticmethod
    def _hash_angle(ix: np.ndarray, iz: np.ndarray, seed: int) -> np.ndarray:
        seed_mixed = int(seed) & 0xFFFFFF  # fold arbitrary-magnitude seeds into 24 bits first —
        # avoids int64 overflow below (world seeds can be up to 2**62, see world_generator's
        # decorrelated seed-stream in sample_params; 24 bits of mixing entropy is plenty here)
        # while still depending on the full seed value.
        h = (ix.astype(np.int64) * 374761393 + iz.astype(np.int64) * 668265263 + seed_mixed * 2246822519)
        h = h & 0xFFFFFFFF
        h = (h ^ (h >> 13)) * 1274126177
        h = h & 0xFFFFFFFF
        return (h.astype(np.float64) / 4294967296.0) * 2.0 * np.pi

    @classmethod
    def _perlin2d(cls, x: np.ndarray, z: np.ndarray, seed: int, lattice_period: int) -> np.ndarray:
        """Classic gradient (Perlin) noise, fully vectorized, evaluated at arbitrary (possibly
        non-integer, possibly domain-warped) coordinate arrays `x`/`z`.

        **Tileable**: the four integer lattice corners are reduced modulo `lattice_period` *before*
        they are hashed, so the gradient table repeats every `lattice_period` cells and the field
        satisfies `noise(x + lattice_period, z) == noise(x, z)` identically. Without this the field
        is aperiodic, and since `model/positional.py`'s `CoordinateEncoder` produces a feature
        vector with period `WORLD_PERIOD` blocks, every alias of a coordinate would carry *different*
        terrain under an *identical* network input — conflicting regression targets whose MSE
        optimum is their mean, i.e. flat terrain. See `spec_constants.WORLD_PERIOD`.

        `np.mod` (not C-style truncating `%`) is required: lattice indices go negative wherever the
        domain warp pushes a sample left of the origin, and a truncating remainder would map -1 and
        +1 to gradients that are not `lattice_period` apart, tearing the field along x<0 / z<0.
        `np.mod` on int64 uses floored division, so `np.mod(-1, 8) == 7`, which is what tiling means.

        Note the *offset* vectors deliberately keep the unwrapped coordinates (`x - x0` etc.): only
        the gradient lookup wraps, the interpolation geometry does not.
        """
        x0 = np.floor(x).astype(np.int64)
        z0 = np.floor(z).astype(np.int64)
        x1, z1 = x0 + 1, z0 + 1
        sx, sz = x - x0, z - z0

        x0w = np.mod(x0, lattice_period)
        x1w = np.mod(x1, lattice_period)
        z0w = np.mod(z0, lattice_period)
        z1w = np.mod(z1, lattice_period)

        def dot(ix: np.ndarray, iz: np.ndarray, dx: np.ndarray, dz: np.ndarray) -> np.ndarray:
            angle = cls._hash_angle(ix, iz, seed)
            return np.cos(angle) * dx + np.sin(angle) * dz

        n00 = dot(x0w, z0w, x - x0, z - z0)
        n10 = dot(x1w, z0w, x - x1, z - z0)
        n01 = dot(x0w, z1w, x - x0, z - z1)
        n11 = dot(x1w, z1w, x - x1, z - z1)

        def fade(t: np.ndarray) -> np.ndarray:
            return t * t * t * (t * (t * 6 - 15) + 10)

        u, v = fade(sx), fade(sz)
        nx0 = n00 + u * (n10 - n00)
        nx1 = n01 + u * (n11 - n01)
        return (nx0 + v * (nx1 - nx0)) * 1.4142  # renormalize roughly to [-1, 1]

    @classmethod
    def _fbm(
        cls, x: np.ndarray, z: np.ndarray, seed: int, base_freq: float,
        octaves: int = 5, persistence: float = 0.5, lacunarity: float = 2.0, ridged: bool = False,
    ) -> np.ndarray:
        """Fractal-Brownian-motion sum of tileable Perlin octaves, every octave sharing the *same*
        world period of exactly `WORLD_PERIOD` blocks (so their sum does too).

        An octave's world period is `lattice_period / freq` blocks, which is only exactly
        `WORLD_PERIOD` when `WORLD_PERIOD * freq` is an integer. The requested `freq` generally is
        not, so each octave's frequency is **quantized** to `n_cells / WORLD_PERIOD` for the nearest
        integer `n_cells` and the quantized value is what actually gets sampled. Rounding
        `n_cells` but sampling at the *requested* frequency would leave the period off by a
        fraction of a cell — which reads as "close enough" and silently reintroduces exactly the
        aliasing the tiling exists to remove, since the mismatch accumulates over 16-32 aliases.
        The cost is that effective frequencies land on a `1/WORLD_PERIOD` grid; at these octave
        counts that is a sub-percent shift in feature wavelength.

        `n_cells` is floored at 2 because a 1-cell period is a constant field (all four corners
        wrap to the same gradient).
        """
        total = np.zeros_like(x, dtype=np.float64)
        amp, freq, norm = 1.0, base_freq, 0.0
        for o in range(octaves):
            n_cells = max(2, int(round(WORLD_PERIOD * freq)))
            # Exact in binary: WORLD_PERIOD is a power of two, so q_freq is an integer scaled by a
            # negative power of two and `integer_coord * q_freq` is computed without rounding.
            q_freq = n_cells / WORLD_PERIOD
            layer = cls._perlin2d(x * q_freq, z * q_freq, seed + o * 1013, n_cells)
            if ridged:
                layer = 1.0 - np.abs(layer)
            total += amp * layer
            norm += amp
            amp *= persistence
            freq *= lacunarity
        return total / norm

    def render_region(self, params: RegionParams) -> Dict[str, Any]:
        """Renders the (CANVAS x CANVAS) heightfield/profile/biome rasters for one macro-region.
        Returns arrays covering the full region plus its HALO_MARGIN border on every side, so
        `data/dataset.py` can slice any of the region's 32x32 chunks (including edge chunks) with
        a full 24x24 halo window with no further padding."""
        # GLOBAL block coordinates (Phase 2 — docs/phase2-plan.md §1a). Day 1 sampled noise at
        # *region-local* coordinates with a *per-region* Perlin seed, which made every macro-region
        # an independent pseudorandom field: unlearnable by a coordinate-conditioned network, and
        # discontinuous at every region boundary. Sampling one canonical field at global coordinates
        # fixes both — terrain now continues smoothly from region to region, so a valley can span
        # hundreds of blocks, and the target becomes a function the network can actually represent.
        local = np.arange(CANVAS, dtype=np.float64) - HALO_MARGIN
        gx = params.region_x * REGION_BLOCKS + local
        gz = params.region_z * REGION_BLOCKS + local
        Z, X = np.meshgrid(gz, gx, indexing="ij")  # [z][x], matches heightmap axis order

        # The world seed translates the canonical field rather than reseeding it.
        off_x, off_z = seed_to_offset(params.seed)
        X = X + off_x
        Z = Z + off_z

        # Fold into the canonical period. The noise below is exactly WORLD_PERIOD-periodic by
        # construction (see `_perlin2d`/`_fbm`), so this is a mathematical no-op — but it is not a
        # numerical one, and that is the point. Global coordinates reach ~35000 (64 regions x 512
        # blocks, plus the seed offset); evaluating the *same* landscape at x and x+2048 through
        # float64 addition, domain warping and floor() would agree only to ~1e-11, which is not the
        # bit-exact identity the training target needs. Reducing the coordinate first makes aliased
        # positions literally the same input, so the periodicity is exact rather than merely close.
        # It also keeps every downstream product small and well away from float64's exact-integer
        # limit. `np.mod`, not `%`-on-negatives semantics: region_x/region_z may be negative.
        X = np.mod(X, WORLD_PERIOD)
        Z = np.mod(Z, WORLD_PERIOD)

        # Canonical, fixed layer ids — deliberately NOT a function of seed/region_x/region_z.
        seed = CANONICAL_NOISE_SEED

        # Domain warp: erosion drives how much the sampling coordinates meander, producing
        # winding valleys rather than perfectly radial noise contours.
        #
        # The warp is itself an fbm, so it is WORLD_PERIOD-periodic on exactly the same terms as the
        # field it warps. DO NOT "optimize" this into a non-periodic offset (a linear ramp, a raw
        # hash, an unwrapped gradient): the composition `f(x + w(x))` is P-periodic *only* because
        # both `f` and `w` are — `f(x+P + w(x+P)) = f(x+P + w(x)) = f(x + w(x))`. A non-periodic
        # warp of a periodic field is not periodic, and would put the whole tiling guarantee back
        # where it was.
        warp_freq = 1.0 / 220.0
        wx = self._fbm(X, Z, seed + 101, warp_freq, octaves=2)
        wz = self._fbm(X, Z, seed + 202, warp_freq, octaves=2)
        warp_amp = 40.0 * params.erosion
        Xw = X + wx * warp_amp
        Zw = Z + wz * warp_amp

        base_freq = params.relief_frequency / REGION_BLOCKS
        raw = self._fbm(Xw, Zw, seed, base_freq, octaves=5, persistence=0.5, lacunarity=2.0)
        ridged = self._fbm(Xw, Zw, seed + 303, base_freq, octaves=5, persistence=0.5, lacunarity=2.0, ridged=True)
        ridged_signed = ridged * 2.0 - 1.0
        shaped = (1.0 - params.ridge_sharpness) * raw + params.ridge_sharpness * ridged_signed
        shaped = np.clip(shaped, -1.3, 1.3)

        # Plateau quantization: stair-step the shape toward discrete terraces (mesa plateaus).
        if params.plateau_quantization > 1e-3:
            levels = max(2.0, 14.0 * (1.0 - params.plateau_quantization) + 2.0)
            quantized = np.round(shaped * levels) / levels
            shaped = (1.0 - params.plateau_quantization) * shaped + params.plateau_quantization * quantized

        heightfield = params.base_height + shaped * params.relief_amplitude
        # Erosion emphasizes valley carving: push already-low areas further down. Phase 2 raises the
        # carving coefficient from 0.35 to 0.60 and squares the mask, so the deepest lows plunge
        # while shallow dips are left alone — Day-1 ground truth never went below y=27.7 anywhere in
        # 200 regions, which is why "deep valleys" was unlearnable regardless of model quality
        # (docs/phase2-plan.md §0). Squaring rather than raising the coefficient further keeps
        # canyon floors varied instead of clipping them flat against TERRAIN_CLIP_MIN.
        # `VALLEY_MASK_SCALE` normalizes by the *realistic* negative excursion of `shaped` (which
        # only reaches about -0.22 for ridge-heavy archetypes, not -1) — without it the squared mask
        # caps at 0.048 and the whole carving term is a measured no-op. See its definition above.
        valley_mask = np.clip(-shaped / VALLEY_MASK_SCALE, 0.0, 1.0) ** 2
        heightfield -= params.erosion * 0.60 * params.relief_amplitude * valley_mask

        heightfield = np.clip(heightfield, TERRAIN_CLIP_MIN, TERRAIN_CLIP_MAX).astype(np.float32)

        # --- profile map: base profile everywhere, stone override on steep/high terrain --------
        profile_map = np.full(heightfield.shape, params.surface_profile_id, dtype=np.uint8)
        if params.relief_amplitude > 25.0:
            # NB: distinct names from the `gx`/`gz` coordinate vectors above — these are the
            # heightfield's partial derivatives, not block coordinates.
            dh_dz, dh_dx = np.gradient(heightfield)
            slope = np.sqrt(dh_dx ** 2 + dh_dz ** 2)
            # Fixed normalizer, never `slope.max()` — see SLOPE_NORM_SCALE for why a per-region
            # statistic here produces a material seam at region boundaries.
            slope_norm = np.clip(slope / SLOPE_NORM_SCALE, 0.0, 1.0)
            peak_thresh = params.base_height + 0.62 * params.relief_amplitude
            stone_score = 0.6 * slope_norm + 0.4 * np.clip(
                (heightfield - peak_thresh) / (params.relief_amplitude + 1e-6), 0.0, 1.0
            )
            stone_score = gaussian_filter(stone_score, sigma=1.5)
            profile_map = np.where(stone_score > 0.35, np.uint8(4), profile_map)

        # --- biome map: base biome everywhere, ocean under water -------------------------------
        biome_map = np.full(heightfield.shape, params.biome_class, dtype=np.uint8)
        underwater = heightfield < (params.water_level - 1.0)
        biome_map = np.where(underwater, np.uint8(7), biome_map)

        assert profile_map.max() < NUM_PROFILES
        assert biome_map.max() < NUM_BIOMES

        return {
            "heightfield": heightfield,
            "profile_map": profile_map,
            "biome_map": biome_map,
            "water_level": np.float32(params.water_level),
            "params": asdict(params),
        }

    def generate_shards(self, num_regions: int, out_dir: str, region_layout_width: int = 64) -> List[str]:
        """Samples `num_regions` macro-regions and writes one `.npz` shard each to `out_dir`.
        Region (x, z) coordinates are laid out on a grid purely for bookkeeping/filenames. Since
        `sample_params` draws an independent *world seed* per region (and the world seed translates
        the canonical noise field, per docs/phase2-plan.md §1a), grid-adjacent shards are not
        continuations of each other — each is a differently-translated, differently-parameterized
        window onto the one shared field. Continuity holds between any two regions that share a
        world seed and parameter vector, which is the property the trained model has to reproduce
        at inference time (and which `render_region`'s global-coordinate sampling guarantees).

        Because the field is WORLD_PERIOD-periodic, every shard — wherever it sits on the layout
        grid — samples the same 2048x2048 canonical domain. 600 regions of 512x512 therefore cover
        that domain ~37x over, and 8000 regions ~500x. The alternative (a 65536-block offset window)
        spread the same shards over a 1024x larger domain at ~3% coverage, which is a memorization
        problem rather than a regression problem."""
        import os

        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for region_id in range(num_regions):
            rx = region_id % region_layout_width
            rz = region_id // region_layout_width
            params = self.sample_params(rx, rz)
            region = self.render_region(params)
            path = os.path.join(out_dir, f"region_{region_id:05d}.npz")
            np.savez_compressed(
                path,
                heightfield=region["heightfield"],
                profile_map=region["profile_map"],
                biome_map=region["biome_map"],
                water_level=region["water_level"],
                **{f"param_{k}": v for k, v in region["params"].items()},
            )
            paths.append(path)
        return paths
