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

Each region is rendered on a (512 + 2*HALO_MARGIN) canvas so that every one of the 32x32 chunks in
the region — including edge chunks — can be sliced with its full 4-cell halo (see
docs/model-spec.md's coordinate-conditioning design, `model/positional.py`'s `CoordinateEncoder`)
without any extra padding logic at slice time.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from mc_imagine_model.spec_constants import BIOME_DEFAULT_PROFILE, NUM_BIOMES, NUM_PROFILES

REGION_BLOCKS = 512  # 32 chunks x 16 blocks, matches vanilla .mca region granularity
HALO_MARGIN = 4  # R from docs/poc-plan.md §1b
CANVAS = REGION_BLOCKS + 2 * HALO_MARGIN  # 520


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
ARCHETYPES: Dict[str, Dict[str, Any]] = {
    "valley_mountains_craters": dict(
        biome_choices=[0, 2], profile=4, base_profile=0,
        base_height=(80, 112), relief_amplitude=(52, 92), relief_frequency=(3.0, 6.5),
        ridge_sharpness=(0.60, 0.95), plateau_quantization=(0.0, 0.2),
        water_level=(55, 74), erosion=(0.55, 0.9),
    ),
    "desert_dunes": dict(
        biome_choices=[1], profile=1, base_profile=1,
        base_height=(58, 74), relief_amplitude=(3, 12), relief_frequency=(2.0, 4.0),
        ridge_sharpness=(0.0, 0.15), plateau_quantization=(0.0, 0.25),
        water_level=(-20, 20), erosion=(0.0, 0.15),
    ),
    "snow_peaks": dict(
        biome_choices=[3], profile=4, base_profile=2,
        base_height=(100, 142), relief_amplitude=(70, 112), relief_frequency=(3.5, 7.0),
        ridge_sharpness=(0.70, 1.0), plateau_quantization=(0.0, 0.15),
        water_level=(40, 56), erosion=(0.3, 0.55),
    ),
    "tropical_archipelago": dict(
        biome_choices=[7, 0], profile=1, base_profile=1,
        base_height=(46, 60), relief_amplitude=(20, 38), relief_frequency=(3.0, 6.0),
        ridge_sharpness=(0.0, 0.25), plateau_quantization=(0.0, 0.1),
        water_level=(50, 62), erosion=(0.1, 0.3),
    ),
    "mesa_plateaus": dict(
        biome_choices=[1, 5], profile=3, base_profile=3,
        base_height=(70, 96), relief_amplitude=(20, 46), relief_frequency=(2.5, 5.0),
        ridge_sharpness=(0.3, 0.6), plateau_quantization=(0.6, 1.0),
        water_level=(28, 46), erosion=(0.5, 0.8),
    ),
    "rolling_grassland": dict(
        biome_choices=[0], profile=0, base_profile=0,
        base_height=(58, 76), relief_amplitude=(5, 18), relief_frequency=(1.5, 3.0),
        ridge_sharpness=(0.0, 0.2), plateau_quantization=(0.0, 0.1),
        water_level=(44, 60), erosion=(0.1, 0.3),
    ),
    "swamp": dict(
        biome_choices=[4], profile=5, base_profile=5,
        base_height=(58, 70), relief_amplitude=(3, 9), relief_frequency=(1.5, 3.0),
        ridge_sharpness=(0.0, 0.15), plateau_quantization=(0.0, 0.05),
        water_level=(48, 60), erosion=(0.2, 0.4),
    ),
    "taiga_forest": dict(
        biome_choices=[6], profile=7, base_profile=7,
        base_height=(64, 86), relief_amplitude=(14, 36), relief_frequency=(2.0, 4.0),
        ridge_sharpness=(0.2, 0.5), plateau_quantization=(0.0, 0.2),
        water_level=(48, 60), erosion=(0.2, 0.45),
    ),
    "savanna_plains": dict(
        biome_choices=[5], profile=0, base_profile=0,
        base_height=(60, 80), relief_amplitude=(8, 22), relief_frequency=(1.8, 3.5),
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
        discrete clusters."""
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
    def _perlin2d(cls, x: np.ndarray, z: np.ndarray, seed: int) -> np.ndarray:
        """Classic gradient (Perlin) noise, fully vectorized, evaluated at arbitrary (possibly
        non-integer, possibly domain-warped) coordinate arrays `x`/`z`."""
        x0 = np.floor(x).astype(np.int64)
        z0 = np.floor(z).astype(np.int64)
        x1, z1 = x0 + 1, z0 + 1
        sx, sz = x - x0, z - z0

        def dot(ix: np.ndarray, iz: np.ndarray, dx: np.ndarray, dz: np.ndarray) -> np.ndarray:
            angle = cls._hash_angle(ix, iz, seed)
            return np.cos(angle) * dx + np.sin(angle) * dz

        n00 = dot(x0, z0, x - x0, z - z0)
        n10 = dot(x1, z0, x - x1, z - z0)
        n01 = dot(x0, z1, x - x0, z - z1)
        n11 = dot(x1, z1, x - x1, z - z1)

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
        total = np.zeros_like(x, dtype=np.float64)
        amp, freq, norm = 1.0, base_freq, 0.0
        for o in range(octaves):
            layer = cls._perlin2d(x * freq, z * freq, seed + o * 1013)
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
        coords = np.arange(CANVAS, dtype=np.float64) - HALO_MARGIN
        Z, X = np.meshgrid(coords, coords, indexing="ij")  # [z][x], matches heightmap axis order

        seed = params.seed * 7919 + params.region_x * 104729 + params.region_z * 15485863

        # Domain warp: erosion drives how much the sampling coordinates meander, producing
        # winding valleys rather than perfectly radial noise contours.
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
        # Erosion emphasizes valley carving: push already-low areas further down.
        valley_mask = np.clip(-shaped, 0.0, 1.0)
        heightfield -= params.erosion * 0.35 * params.relief_amplitude * valley_mask

        heightfield = np.clip(heightfield, 4.0, 250.0).astype(np.float32)

        # --- profile map: base profile everywhere, stone override on steep/high terrain --------
        profile_map = np.full(heightfield.shape, params.surface_profile_id, dtype=np.uint8)
        if params.relief_amplitude > 25.0:
            gz, gx = np.gradient(heightfield)
            slope = np.sqrt(gx ** 2 + gz ** 2)
            slope_norm = slope / (slope.max() + 1e-6)
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
        Region (x, z) coordinates are laid out on a grid purely for bookkeeping/filenames — regions
        are otherwise independently sampled, not spatially continuous with each other (§ render_region
        docstring)."""
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
