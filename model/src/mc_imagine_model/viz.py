"""
Small rendering helpers shared by the Phase 4 data-sanity dump, the Phase 5 per-epoch qualitative
dump, and the Phase 5 gate's 6-held-out-prompt comparison. Kept dependency-light (matplotlib only,
already in requirements.txt) rather than pulling in a heavier plotting stack.
"""

from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mc_imagine_model.spec_constants import BIOME_PALETTE, NUM_BIOMES, NUM_PROFILES, PROFILE_TABLE

# Fixed, distinct colors per profile / biome class id, for consistent visual comparison across
# figures (shading also modulated by relative height so terrain shape reads clearly).
_PROFILE_COLORS = np.array([
    [0.42, 0.62, 0.29],  # 0 grass
    [0.86, 0.78, 0.52],  # 1 sand
    [0.95, 0.97, 0.99],  # 2 snow
    [0.72, 0.33, 0.20],  # 3 mesa (red)
    [0.55, 0.55, 0.58],  # 4 stone
    [0.36, 0.40, 0.24],  # 5 mud (olive-brown swamp muck)
    [0.62, 0.60, 0.55],  # 6 gravel
    [0.48, 0.34, 0.20],  # 7 podzol (warm reddish-brown forest floor)
])
_BIOME_COLORS = np.array([
    [0.55, 0.75, 0.35],  # plains
    [0.90, 0.80, 0.45],  # desert
    [0.20, 0.45, 0.20],  # forest
    [0.90, 0.95, 1.00],  # snowy_plains
    [0.30, 0.35, 0.20],  # swamp
    [0.80, 0.70, 0.35],  # savanna
    [0.15, 0.35, 0.30],  # taiga
    [0.10, 0.30, 0.55],  # ocean
])
_WATER_COLOR = np.array([0.15, 0.35, 0.75])


def render_terrain_rgb(
    heightfield: np.ndarray,
    profile_map: np.ndarray,
    water_level: float,
    height_range: Optional[tuple] = None,
) -> np.ndarray:
    """Renders a shaded-relief RGB image (float in [0,1], HxWx3) from a heightfield + per-column
    profile map, with a flat water color under `water_level`. Shading = simple slope-based hillshade
    blended with the profile's base color, so terrain shape (valleys vs. peaks vs. flat) is visually
    legible even without a real lighting model."""
    h = heightfield.astype(np.float64)
    if height_range is None:
        lo, hi = float(h.min()), float(h.max())
    else:
        lo, hi = height_range
    span = max(hi - lo, 1e-6)
    norm_h = np.clip((h - lo) / span, 0.0, 1.0)

    gz, gx = np.gradient(h)
    slope = np.sqrt(gx ** 2 + gz ** 2)
    shade = 1.0 - np.clip(slope / (slope.max() + 1e-6), 0.0, 1.0) * 0.55
    shade = 0.55 + 0.45 * shade  # keep a floor so nothing goes fully black

    profile_ids = np.clip(profile_map.astype(np.int64), 0, NUM_PROFILES - 1)
    base_color = _PROFILE_COLORS[profile_ids]
    # brighten slightly with height so peaks read lighter than valleys, in addition to the profile
    height_tint = 0.85 + 0.3 * norm_h[..., None]
    rgb = np.clip(base_color * shade[..., None] * height_tint, 0.0, 1.0)

    underwater = h < water_level
    rgb = np.where(underwater[..., None], _WATER_COLOR[None, None, :] * (0.7 + 0.3 * shade[..., None]), rgb)
    return rgb


def save_prompt_grid(
    entries,
    out_path: str,
    title: Optional[str] = None,
) -> None:
    """
    entries: list of dicts with keys "caption", "heightfield" (HxW), "profile_map" (HxW),
    "water_level" (float). Renders a grid, one panel per entry, each panel a shaded-relief render
    with its caption as the subplot title (wrapped).
    """
    n = len(entries)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).reshape(rows, cols)
    for i, entry in enumerate(entries):
        ax = axes[i // cols, i % cols]
        rgb = render_terrain_rgb(entry["heightfield"], entry["profile_map"], entry["water_level"])
        ax.imshow(rgb, origin="upper")
        caption = entry.get("caption", "")
        ax.set_title(caption, fontsize=9, wrap=True)
        ax.set_xticks([])
        ax.set_yticks([])
    for i in range(n, rows * cols):
        axes[i // cols, i % cols].axis("off")
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
