"""
Small rendering helpers shared by the Phase 4 data-sanity dump, the Phase 5 per-epoch qualitative
dump, and the Phase 5 gate's 6-held-out-prompt comparison. Kept dependency-light (matplotlib only,
already in requirements.txt) rather than pulling in a heavier plotting stack.
"""

from typing import Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from mc_imagine_model.spec_constants import (
    BAND_BOTTOM_OFFSET,
    BAND_HEIGHT,
    BAND_TOP_OFFSET,
    BIOME_PALETTE,
    NUM_BIOMES,
    NUM_PROFILES,
    PROFILE_TABLE,
)

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

# --- cross-section palette (Phase 4 — docs/phase4-plan.md §2.4) ----------------------------------
# The cross-section's whole job is that a reader can tell an overhang from a steep slope at a
# glance, so these four are chosen to be separable by hue AND by lightness (a steep-slope render and
# an overhang render differ by one color; if that color is a near-neighbour of rock the picture is
# worthless). Sky is pale and desaturated, rock is mid grey, water is dark saturated blue, and
# SHELTERED AIR — air with solid above it, i.e. the thing the whole release exists to produce — is
# the only warm color in the image. Blue/orange/grey also survives the common red-green deficiencies,
# which matters because these PNGs get pasted into the plan docs and read by whoever is around.
_SKY_COLOR = np.array([0.80, 0.87, 0.95])
_ROCK_COLOR = np.array([0.46, 0.46, 0.50])
_SHELTERED_AIR_COLOR = np.array([1.00, 0.66, 0.13])
_BEDROCK_COLOR = np.array([0.10, 0.10, 0.13])
# Rows outside [_WORLD_FLOOR_Y, _WORLD_CEIL_Y] are not "empty world", they are *not world at all*,
# and drawing them as sky or rock is exactly the misleading picture spec_constants.py's band-hangs-
# below-the-floor comment warns about. They get their own near-black, faintly striped fill.
_NOWORLD_COLOR = np.array([0.05, 0.05, 0.07])

# Mirrors export/export_onnx.py's MIN_Y / MAX_Y. Deliberately re-declared rather than imported:
# export_onnx imports torch at module scope, and viz.py is imported by the data-sanity dump, which
# has no reason to pull in torch. If Minecraft's world bounds ever change these two must move
# together — they describe the same world, not two conventions.
_WORLD_FLOOR_Y = -64
_WORLD_CEIL_Y = 319


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


# =================================================================================================
# Cross sections (Phase 4 — docs/phase4-plan.md §2.4)
#
# `render_terrain_rgb` above is a top-down view, and a top-down view of an overhang is a picture of
# the overhang's lip: the thing underneath it is, by definition, hidden. Everything volumetric this
# release adds is invisible from up there. These functions cut a vertical slice instead, so an
# undercut is a *shape* on the page rather than a number in a diagnostic table — which is what
# Gate 0's exit criterion asks a human to look at and sign off on.
#
# THE COORDINATE TRAP, stated once so the rest of the code can be short about it. The band is
# anchored per column (spec_constants.BAND_BOTTOM_OFFSET):
#
#     band index i of column c  <->  world y = round(h[c]) + BAND_BOTTOM_OFFSET + i
#
# so two adjacent columns' band index 30 are at *different* world y whenever their heights differ.
# Drawing band index straight up the page ("un-anchoring" skipped) produces a picture that is
# perfectly plausible — smooth, terrain-shaped, with convincing cavities — and wrong: it is the
# terrain flattened onto its own heightmap. Everything below inverts the mapping instead, going from
# each output pixel's absolute world y back to a band index per column, which is the direction that
# cannot silently succeed.
# =================================================================================================


def _resolve_slice_axis(axis: str) -> str:
    """`axis` names the HORIZONTAL axis of the resulting image, not the axis being cut across.
    `axis="x"` therefore means "columns of the image are x, sampled at a fixed z" — which reads the
    same way as "a slice along x". Spelled out because this repo's rasters are indexed `[z][x]` and
    docs/model-spec.md's whole axis-order section exists because that convention has bitten before.
    """
    a = str(axis).lower()
    if a not in ("x", "z"):
        raise ValueError(f"axis must be 'x' or 'z' (got {axis!r})")
    return a


def _as_int_index(value) -> int:
    """One slice position, from a Python int, a numpy integer, or a 0-d array. `np.isscalar` was
    used here first and is the wrong predicate: it is False for `np.array(5)` and for `np.int64(5)`
    from any argmax/nonzero result, so the natural way a caller produces an index (find the tallest
    column, slice there) raised TypeError. A non-integral float is rejected rather than truncated —
    `index=5.9` silently becoming row 5 is a wrong picture, not a convenience."""
    if isinstance(value, str) or np.ndim(value) != 0:
        raise TypeError(f"slice index must be a single whole number, got {value!r}")
    scalar = np.asarray(value).item()
    if isinstance(scalar, float) and not scalar.is_integer():
        raise TypeError(f"slice index must be a whole number of blocks, got {value!r}")
    return int(scalar)


def _resolve_indices(
    heightfield: np.ndarray,
    axis: str,
    index: Optional[Union[int, Sequence[int]]],
) -> list:
    """Normalizes `index` to a list of valid slice positions. `None` = the middle of the region,
    which for a `[CANVAS, CANVAS]` raster is the middle of the *canvas* (halo included) — that is
    still inside the real region, and picking the middle rather than 0 avoids slicing through halo,
    which is the least interesting and most edge-effect-prone part of the raster."""
    n = heightfield.shape[0] if axis == "x" else heightfield.shape[1]
    if index is None:
        idxs = [n // 2]
    elif np.ndim(index) == 0:
        idxs = [_as_int_index(index)]
    else:
        idxs = [_as_int_index(i) for i in index]
    for i in idxs:
        if not (0 <= i < n):
            raise IndexError(f"slice index {i} out of range for {axis}-slice of a {n}-wide raster")
    return idxs


def cross_section_y_range(
    heightfield: np.ndarray,
    water_level: Optional[float] = None,
    axis: str = "x",
    index: Optional[Union[int, Sequence[int]]] = None,
    margin: int = 4,
    clamp_to_world: bool = True,
) -> Tuple[int, int]:
    """Picks the `(y_lo, y_hi)` world-y window a cross section should draw, from the data.

    The band is 64 blocks in a 384-block world, so a fixed full-world window renders the only
    interesting part of the image as a 17%-tall stripe and the rest as sky — which is how you end up
    with a "the data looks fine" sign-off on a picture nobody could actually read. The window is the
    union of the sliced columns' band extents, so the band fills the frame at whatever elevation it
    happens to sit, plus `margin` blocks of air/rock context on each side.

    `water_level` is included when it is above the band, so an ocean surface is visible rather than
    cropped off the top of a picture of a seafloor.

    With `clamp_to_world`, the window is trimmed to the real world bounds: a column anchored at
    TERRAIN_CLIP_MIN puts its band bottom at -109, and 45 rows of "world that does not exist" is
    both noise and a lie. Pass `clamp_to_world=False` to deliberately look at the overhang (the
    renderer draws the out-of-world rows in `_NOWORLD_COLOR`, never as rock).
    """
    axis = _resolve_slice_axis(axis)
    idxs = _resolve_indices(heightfield, axis, index)
    h = np.asarray(heightfield, dtype=np.float64)
    lines = [h[i, :] if axis == "x" else h[:, i] for i in idxs]
    anchors = np.rint(np.concatenate(lines)).astype(np.int64)
    y_lo = int(anchors.min()) + BAND_BOTTOM_OFFSET - int(margin)
    y_hi = int(anchors.max()) + BAND_TOP_OFFSET + int(margin)
    if water_level is not None and float(water_level) > y_hi - margin:
        sea_top = int(np.ceil(float(water_level))) + int(margin)
        # ...but not at any price. A seafloor at TERRAIN_CLIP_MIN with sea level 63 is 111 blocks of
        # open water above a 64-block band, and the band is the subject. Past 2x the band height the
        # sea surface is dropped from the frame; what stays is the band plus up to `margin` rows
        # above it, which are drawn as water wherever they are at or below sea level and as sky
        # where they are not. Nothing is misrendered, the frame is simply cropped tighter.
        if sea_top - y_lo + 1 <= 2 * BAND_HEIGHT:
            y_hi = sea_top
    if clamp_to_world:
        y_lo = max(y_lo, _WORLD_FLOOR_Y)
        y_hi = min(y_hi, _WORLD_CEIL_Y)
    if y_hi < y_lo:  # only reachable from a hand-passed nonsense heightfield; degenerate, not fatal
        y_hi = y_lo
    return y_lo, y_hi


def render_cross_section(
    heightfield: np.ndarray,
    band: np.ndarray,
    water_level: Optional[float] = None,
    profile_map: Optional[np.ndarray] = None,
    axis: str = "x",
    index: Optional[int] = None,
    y_range: Optional[tuple] = None,
    margin: int = 4,
    clamp_to_world: bool = True,
) -> np.ndarray:
    """Renders one vertical slice as an RGB image (float in [0,1], `[n_y, n_horizontal, 3]`), the
    volumetric sibling of `render_terrain_rgb` and with the same return convention.

    Args:
        heightfield: `[CANVAS, CANVAS]` absolute world y per column, indexed `[z][x]`.
        band: `[CANVAS, CANVAS, BAND_HEIGHT]` occupancy (bool/uint8, 1 = solid), indexed `[z][x][i]`,
            anchored to `heightfield` per spec_constants' band block.
        water_level: scalar sea level for the region, in world y; `None` renders a waterless world
            (the same as passing a level below every cell, and useful for a dry-land archetype).
        profile_map: `[CANVAS, CANVAS]` surface-profile ids. When given, the topmost solid run of
            each column is dressed with its profile color (surface cell + 4 subsurface), matching
            docs/phase4-plan.md §1.5's "dress only the topmost solid run" rule — so the *underside*
            of an overhang renders as bare rock here exactly as it will in the exported volume.
        axis: horizontal axis of the image, "x" or "z" (see `_resolve_slice_axis`).
        index: row/column to slice at; `None` = middle of the canvas.
        y_range: `(y_lo, y_hi)` inclusive world-y window; `None` = `cross_section_y_range(...)`.
        margin: context blocks above/below the band when `y_range` is None.
        clamp_to_world: forwarded to `cross_section_y_range` when `y_range` is None. Set False to
            see the part of the band that hangs below y=-64 (drawn as not-world, never as rock).

    ROW ORDER: row 0 of the returned array is the TOP of the image, world y == `y_hi`, i.e.
    `row = y_hi - world_y`. Y increases *upward* on the page, which is the only orientation a reader
    will interpret correctly, and it is the flip matplotlib's default `origin="upper"` needs to make
    that true. An upside-down cross section is not obviously wrong to look at — sky and rock swap
    and it just reads as a strange cave system — so this is asserted in tests/test_viz.py rather
    than left to eyeballing.
    """
    axis = _resolve_slice_axis(axis)
    h = np.asarray(heightfield, dtype=np.float64)
    b = np.asarray(band)
    if b.ndim != 3 or b.shape[:2] != h.shape:
        raise ValueError(
            f"band must be [Z, X, BAND_HEIGHT] matching the heightfield's {h.shape}, got {b.shape}"
        )
    if b.shape[2] != BAND_HEIGHT:
        # A band with the wrong depth would still render — as terrain shifted vertically by the
        # difference — so this is checked rather than trusted. See spec_constants' assertion on
        # BAND_TOP_OFFSET - BAND_BOTTOM_OFFSET + 1 for the same class of silent misalignment.
        raise ValueError(f"band depth {b.shape[2]} != BAND_HEIGHT ({BAND_HEIGHT})")

    idx = _resolve_indices(h, axis, index)[0]
    if axis == "x":
        h_line, band_line = h[idx, :], b[idx, :, :]
        prof_line = None if profile_map is None else np.asarray(profile_map)[idx, :]
    else:
        h_line, band_line = h[:, idx], b[:, idx, :]
        prof_line = None if profile_map is None else np.asarray(profile_map)[:, idx]
    n_h = h_line.shape[0]
    band_line = band_line.astype(bool)

    # `np.rint` is half-to-even, matching Python's `round()` in the spec_constants formula. The
    # generator that writes the band MUST round the same way: rounding one direction here and the
    # other there shifts the whole picture by one block against its own heightmap, which looks fine.
    anchors = np.rint(h_line).astype(np.int64)
    band_bottom_y = anchors + BAND_BOTTOM_OFFSET

    if y_range is None:
        y_lo, y_hi = cross_section_y_range(h, water_level, axis, idx, margin, clamp_to_world)
    else:
        y_lo, y_hi = int(y_range[0]), int(y_range[1])
    ys = np.arange(y_hi, y_lo - 1, -1, dtype=np.int64)  # row 0 == y_hi: the flip, done once, here
    n_y = ys.size
    # A waterless region is "sea level below everything", not a special case in every mask below.
    water_y = -np.inf if water_level is None else float(water_level)

    # --- un-anchor: pixel world y -> per-column band index -----------------------------------
    band_i = ys[:, None] - band_bottom_y[None, :]          # [n_y, n_h]
    in_band = (band_i >= 0) & (band_i < BAND_HEIGHT)
    safe_i = np.clip(band_i, 0, BAND_HEIGHT - 1)
    cols = np.arange(n_h)[None, :]
    occ = band_line[cols, safe_i]                          # garbage outside the band; masked next
    # Below the band is solid rock by definition (spec_constants: "Below the band: solid rock,
    # deterministically"); above it is air. Both are structural, not data.
    solid = np.where(in_band, occ, band_i < 0)
    air = ~solid

    # --- sheltered air: the measurement the whole picture is for ------------------------------
    # "Air with at least one solid cell above it." Computed on the BAND, not on the drawn rows, so a
    # tight `y_range` that crops an overhang's ceiling out of frame still colors the air beneath it
    # correctly — cropping must change what you can see, never what the image claims.
    rev = band_line[:, ::-1]
    any_at_or_above = np.maximum.accumulate(rev, axis=1)[:, ::-1]  # [c, i] = any solid at index >= i
    solid_above = np.zeros_like(band_line)
    solid_above[:, :-1] = any_at_or_above[:, 1:]                   # strictly above
    sheltered = in_band & air & solid_above[cols, safe_i]

    # --- water: §1.5's openness rule, drawn exactly as it will be exported --------------------
    # Water iff air, at or below sea level, AND open to the sky. A dry cavern 40 blocks under sea
    # level stays dry here, which is the point: if this render floods it, the export rule and the
    # render disagree and one of them is a bug.
    #
    # The openness clause is only load-bearing because water is painted AFTER sheltered air below.
    # It was the other way round first, and an audit proved the result: with sheltered painted last,
    # dropping `open_to_sky` entirely gave byte-identical output on every scene, i.e. the render was
    # structurally incapable of showing the flooding it claims to check for, and the test that
    # claimed to check it could not fail. Do not reorder these two without re-reading this.
    open_to_sky = ~sheltered
    water = air & open_to_sky & (ys[:, None] <= water_y)

    # --- topmost solid run, for surface dressing ----------------------------------------------
    has_solid = band_line.any(axis=1)
    top_i = BAND_HEIGHT - 1 - np.argmax(band_line[:, ::-1], axis=1)
    # A column whose band is entirely air is legal and means the opposite of an overhang: there is
    # no solid anywhere in the interesting 64 blocks, i.e. a pit or chasm floor deeper than the band
    # reaches. Its topmost solid cell is then the first cell *below* the band, which is solid by
    # definition, so the dressing still has somewhere to go instead of vanishing.
    top_y = np.where(has_solid, band_bottom_y + top_i, band_bottom_y - 1)
    depth_below_surface = top_y[None, :] - ys[:, None]

    rgb = np.tile(_SKY_COLOR, (n_y, n_h, 1))

    # Rock, darkened with depth so a 100-block cliff face does not read as one flat grey slab. Small
    # amplitude on purpose — the eye must still bin it all as "rock", because the only distinction
    # this image is required to make loudly is solid vs sheltered air.
    #
    # Shaded by ABSOLUTE y, not by depth below each column's own surface. The per-column version was
    # tried first and is worse: at any cliff the surface jumps ~40 blocks between adjacent columns,
    # so the shading jumps too, and the render grows a crisp vertical line running from the cliff
    # foot to the bottom of the frame — a fault plane that is not in the data. A viz whose only job
    # is "believe the shapes you see" must not invent shapes.
    rock_shade = 1.0 - 0.30 * np.clip((float(top_y.max()) - ys) / 96.0, 0.0, 1.0)
    rgb = np.where(solid[..., None], _ROCK_COLOR[None, None, :] * rock_shade[:, None, None], rgb)

    if prof_line is not None:
        profile_ids = np.clip(np.asarray(prof_line).astype(np.int64), 0, NUM_PROFILES - 1)
        pcol = _PROFILE_COLORS[profile_ids]                       # [n_h, 3]
        surface = solid & (depth_below_surface == 0)
        subsurface = solid & (depth_below_surface > 0) & (depth_below_surface <= 4)
        rgb = np.where(surface[..., None], pcol[None, :, :], rgb)
        rgb = np.where(subsurface[..., None], pcol[None, :, :] * 0.72, rgb)

    # Sheltered air at full saturation, over the solid/dressed fills.
    rgb = np.where(sheltered[..., None], _SHELTERED_AIR_COLOR[None, None, :], rgb)

    # Water AFTER sheltered air, deliberately — see the openness note above. Under the correct rule
    # the two masks are disjoint and the order is a no-op; under a broken one this is what makes a
    # wrongly-flooded cavern show up as blue on the page instead of hiding behind the amber.
    water_shade = 1.0 - 0.25 * np.clip((water_y - ys) / 48.0, 0.0, 1.0) if np.isfinite(water_y) else np.ones(n_y)
    rgb = np.where(water[..., None], _WATER_COLOR[None, None, :] * water_shade[:, None, None], rgb)

    # y == -64 is bedrock in every Minecraft world and export_onnx.py writes it unconditionally;
    # drawing it gives the reader an absolute reference edge in an image whose y window moves.
    rgb = np.where((ys == _WORLD_FLOOR_Y)[:, None, None], _BEDROCK_COLOR[None, None, :], rgb)

    # Out-of-world rows (only reachable via an explicit y_range, since the default window clamps).
    # Faint stripes rather than flat fill so it cannot be mistaken for "very dark rock".
    outside = (ys < _WORLD_FLOOR_Y) | (ys > _WORLD_CEIL_Y)
    if outside.any():
        stripe = 1.0 + 0.6 * (((np.arange(n_h)[None, :] + ys[:, None]) // 3) % 2)
        void = np.clip(_NOWORLD_COLOR[None, None, :] * stripe[..., None], 0.0, 1.0)
        rgb = np.where(outside[:, None, None], void, rgb)

    return np.clip(rgb, 0.0, 1.0)


def _cross_section_legend_handles(profile_ids: Optional[np.ndarray]) -> list:
    """Legend for one figure. The dressing swatches are derived from the profile ids actually in the
    sliced columns, not hardcoded: the first version pinned them to `_PROFILE_COLORS[0]` (grass
    green), so an all-desert scene showed a green swatch labelled "dressed surface" above an image
    in which every dressed cell was tan — a reader matching swatch to image would conclude the
    legend described something that was not on the page."""
    handles = [
        Patch(facecolor=_SHELTERED_AIR_COLOR, edgecolor="0.3", label="air with solid above (overhang / cavity)"),
        Patch(facecolor=_SKY_COLOR, edgecolor="0.3", label="open air"),
        Patch(facecolor=_WATER_COLOR, edgecolor="0.3", label="water (open to sky)"),
        Patch(facecolor=_ROCK_COLOR, edgecolor="0.3", label="rock"),
        Patch(facecolor=_BEDROCK_COLOR, edgecolor="0.3", label="bedrock (y=-64)"),
    ]
    if profile_ids is None:
        return handles
    ids = np.clip(np.asarray(profile_ids).astype(np.int64).ravel(), 0, NUM_PROFILES - 1)
    present, counts = np.unique(ids, return_counts=True)
    # Most-covered profiles first, capped at 3: a slice across a biome boundary can legitimately
    # carry five, and a legend longer than the picture is its own kind of unreadable.
    order = present[np.argsort(-counts)][:3]
    for pid in order:
        handles.append(Patch(
            facecolor=_PROFILE_COLORS[pid], edgecolor="0.3",
            label=f"dressed surface: {PROFILE_TABLE[pid][0]} (topmost run)",
        ))
    return handles


def save_cross_section(
    heightfield: np.ndarray,
    band: np.ndarray,
    water_level: Optional[float],  # no default: out_path stays a required positional after it
    out_path: str,
    profile_map: Optional[np.ndarray] = None,
    axis: str = "x",
    index: Optional[Union[int, Sequence[int]]] = None,
    y_range: Optional[tuple] = None,
    margin: int = 4,
    clamp_to_world: bool = True,
    title: Optional[str] = None,
    caption: Optional[str] = None,
) -> None:
    """Writes a labelled cross-section figure, one panel per entry in `index` (an int, a sequence, or
    None for the middle of the canvas), the way `save_prompt_grid` writes the top-down dumps.

    Two callers, and they are the same call: the Gate 0 data-sanity dump over generator ground truth,
    and the per-epoch/gate dump over model output. Passing both through one function is what makes
    "did the 3D structure survive training" a comparison of two pictures rather than of two habits —
    the same argument §2.4 makes for running `diagnose_overhangs.py` on both.

    Unlike `save_prompt_grid` the y axis is LABELLED IN WORLD Y and the aspect is kept 1:1, because
    the number Gate 0 reads off this figure is a cavity's height in blocks (that measurement is what
    confirms or grows BAND_HEIGHT), and you cannot read a height off a stretched axis.
    """
    axis = _resolve_slice_axis(axis)
    idxs = _resolve_indices(np.asarray(heightfield), axis, index)
    # One shared y window across panels: panels at different windows silently rescale, and two
    # cavities of different size then look identical, which defeats the measurement above.
    if y_range is None:
        y_lo, y_hi = cross_section_y_range(heightfield, water_level, axis, idxs, margin, clamp_to_world)
    else:
        y_lo, y_hi = int(y_range[0]), int(y_range[1])

    n_h = np.asarray(heightfield).shape[1] if axis == "x" else np.asarray(heightfield).shape[0]
    span_y = max(y_hi - y_lo + 1, 1)
    # Aspect stays 1:1 (a block is square), so the panel's height follows from its width and the y
    # window. When that would be absurdly tall — a narrow test canvas, or a slice down a 250-block
    # canyon wall — the FIGURE shrinks rather than the aspect stretching: a stretched y axis makes
    # every cavity look taller than it is, and cavity height in blocks is the number Gate 0 reads.
    width_in = 15.0
    panel_h_in = max(1.5, width_in * span_y / max(n_h, 1))
    if panel_h_in > 7.0:
        width_in *= 7.0 / panel_h_in
        panel_h_in = 7.0
    # Chrome is reserved in INCHES, not in figure fractions: panel height varies with the y window
    # (a 70-block band is 2in tall, a slice down a 250-block canyon wall is 8in), and fractional
    # placement puts the legend through the middle of the short ones.
    chrome_bottom_in, chrome_top_in = 1.05, 0.55
    fig_h_in = panel_h_in * len(idxs) + chrome_bottom_in + chrome_top_in
    fig, axes = plt.subplots(len(idxs), 1, figsize=(width_in, fig_h_in), squeeze=False)
    for panel, i in enumerate(idxs):
        ax = axes[panel, 0]
        rgb = render_cross_section(
            heightfield, band, water_level, profile_map=profile_map,
            axis=axis, index=i, y_range=(y_lo, y_hi),  # window already resolved: shared by all panels
        )
        # extent is in world coordinates and row 0 is y_hi (see render_cross_section's ROW ORDER),
        # so origin="upper" is what puts high y at the top — the same flip, stated in axis units.
        ax.imshow(
            rgb, origin="upper", interpolation="nearest", aspect="equal",
            extent=(0.0, float(n_h), y_lo - 0.5, y_hi + 0.5),
        )
        ax.set_ylabel("world y")
        ax.set_xlabel(f"{axis} (canvas index)")
        sea = "no water" if water_level is None else f"sea level {water_level:.0f}"
        ax.set_title(
            f"slice along {axis} at {'z' if axis == 'x' else 'x'}={i}   "
            f"({sea}, band {BAND_HEIGHT} tall)",
            fontsize=9,
        )
        ax.grid(axis="y", color="0.5", alpha=0.35, linewidth=0.5)
        ax.set_axisbelow(False)

    # If any sliced column's band hangs below the world floor, say so in words. It is expected (see
    # spec_constants) and the picture simply cannot show it, so an unlabelled figure would leave a
    # reader believing they had seen the whole band.
    h_arr = np.asarray(heightfield, dtype=np.float64)
    sliced = np.concatenate([h_arr[i, :] if axis == "x" else h_arr[:, i] for i in idxs])
    lowest_band_bottom = int(np.rint(sliced).min()) + BAND_BOTTOM_OFFSET
    notes = []
    if caption:
        notes.append(caption)
    if lowest_band_bottom < y_lo:
        notes.append(
            f"band bottom reaches y={lowest_band_bottom}"
            + (f", {_WORLD_FLOOR_Y - lowest_band_bottom} blocks below the world floor"
               if lowest_band_bottom < _WORLD_FLOOR_Y else "")
            + f"; rows below y={y_lo} are not drawn"
        )
    # Legend swatches come from the profiles in the columns this figure actually drew, not from the
    # whole region: a slice through desert must not advertise the grass the next region over has.
    if profile_map is None:
        sliced_profiles = None
    else:
        p_arr = np.asarray(profile_map)
        sliced_profiles = np.concatenate([p_arr[i, :] if axis == "x" else p_arr[:, i] for i in idxs])
    fig.legend(
        handles=_cross_section_legend_handles(sliced_profiles),
        loc="lower center", bbox_to_anchor=(0.5, 0.30 / fig_h_in), ncol=3, fontsize=9, frameon=False,
    )
    if title:
        fig.suptitle(title, fontsize=13)
    if notes:
        fig.text(0.5, 0.08 / fig_h_in, "   ".join(notes), ha="center", fontsize=8, color="0.35")
    fig.tight_layout(
        rect=(0, chrome_bottom_in / fig_h_in, 1, 1.0 - (chrome_top_in / fig_h_in if title else 0.0))
    )
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
