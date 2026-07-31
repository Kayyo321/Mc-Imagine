"""
Tests for viz.render_cross_section (Phase 4 — docs/phase4-plan.md §2.4).

These are all assertions about *where things land in world y*, because that is the failure this
function can have without anyone noticing: the band is anchored per column
(`world y = round(h) + BAND_BOTTOM_OFFSET + i`), and a render that forgets to un-anchor, or flips
the y axis, or is off by one, still produces a smooth, terrain-shaped, entirely convincing picture.
Every case below therefore carves a cavity at a **known** world y and asserts the pixels come back
at exactly that world y — never "looks about right".
"""

import numpy as np

from mc_imagine_model.spec_constants import BAND_BOTTOM_OFFSET, BAND_HEIGHT, BAND_TOP_OFFSET
from mc_imagine_model.viz import (
    _NOWORLD_COLOR,
    _PROFILE_COLORS,
    _SHELTERED_AIR_COLOR,
    _SKY_COLOR,
    cross_section_y_range,
    render_cross_section,
)

N = 48  # small stand-in for CANVAS; these tests are about y, not about region size


def _heightfield(value: float) -> np.ndarray:
    return np.full((N, N), value, dtype=np.float32)


def _pure_heightfield_band(h: np.ndarray) -> np.ndarray:
    """The degenerate band: solid at and below the anchor, air above — i.e. exactly what v1.1.1
    terrain is, and exactly what an occupancy head that learns nothing must fall back to."""
    anchors = np.rint(h).astype(np.int64)
    ys = anchors[..., None] + BAND_BOTTOM_OFFSET + np.arange(BAND_HEIGHT)[None, None, :]
    return (ys <= anchors[..., None]).astype(np.uint8)


def _mask_of(rgb: np.ndarray, color: np.ndarray) -> np.ndarray:
    return np.all(np.isclose(rgb, color), axis=-1)


def test_degenerate_band_renders_as_an_ordinary_heightfield() -> None:
    """Air must be one contiguous run at the top of every column, with its lowest cell exactly one
    block above that column's rounded height. This is the case where the render is checkable against
    the heightfield alone, so it is where an inverted or off-by-one y mapping shows up first."""
    zz, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    h = (90.0 + 12.0 * np.sin(xx / 7.0) + 4.0 * np.cos(zz / 5.0)).astype(np.float32)
    band = _pure_heightfield_band(h)
    y_lo, y_hi = cross_section_y_range(h, water_level=None, axis="x", index=N // 2)
    rgb = render_cross_section(h, band, water_level=-1000.0, axis="x", index=N // 2,
                               y_range=(y_lo, y_hi))

    sky = _mask_of(rgb, _SKY_COLOR)
    anchors = np.rint(h[N // 2, :]).astype(np.int64)
    for c in range(N):
        col = sky[:, c]
        first_solid_row = int(np.argmax(~col))
        assert col[:first_solid_row].all() and not col[first_solid_row:].any(), (
            f"column {c}: air is not a single run at the top — the band was not un-anchored"
        )
        assert y_hi - first_solid_row == anchors[c], (
            f"column {c}: topmost solid at world y {y_hi - first_solid_row}, height says {anchors[c]}"
        )
    assert not _mask_of(rgb, _SHELTERED_AIR_COLOR).any(), (
        "a pure heightfield has no air with solid above it, by construction"
    )


def test_known_overhang_lands_at_the_world_y_it_was_carved_at() -> None:
    """A rectangular slab of air at world y 86..92 under a plateau at y=100 must come back as
    sheltered-air pixels covering exactly rows y=86..92 and exactly columns 20..29, with solid
    immediately above and below. Both bounds matter: too-high catches a sign flip, too-tall catches
    an inclusive/exclusive slip."""
    h = _heightfield(100.0)
    band = _pure_heightfield_band(h)
    air_lo, air_hi, x0, x1 = 86, 92, 20, 30
    i_lo = air_lo - 100 - BAND_BOTTOM_OFFSET
    i_hi = air_hi - 100 - BAND_BOTTOM_OFFSET
    band[:, x0:x1, i_lo:i_hi + 1] = 0

    y_lo, y_hi = cross_section_y_range(h, water_level=None, axis="x", index=N // 2)
    rgb = render_cross_section(h, band, water_level=-1000.0, axis="x", index=N // 2,
                               y_range=(y_lo, y_hi))
    sheltered = _mask_of(rgb, _SHELTERED_AIR_COLOR)
    rows, cols = np.nonzero(sheltered)
    world_ys = y_hi - rows

    assert (world_ys.min(), world_ys.max()) == (air_lo, air_hi)
    assert (cols.min(), cols.max()) == (x0, x1 - 1)
    assert sheltered.sum() == (air_hi - air_lo + 1) * (x1 - x0)
    # ceiling and floor of the cavity are solid, and the plateau above it is still open sky
    assert not sheltered[y_hi - (air_hi + 1), x0] and not _mask_of(rgb, _SKY_COLOR)[y_hi - (air_hi + 1), x0]
    assert not _mask_of(rgb, _SKY_COLOR)[y_hi - (air_lo - 1), x0]
    assert _mask_of(rgb, _SKY_COLOR)[y_hi - 101, x0]


def test_horizontal_slab_on_a_slope_stays_horizontal() -> None:
    """THE discriminating case for un-anchoring, and the one the flat-plateau tests above cannot be:
    on a flat heightfield, anchored and un-anchored rendering are indistinguishable, because every
    column's band sits at the same absolute y. Here the terrain slopes (h = 100 + x) while the
    carved cavity is horizontal in ABSOLUTE world y (95..99 in every column). A render that drew
    band index straight up the page would tilt that slab into a diagonal parallel to the surface —
    a picture that still looks like a perfectly reasonable cave. It must come back flat."""
    zz, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    h = (100.0 + xx).astype(np.float32)  # 1 block of rise per column
    band = _pure_heightfield_band(h)
    air_lo, air_hi = 95, 99
    anchors = np.rint(h[0, :]).astype(np.int64)
    for c in range(N):  # carve by absolute world y, per column, at that column's own band offset
        for y in range(air_lo, air_hi + 1):
            i = y - anchors[c] - BAND_BOTTOM_OFFSET
            if 0 <= i < BAND_HEIGHT:
                band[:, c, i] = 0

    y_lo, y_hi = 60, 160
    rgb = render_cross_section(h, band, water_level=None, axis="x", index=0, y_range=(y_lo, y_hi))
    sheltered = _mask_of(rgb, _SHELTERED_AIR_COLOR)

    # every column whose band actually covers y 95..99 must show all 5 cells, at exactly those rows
    for c in range(N):
        covered = [y for y in range(air_lo, air_hi + 1)
                   if 0 <= y - anchors[c] - BAND_BOTTOM_OFFSET < BAND_HEIGHT]
        rows = np.nonzero(sheltered[:, c])[0]
        assert set(y_hi - rows) == set(covered), (
            f"column {c} (h={anchors[c]}): cavity at world y {sorted(y_hi - rows)}, carved at {covered}"
        )
    # and the slab is flat while the terrain above it is not — the whole point of the case
    heights_of_slab = [np.nonzero(sheltered[:, c])[0].min() for c in range(N) if sheltered[:, c].any()]
    assert len(set(heights_of_slab)) == 1, "the cavity tilted with the terrain: band was not un-anchored"
    assert np.ptp(_mask_of(rgb, _SKY_COLOR).sum(axis=0)) > 0, "the terrain above it must still slope"


def test_row_zero_is_the_top_of_the_image() -> None:
    """Row 0 == y_hi. An upside-down cross section reads as a plausible cave system rather than as
    an obvious error, so the orientation is pinned here rather than left to whoever opens the PNG."""
    h = _heightfield(100.0)
    band = _pure_heightfield_band(h)
    rgb = render_cross_section(h, band, water_level=-1000.0, axis="x", index=0, y_range=(50, 150))
    assert rgb.shape == (101, N, 3)
    assert _mask_of(rgb, _SKY_COLOR)[0, 0], "row 0 (y=150, well above the surface) must be open air"
    assert not _mask_of(rgb, _SKY_COLOR)[-1, 0], "last row (y=50, deep underground) must be solid"


def test_axis_x_and_axis_z_slice_the_documented_axis() -> None:
    """`axis="x"` means the image's horizontal axis is x, sampled at a fixed z. Rasters here are
    indexed [z][x], so getting this backwards is a transpose that is invisible on symmetric test
    data — hence a deliberately asymmetric heightfield."""
    zz, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    h = (100.0 + 0.5 * xx).astype(np.float32)  # varies along x only
    band = _pure_heightfield_band(h)
    rgb_x = render_cross_section(h, band, -1000.0, axis="x", index=3, y_range=(90, 130))
    rgb_z = render_cross_section(h, band, -1000.0, axis="z", index=3, y_range=(90, 130))
    sky_x, sky_z = _mask_of(rgb_x, _SKY_COLOR), _mask_of(rgb_z, _SKY_COLOR)
    assert np.ptp(sky_x.sum(axis=0)) > 0, "an x-slice of x-varying terrain must not be flat"
    assert np.ptp(sky_z.sum(axis=0)) == 0, "a z-slice of x-varying terrain must be flat"


def test_dry_cavity_below_sea_level_is_not_flooded() -> None:
    """docs/phase4-plan.md §1.5: water iff air, at or below sea level, AND open to the sky.

    The discriminating pair is the point of this test. An earlier version asserted only that the
    sealed pocket was amber, which it is under *any* water rule, because sheltered air used to be
    painted over water — deleting the openness clause outright left the output byte-identical and
    the test green. So: two air cells at the SAME world y, one sealed and one open to the sky, whose
    colors must differ. Only the openness clause decides that, and nothing else in the render does.
    """
    h = _heightfield(100.0)
    band = _pure_heightfield_band(h)
    i_lo = 40 - 100 - BAND_BOTTOM_OFFSET
    band[:, 10:20, i_lo:i_lo + 5] = 0        # sealed pocket at y 40..44, deep under sea level
    open_col = 30
    band[:, open_col, i_lo:] = 0             # open shaft: air from y=40 clear up out of the band
    rgb = render_cross_section(h, band, water_level=63.0, axis="x", index=0, y_range=(30, 110))

    row_42 = 110 - 42
    sealed_px, open_px = rgb[row_42, 15], rgb[row_42, open_col]
    assert np.allclose(sealed_px, _SHELTERED_AIR_COLOR), "sealed cavern below sea level must stay dry"
    assert open_px[2] > open_px[0] + 0.3, "a shaft open to the sky below sea level must hold water"
    assert not np.allclose(sealed_px, open_px), (
        "same world y, same sea level: if these render the same, the openness rule is not being applied"
    )

    sheltered = _mask_of(rgb, _SHELTERED_AIR_COLOR)
    rows, _ = np.nonzero(sheltered)
    assert sheltered.sum() == 5 * 10
    assert set(110 - rows) == set(range(40, 45)), "the dry pocket moved or changed size"


def test_only_the_topmost_solid_run_is_dressed() -> None:
    """docs/phase4-plan.md §1.5: dress the topmost solid run only — surface block at its top cell,
    subsurface for the 4 below; every other solid cell is bare rock. The failure this pins is the
    one §8's risk table tracks: grass growing on the underside of an overhang and on cave floors,
    which is what "dress every solid run" produces and what an unmutated eyeball will not catch."""
    h = _heightfield(100.0)
    band = _pure_heightfield_band(h)
    air_lo, air_hi = 86, 92
    i_lo = air_lo - 100 - BAND_BOTTOM_OFFSET
    band[:, 20:30, i_lo:i_lo + (air_hi - air_lo) + 1] = 0
    profile_map = np.full((N, N), 1, dtype=np.uint8)  # sand, deliberately not profile 0
    y_lo, y_hi = 60, 110
    rgb = render_cross_section(h, band, water_level=None, profile_map=profile_map,
                               axis="x", index=0, y_range=(y_lo, y_hi))
    sand = _PROFILE_COLORS[1]
    col = 25  # a column containing the cavity

    assert np.allclose(rgb[y_hi - 100, col], sand), "y=100 is the topmost solid cell: surface block"
    for y in range(96, 100):
        assert np.allclose(rgb[y_hi - y, col], sand * 0.72), f"y={y} is within 4 of the surface"
    assert not np.allclose(rgb[y_hi - 95, col], sand * 0.72), "dressing must stop 4 blocks down"

    # The ceiling of the cavity (top of the second solid run, y=93) and its floor (y=85) are rock.
    for y in (93, 85):
        assert not np.allclose(rgb[y_hi - y, col], sand), (
            f"y={y} is not in the topmost run — dressing it grows grass under an overhang"
        )
        assert not np.allclose(rgb[y_hi - y, col], sand * 0.72), f"y={y} must be bare rock"
    assert rgb[y_hi - 93, col][0] == rgb[y_hi - 93, col][1], "rock is grey; sand is not"


def test_band_below_the_world_floor_is_clipped_by_default_and_marked_when_not() -> None:
    """spec_constants: a column at TERRAIN_CLIP_MIN (-48) has its band bottom at -109, 45 blocks
    below Minecraft's floor. The default window must stop at the floor; an explicitly wider window
    must draw those rows as not-world, never as rock (rock down there would be a lie about a world
    that does not extend that far)."""
    h = _heightfield(-48.0)
    band = _pure_heightfield_band(h)
    y_lo, y_hi = cross_section_y_range(h, water_level=63.0, axis="x", index=0)
    assert y_lo == -64, "default window must clamp to the world floor"
    assert y_hi >= -48 + BAND_TOP_OFFSET

    rgb = render_cross_section(h, band, 63.0, axis="x", index=0, y_range=(y_lo, y_hi))
    assert rgb.max() > 0.2, "clamped window must contain real terrain, not the not-world fill"

    wide = render_cross_section(h, band, 63.0, axis="x", index=0, y_range=(-120, -40))
    row_of_minus_100 = -40 - (-100)
    assert wide[row_of_minus_100, 0].max() < 0.15, "y=-100 is outside the world; it must not be rock"
    assert np.all(wide[row_of_minus_100, 0] >= _NOWORLD_COLOR * 0.9)
