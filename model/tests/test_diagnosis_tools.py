"""
Tests for the diagnosis tools: diagnose_terracing.py, diagnose_speckle.py, diagnose_overhangs.py.

The overhang tests below are all built on synthetic bands with ANALYTICALLY KNOWN answers and
assert exact integer counts. That is deliberate and it is worth saying why: a volumetric metric
that only asserts "> 0" passes on a transposed world, on an off-by-one at the band boundary, and on
a walkable-void definition that counts every column of a cave separately. Every one of those bugs
produces a plausible number, and docs/phase4-plan.md §9 copies these numbers into a release table.
"""

import json
import os

import numpy as np
import pytest
import torch

from mc_imagine_model.scripts.diagnose_terracing import compute_terracing_metrics
from mc_imagine_model.scripts.diagnose_speckle import compute_speckle_metrics
from mc_imagine_model.scripts.diagnose_overhangs import (
    aggregate_overhang_metrics,
    compute_overhang_metrics,
    compute_retention,
    evaluate_baseline_gate,
    heightfield_band,
    load_band,
    marginal_baseline_band,
    marginal_baseline_from_profile,
    resolve_band_paths,
    save_diagnostic_plot,
    print_report,
    _normalize_band,
    _percentiles_from_counts,
)
from mc_imagine_model.scripts.diagnose_overhangs import BAND_SIZE_WARNING_HEIGHT
from mc_imagine_model.spec_constants import (
    BAND_BOTTOM_OFFSET,
    BAND_HEIGHT,
    BAND_TOP_OFFSET,
)
from mc_imagine_model.model.imagine_net import ImagineNet


def test_terracing_metrics_flat_terrain() -> None:
    """A perfectly flat heightfield must have 0.0 mean grade, 100% step_0_flat, and 0.0 spike ratio."""
    flat_h = np.full((32, 32), 100.0, dtype=np.float32)
    metrics = compute_terracing_metrics(flat_h, chunk_span=2)

    assert metrics["mean_grade"] == 0.0
    assert metrics["flat_shelf_ratio"] == 1.0
    assert metrics["step_distribution_counts"]["step_0_flat"] == metrics["total_sampled_edges"]
    assert metrics["integer_cliff_spike_ratio"] == 0.0
    assert metrics["seam_analysis"]["seam_jump_ratio"] == 1.0


def test_terracing_metrics_smooth_ramp() -> None:
    """A smooth fractional ramp heightfield should have non-zero mean grade and low integer cliff spike ratio."""
    x = np.linspace(100.0, 110.0, 32)
    ramp_h = np.tile(x, (32, 1)).astype(np.float32)
    metrics = compute_terracing_metrics(ramp_h, chunk_span=2)

    assert metrics["mean_grade"] > 0.0
    assert metrics["max_delta"] < 1.0
    assert "integer_cliff_spike_ratio" in metrics


def test_terracing_metrics_staircase() -> None:
    """A synthetic staircase (integer steps of 2 blocks every 4 columns) must trigger high integer cliff spike ratio."""
    col_steps = np.floor(np.arange(32) / 4) * 2.0  # 0, 0, 0, 0, 2, 2, 2, 2, 4, 4...
    staircase_h = np.tile(col_steps, (32, 1)).astype(np.float32)
    metrics = compute_terracing_metrics(staircase_h, chunk_span=2)

    assert metrics["integer_cliff_spike_ratio"] > 0.8
    assert metrics["step_distribution_counts"]["step_2"] > 0


def test_speckle_metrics_coherent_biome() -> None:
    """A uniform biome grid must be classified as SPATIALLY_COHERENT with 100% neighbor identity ratio."""
    uniform_biome = np.zeros((32, 32), dtype=np.int64)
    metrics = compute_speckle_metrics(uniform_biome, chunk_span=8)

    assert metrics["coherence_classification"] == "SPATIALLY_COHERENT"
    assert metrics["neighbor_identity_ratio"] == 1.0
    assert metrics["transition_frequency"] == 0.0
    assert metrics["isolated_cell_percentage"] == 0.0
    assert metrics["patch_analysis"]["total_patches"] == 1
    assert metrics["patch_analysis"]["mean_patch_size_cells"] == 1024.0


def test_speckle_metrics_speckled_noise() -> None:
    """A checkerboard biome grid (alternating biomes) must be classified as HIGH_FREQUENCY_SPECKLED_NOISE."""
    r, c = np.ogrid[:32, :32]
    checkerboard = ((r + c) % 2).astype(np.int64)
    metrics = compute_speckle_metrics(checkerboard, chunk_span=8)

    assert metrics["coherence_classification"] == "HIGH_FREQUENCY_SPECKLED_NOISE"
    assert metrics["neighbor_identity_ratio"] == 0.0
    assert metrics["transition_frequency"] == 1.0
    assert metrics["isolated_cell_percentage"] == 100.0
    assert metrics["patch_analysis"]["patch_size_buckets"]["singleton_1_cell"] == 1024


def test_diagnose_scripts_on_checkpoint_if_available() -> None:
    """If run1/best.pt exists, verify that both compute tools work end-to-end with ImagineNet model."""
    ckpt_path = "model/training_runs/run1/best.pt"
    if not os.path.isfile(ckpt_path):
        pytest.skip(f"{ckpt_path} not found; skipping checkpoint test")

    from mc_imagine_model.export.export_onnx import load_imagine_net
    from mc_imagine_model.inference_utils import render_area
    from mc_imagine_model.scripts.diagnose_speckle import render_biome_area

    net = load_imagine_net(ckpt_path)
    net.eval()

    # Terracing check
    full_h, full_p, w_level = render_area(net, "towering snow-capped peaks", seed=12345, chunk_span=2, device="cpu")
    t_metrics = compute_terracing_metrics(full_h, chunk_span=2)
    assert "mean_grade" in t_metrics
    assert t_metrics["total_sampled_edges"] > 0

    # Speckle check
    b_grid = render_biome_area(net, "towering snow-capped peaks", seed=12345, chunk_span=2, device="cpu")
    s_metrics = compute_speckle_metrics(b_grid, chunk_span=2)
    assert "neighbor_identity_ratio" in s_metrics
    assert s_metrics["total_cells"] == 64


# =================================================================================================
# diagnose_overhangs.py (docs/phase4-plan.md §2.4)
#
# Fixture convention for everything below: bands are `[Z, X, I]` uint8, 1 = solid, `i` ascending
# from the band bottom — the ground-truth raster layout. Small band depths (8-12 instead of the
# real 64) are used so every expected count can be written out by hand; `band_height=` is passed
# explicitly so the layout auto-detection has something to match against.
# =================================================================================================


def _carve(band: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Carves air into every column across band indices `[lo, hi)`. Produces one continuous void
    spanning the whole grid — a room, as opposed to speckle."""
    band = band.copy()
    band[:, :, lo:hi] = 0
    return band


def _flat_band(nz: int, nx: int, h: int, surface_i: int) -> np.ndarray:
    """The degenerate heightfield column, tiled: solid for `i <= surface_i`, air above.
    Exactly one solid<->air transition per column, which is the whole thing Phase 4 exists to
    escape (docs/phase4-plan.md §0.2)."""
    band = np.zeros((nz, nx, h), dtype=np.uint8)
    band[:, :, : surface_i + 1] = 1
    return band


def test_overhang_pure_heightfield_reads_exactly_zero() -> None:
    """The degenerate case: a band that is a plain heightfield must read 0 on all four metrics,
    with no division by zero anywhere. This is what v1.1.1 produces, so it is the number Gate 0's
    new generator has to beat and it must be exactly 0, not 'small'."""
    band = _flat_band(4, 4, 10, surface_i=6)
    m = compute_overhang_metrics(band, band_height=10)

    assert m["overhang_cells"] == 0
    assert m["overhang_cell_pct"] == 0.0
    assert m["multi_run_columns"] == 0
    assert m["multi_run_column_pct"] == 0.0
    assert m["max_transitions_per_column"] == 1          # solid run -> air run, once
    assert m["transition_histogram"]["1_heightfield"] == 16
    assert m["cavity"]["count"] == 0
    assert m["cavity"]["mean_height"] == 0.0
    assert m["cavity"]["max_height"] == 0
    assert m["cavity"]["percentiles"] == {"p50": 0, "p90": 0, "p95": 0, "p99": 0}
    assert m["cavity"]["clipped_at_band_floor_pct"] == 0.0   # 0/0 must be 0.0, not nan
    assert m["walkable"]["void_count"] == 0
    assert m["walkable"]["floor_cells_per_1000_columns"] == 0.0


def test_overhang_retention_against_zero_ground_truth_is_undefined() -> None:
    """Retention against a heightfield ground truth divides by zero for every headline metric.
    `None` ("undefined"), never inf, nan or 0.0 — docs/phase4-plan.md §9 copies this column into a
    release table and '0%' would be a false claim about the model."""
    gt = compute_overhang_metrics(_flat_band(4, 4, 10, surface_i=6), band_height=10)
    model_band = _flat_band(4, 4, 10, surface_i=6)
    model_band[1, 1:3, 3:6] = 0                       # the model invented an overhang
    model = compute_overhang_metrics(model_band, band_height=10)

    retention = compute_retention(gt, model)
    assert set(retention.values()) == {None}
    for v in retention.values():
        assert v is None
        assert not isinstance(v, float)

    # And the defined case still divides normally.
    assert compute_retention(model, model)["overhang_cell_pct"] == pytest.approx(1.0)
    # No model at all (Gate 0) is also "undefined", not zero.
    assert set(compute_retention(model, None).values()) == {None}


def test_overhang_single_rectangular_cavity_exact_counts() -> None:
    """One known 3x2x1 cavity in an otherwise heightfield band. Every number below is counted by
    hand from the fixture, not read off the implementation."""
    band = _flat_band(4, 4, 10, surface_i=6)
    band[1, 1:3, 3:6] = 0        # columns (z=1, x=1) and (z=1, x=2), band indices 3,4,5 -> air
    m = compute_overhang_metrics(band, band_height=10)

    # 1. overhang cells: 3 air cells x 2 columns, each with solid at i=6 above them. The ordinary
    #    above-surface air at i=7,8,9 has nothing solid above it and must NOT count.
    assert m["overhang_cells"] == 6
    assert m["overhang_cell_pct"] == pytest.approx(100.0 * 6 / (4 * 4 * 10))   # 3.75%

    # 2. multi-run columns: the two carved columns are solid/air/solid/air = 3 transitions.
    assert m["multi_run_columns"] == 2
    assert m["multi_run_column_pct"] == pytest.approx(12.5)
    assert m["max_transitions_per_column"] == 3
    assert m["transition_histogram"]["3"] == 2
    assert m["transition_histogram"]["1_heightfield"] == 14

    # 3. cavities: two roofed air runs of height 3. The unroofed sky run in every column is excluded.
    assert m["cavity"]["count"] == 2
    assert m["cavity"]["mean_height"] == pytest.approx(3.0)
    assert m["cavity"]["max_height"] == 3
    assert m["cavity"]["percentiles"]["p50"] == 3
    assert m["cavity"]["height_counts"][3] == 2
    assert sum(m["cavity"]["height_counts"]) == 2
    assert m["cavity"]["clipped_at_band_floor"] == 0

    # 4. walkable: >=3 tall, and the two floors are x-adjacent at the same band index -> ONE void
    #    with two floor cells. Not two voids: the count is components, so a wide cave mouth is 1.
    assert m["walkable"]["void_count"] == 1
    assert m["walkable"]["floor_cells"] == 2
    assert m["walkable"]["largest_void_floor_cells"] == 2
    assert m["walkable"]["voids_per_1000_columns"] == pytest.approx(1000.0 / 16)


def test_overhang_walkable_height_boundary_two_versus_three() -> None:
    """Pins §2.4's ">=3 blocks tall". A 2-block cavity is a crawlspace and must not count; the same
    cavity one block taller must. Everything else about the fixture is identical."""
    two = _flat_band(4, 4, 10, surface_i=6)
    two[1, 1:3, 4:6] = 0                              # 2 tall
    m2 = compute_overhang_metrics(two, band_height=10)
    assert m2["cavity"]["count"] == 2 and m2["cavity"]["max_height"] == 2
    assert m2["walkable"]["void_count"] == 0
    assert m2["walkable"]["floor_cells"] == 0

    three = _flat_band(4, 4, 10, surface_i=6)
    three[1, 1:3, 3:6] = 0                            # 3 tall, same footprint
    m3 = compute_overhang_metrics(three, band_height=10)
    assert m3["cavity"]["count"] == 2 and m3["cavity"]["max_height"] == 3
    assert m3["walkable"]["void_count"] == 1


def test_overhang_insufficient_floor_is_not_walkable() -> None:
    """A tall cavity one column wide has 1 block of floor, not 2, so it is a chimney and not a room.
    It is still a cavity and still counts toward the height distribution."""
    band = _flat_band(4, 4, 10, surface_i=6)
    band[1, 1, 3:6] = 0
    m = compute_overhang_metrics(band, band_height=10)

    assert m["cavity"]["count"] == 1
    assert m["cavity"]["max_height"] == 3
    assert m["multi_run_columns"] == 1
    assert m["walkable"]["void_count"] == 0
    assert m["walkable"]["floor_cells"] == 0
    assert m["walkable"]["largest_void_floor_cells"] == 1   # the component exists, it is just small


def test_overhang_walkable_floor_step_tolerance_is_actually_applied() -> None:
    """Two adjacent columns whose floors differ by N blocks are the same walkable surface at
    tolerance >= N and not below it.

    THIS TEST EXISTS BECAUSE THE TOLERANCE ONCE DID NOTHING. The first implementation labelled a
    dense mask with a 3x3x3 structuring element, which can only reach +/-1 along the level axis, so
    every tolerance >= 1 was identical: 2, 3, 5, 10 and 50 all produced byte-identical output while
    the report echoed the requested value back as if it had been applied. The one documented
    workaround for the band-index problem therefore silently did nothing, and an operator probing
    sensitivity would have concluded the terrain genuinely lacked connected floors. Every tolerance
    below is checked at a step it must and must not bridge."""
    band = _flat_band(1, 3, 16, surface_i=13)
    band[0, 0, 3:6] = 0        # floor at band index 3
    band[0, 1, 5:8] = 0        # floor at band index 5  -> 2 away from its left neighbour
    band[0, 2, 8:11] = 0       # floor at band index 8  -> 3 away from its left neighbour

    def voids(tol: int) -> int:
        return compute_overhang_metrics(band, band_height=16,
                                        floor_step_tolerance=tol)["walkable"]["void_count"]

    assert voids(0) == 0       # nothing level with anything: three components of size 1
    assert voids(1) == 0       # steps of 2 and 3 are still too big
    assert voids(2) == 1       # joins columns 0-1 only: one component of size 2
    assert voids(3) == 1       # joins all three: still one component, now of size 3
    m3 = compute_overhang_metrics(band, band_height=16, floor_step_tolerance=3)
    assert m3["walkable"]["floor_cells"] == 3
    assert m3["walkable"]["largest_void_floor_cells"] == 3
    m2 = compute_overhang_metrics(band, band_height=16, floor_step_tolerance=2)
    assert m2["walkable"]["floor_cells"] == 2
    # ...and a tolerance far beyond the data does not keep changing anything.
    assert voids(50) == 1


def test_walkable_voids_connect_along_z_as_well_as_x() -> None:
    """`compute_overhang_metrics` claims 4-connectivity in BOTH x and z, on the grounds that a ledge
    running along z is as walkable as one along x and the headline number must not depend on world
    orientation. Nothing tested it: every other walkable fixture is x-adjacent, so restricting the
    implementation to x alone passed the whole suite. Both orientations of the same logical scene
    must give the same answer."""
    along_x = _flat_band(4, 4, 12, surface_i=9)
    along_x[1, 1:3, 3:6] = 0                 # two columns adjacent in x
    m_x = compute_overhang_metrics(along_x, band_height=12)

    along_z = _flat_band(4, 4, 12, surface_i=9)
    along_z[1:3, 1, 3:6] = 0                 # the same pair, adjacent in z
    m_z = compute_overhang_metrics(along_z, band_height=12)

    assert m_x["walkable"]["void_count"] == 1
    assert m_z["walkable"]["void_count"] == 1
    assert m_x["walkable"]["floor_cells"] == m_z["walkable"]["floor_cells"] == 2
    # Diagonal-only neighbours are NOT connected: 4-connectivity, not 8.
    diagonal = _flat_band(4, 4, 12, surface_i=9)
    diagonal[1, 1, 3:6] = 0
    diagonal[2, 2, 3:6] = 0
    assert compute_overhang_metrics(diagonal, band_height=12)["walkable"]["void_count"] == 0


def test_walkable_voids_cliff_fixture_world_y() -> None:
    """BLOCKER FIXTURE: docs/phase4-plan.md's own "definition of done" scene — a 2:1 cliff carrying
    ONE continuous shelter you can walk under, whose floor is flat in WORLD Y.

    Ground truth is unambiguous by construction: 1 shelter, 8 x 24 = 192 floor cells. Measured in
    band-index space it reads 24 voids, because the per-column anchor moves 2 blocks per column and
    band index therefore severs the shelter into 24 one-column-wide strips — and severing a
    component RAISES the count. A 24x overcount of the one number §9 describes as "the number that
    matches the screenshot".

    Note the tol=2 row: in band-index space the tolerance that works is the local grade, so there is
    no single correct value. In world y, tol=0 is already exactly right."""
    Z, X = 8, 24
    hf = np.tile(100.0 - 2.0 * np.arange(X), (Z, 1)).astype(np.float32)   # 2 blocks of drop per column
    anchor = np.rint(hf).astype(np.int64)
    i = np.arange(BAND_HEIGHT)[None, None, :]
    world_y = anchor[:, :, None] + BAND_BOTTOM_OFFSET + i
    band = (world_y <= anchor[:, :, None])                 # plain heightfield expansion
    band &= ~((world_y >= 45) & (world_y <= 47))           # the shelter: 3 blocks tall, flat in y
    # The shelter must lie inside every column's band for the fixture to mean anything.
    assert anchor.max() - 61 <= 45 and 47 <= anchor.min() + 2

    def walk(tol: int, heightfield):
        return compute_overhang_metrics(band, heightfield=heightfield, band_height=BAND_HEIGHT,
                                        floor_step_tolerance=tol)["walkable"]

    w = walk(1, hf)
    assert w["floor_space"] == "world_y"
    assert w["void_count"] == 1                    # <- the truth
    assert w["floor_cells"] == Z * X == 192
    assert w["largest_void_floor_cells"] == 192
    assert walk(0, hf)["void_count"] == 1          # world y needs no tolerance at all here

    b0, b1, b2 = walk(0, None), walk(1, None), walk(2, None)
    assert b0["floor_space"] == "band_index"
    assert b0["void_count"] == 24 and b1["void_count"] == 24     # 24x overcount at the default
    assert b0["largest_void_floor_cells"] == 8                   # severed into per-column strips
    assert b2["void_count"] == 1                                 # only once tol matches the grade
    # Floor CELLS are unaffected — it is the void COUNT that inflates, which is the headline.
    assert b0["floor_cells"] == b1["floor_cells"] == w["floor_cells"] == 192


def test_walkable_voids_world_y_matches_band_index_on_flat_terrain() -> None:
    """The two spaces must agree exactly where the anchor is constant — otherwise the world-y path
    is not a refinement of the band-index one but a different metric."""
    flat_h = np.full((4, 4), 64.0, dtype=np.float32)
    band = _flat_band(4, 4, BAND_HEIGHT, surface_i=40)
    band[1, 1:3, 20:24] = 0
    a = compute_overhang_metrics(band, heightfield=flat_h)
    b = compute_overhang_metrics(band)
    assert a["walkable"]["void_count"] == b["walkable"]["void_count"] == 1
    assert a["walkable"]["floor_cells"] == b["walkable"]["floor_cells"] == 2


def test_heightfield_shape_mismatch_raises() -> None:
    """A heightfield that does not match the column grid would silently anchor floors to the wrong
    columns — a transposition bug wearing a different hat."""
    band = _flat_band(4, 5, 8, surface_i=5)
    with pytest.raises(ValueError, match="does not match the band"):
        compute_overhang_metrics(band, heightfield=np.zeros((5, 4)), band_height=8)


def test_overhang_three_solid_runs_does_not_saturate() -> None:
    """A column with THREE solid runs must report 5 transitions, not 'multi-run, saturated at 2'.
    Gate 2's statistic is a column *rate*, but the transition histogram is what says whether those
    columns hold one cavity or a stack of them."""
    band = np.zeros((1, 1, 12), dtype=np.uint8)
    band[0, 0, 0:2] = 1
    band[0, 0, 4:6] = 1
    band[0, 0, 8:10] = 1        # solid, air, solid, air, solid, air(top)
    m = compute_overhang_metrics(band, band_height=12)

    assert m["max_transitions_per_column"] == 5
    assert m["transition_histogram"]["5_plus"] == 1
    assert m["multi_run_columns"] == 1
    # Two roofed air runs (i=2..3 and i=6..7); the run at i=10..11 reaches the band top and is open
    # to whatever is above, so it is not a cavity.
    assert m["cavity"]["count"] == 2
    assert m["cavity"]["height_counts"][2] == 2
    assert m["overhang_cells"] == 4


def test_overhang_cavity_touching_band_floor_counts_and_is_flagged() -> None:
    """A cavity whose bottom is band index 0 is bounded below by the known-solid region under the
    band, so it is a real cavity with a real floor and the step into it is a real transition. Its
    measured height is a lower bound, which `clipped_at_band_floor` is what reports."""
    band = np.zeros((1, 1, 10), dtype=np.uint8)
    band[0, 0, 3:8] = 1          # air 0..2, solid 3..7, air 8..9
    m = compute_overhang_metrics(band, band_height=10)

    assert m["cavity"]["count"] == 1
    assert m["cavity"]["max_height"] == 3
    assert m["cavity"]["clipped_at_band_floor"] == 1
    assert m["cavity"]["clipped_at_band_floor_pct"] == pytest.approx(100.0)
    # 2 internal transitions + 1 for the virtual solid cell below index 0.
    assert m["max_transitions_per_column"] == 3
    assert m["multi_run_columns"] == 1
    assert m["overhang_cells"] == 3


def test_overhang_both_input_layouts_give_identical_metrics() -> None:
    """THE TRANSPOSITION GUARD. `[z][x][i]` (ground truth) and `[i][z][x]` (occupancy head output)
    describing the same logical volume must produce bit-identical metric dicts. A transposed band
    reports a transposed world and every summary statistic still looks reasonable, so this is
    asserted rather than eyeballed. Z != X on purpose: a swap of those two axes must also fail."""
    rng = np.random.default_rng(0)
    zxi = rng.integers(0, 2, size=(4, 5, 8)).astype(np.uint8)
    izx = np.moveaxis(zxi, 2, 0)
    assert izx.shape == (8, 4, 5)

    m_zxi = compute_overhang_metrics(zxi, band_height=8)
    m_izx = compute_overhang_metrics(izx, band_height=8)
    assert m_zxi == m_izx
    assert m_zxi["grid_shape"] == [4, 5]
    # The fixture is only a guard if it actually contains 3D structure to get wrong.
    assert m_zxi["multi_run_columns"] > 0 and m_zxi["cavity"]["count"] > 0

    # Explicit layouts agree with auto-detection...
    assert compute_overhang_metrics(izx, layout="izx", band_height=8) == m_izx
    assert compute_overhang_metrics(zxi, layout="zxi", band_height=8) == m_zxi
    # ...and naming the WRONG one on a non-cubic band is caught by the post-normalization depth
    # check rather than quietly measuring a transposed world.
    with pytest.raises(ValueError, match="band depth"):
        compute_overhang_metrics(zxi, layout="izx", band_height=8)


def test_overhang_ambiguous_cube_layout_raises_rather_than_guessing() -> None:
    """A `[H, H, H]` band matches both layouts. Guessing would be a coin flip on which world gets
    reported, so it raises and asks for `layout=`."""
    cube = np.ones((8, 8, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="ambiguous"):
        compute_overhang_metrics(cube, band_height=8)
    # ...and with the layout named it works.
    assert compute_overhang_metrics(cube, layout="zxi", band_height=8)["num_columns"] == 64

    with pytest.raises(ValueError, match="band depth"):
        compute_overhang_metrics(np.ones((4, 5, 6), dtype=np.uint8), band_height=8)


def test_overhang_accepts_head_output_batch_layout() -> None:
    """`[B, BAND_HEIGHT, 16, 16]` straight off the occupancy head is tiled into a square chunk grid
    and must match the same volume assembled by hand."""
    rng = np.random.default_rng(1)
    chunks = rng.integers(0, 2, size=(4, 8, 2, 2)).astype(np.uint8)   # B=4 -> 2x2 chunk grid
    grid = np.zeros((4, 4, 8), dtype=np.uint8)
    for idx in range(4):
        r, c = idx // 2, idx % 2
        grid[r * 2:(r + 1) * 2, c * 2:(c + 1) * 2, :] = np.moveaxis(chunks[idx], 0, 2)

    assert compute_overhang_metrics(chunks, band_height=8) == compute_overhang_metrics(grid, band_height=8)

    # REGRESSION: the assembled grid must be treated as [Z,X,I] by construction and not re-sniffed.
    # A 4x4 tile of 16x16 chunks is [64,64,64], which auto-detection rightly calls ambiguous — so
    # the most ordinary Gate 2 invocation there is (`--model band.npy` straight off the head) used
    # to die with "pass layout= explicitly". Same shape here in miniature: B=4, H=4, 2x2 chunks.
    cubic = rng.integers(0, 2, size=(4, 4, 2, 2)).astype(np.uint8)
    assert compute_overhang_metrics(cubic, band_height=4)["grid_shape"] == [4, 4]

    with pytest.raises(ValueError, match="not a square grid"):
        compute_overhang_metrics(rng.integers(0, 2, size=(3, 8, 2, 2)).astype(np.uint8), band_height=8)


def test_overhang_float_band_is_thresholded() -> None:
    """Occupancy probabilities work directly at 0.5; raw logits work at 0.0. Integer bands are
    compared != 0 and ignore the threshold, so a uint8 0/1 raster cannot be silently emptied by a
    threshold argument meant for probabilities."""
    band = _flat_band(2, 2, 8, surface_i=4)
    probs = np.where(band > 0, 0.9, 0.1).astype(np.float32)
    assert compute_overhang_metrics(probs, band_height=8) == compute_overhang_metrics(band, band_height=8)

    logits = np.where(band > 0, 2.0, -2.0).astype(np.float32)
    assert compute_overhang_metrics(logits, band_height=8, threshold=0.0) == \
        compute_overhang_metrics(band, band_height=8)
    # A uint8 band is unaffected by --threshold; a float one is not.
    assert compute_overhang_metrics(band, band_height=8, threshold=0.0)["solid_fraction"] == \
        pytest.approx(5 / 8)


def test_marginal_baseline_is_the_thresholded_average_band() -> None:
    """The baseline is computable by hand here: three heightfield columns and one all-solid column
    give a per-index marginal of [1, 1, 0.25, 0.25], which thresholds to a single template column
    [solid, solid, air, air] tiled everywhere."""
    band = np.zeros((2, 2, 4), dtype=np.uint8)
    band[:, :, 0:2] = 1                # every column solid at i=0,1
    band[1, 1, :] = 1                  # one column solid all the way up
    m = compute_overhang_metrics(band, band_height=4)
    assert m["solid_fraction_by_band_index"] == [1.0, 1.0, 0.25, 0.25]

    baseline = marginal_baseline_band(band, band_height=4)
    expected = np.zeros((2, 2, 4), dtype=bool)
    expected[:, :, 0:2] = True
    assert np.array_equal(baseline, expected)

    b_metrics = compute_overhang_metrics(baseline, band_height=4)
    assert b_metrics["multi_run_column_pct"] == 0.0
    assert b_metrics["overhang_cell_pct"] == 0.0
    gate = evaluate_baseline_gate(m, b_metrics, margin_pts=5.0)
    assert gate["baseline_multi_run_column_pct"] == 0.0
    assert gate["baseline_is_degenerate"] is False
    assert gate["conditions"]["multi_run_margin"] is False   # this fixture's own rate is 0% too
    assert gate["verdict"] == "FAIL"


def test_marginal_baseline_degenerate_all_or_nothing_property() -> None:
    """Every column of the baseline is the same column, so its multi-run rate is 0% or 100% and
    nothing between. A 100% baseline means the AVERAGE band has multiple solid runs, which makes
    the Gate 2 comparison uninformative — so it is flagged rather than silently passed through."""
    band = np.zeros((2, 2, 4), dtype=np.uint8)
    band[:, :, 0] = 1
    band[:, :, 2] = 1                  # every column is solid/air/solid/air
    b_metrics = compute_overhang_metrics(marginal_baseline_band(band, band_height=4), band_height=4)
    assert b_metrics["multi_run_column_pct"] == 100.0

    gate = evaluate_baseline_gate(b_metrics, b_metrics, margin_pts=5.0)
    assert gate["baseline_is_degenerate"] is True
    assert gate["delta_pts"] == 0.0
    assert gate["verdict"] == "FAIL"


def test_baseline_gate_condition_1_uses_the_margin() -> None:
    """'Meaningfully above' is a number, or it is not a gate. Condition 1 in isolation, driven with
    minimal dicts so the margin arithmetic is visible on its own."""
    base = {"multi_run_column_pct": 0.0}
    assert evaluate_baseline_gate({"multi_run_column_pct": 4.9}, base, 5.0)["conditions"]["multi_run_margin"] is False
    assert evaluate_baseline_gate({"multi_run_column_pct": 5.0}, base, 5.0)["conditions"]["multi_run_margin"] is True
    g = evaluate_baseline_gate({"multi_run_column_pct": 12.0}, {"multi_run_column_pct": 2.0}, 5.0)
    assert g["delta_pts"] == pytest.approx(10.0)
    assert g["conditions"]["multi_run_margin"] is True
    # The margin is APPLIED, not merely echoed: raising it past the delta flips the condition.
    assert evaluate_baseline_gate({"multi_run_column_pct": 12.0}, {"multi_run_column_pct": 2.0},
                                  10.01)["conditions"]["multi_run_margin"] is False


def test_baseline_gate_speckle_fails_the_conjunction() -> None:
    """§7's go/no-go is a conjunction because condition 1 alone is gameable by exactly the failure
    §3.1 predicts. A 1-block-speckle band scores a PERFECT multi-run rate — 100.00% against a 0.00%
    baseline — while containing nothing a player can enter. Every number in this test is the
    justification cited in the plan, so it is pinned here rather than left in a docstring.

    This band would have PASSED the pre-conjunction gate. It now FAILS conditions 2 and 3."""
    speckle = _flat_band(8, 8, 12, surface_i=9)
    speckle[:, :, 3] = 0
    speckle[:, :, 6] = 0                      # two 1-block cavities in every column
    m = compute_overhang_metrics(speckle, band_height=12)
    assert m["multi_run_column_pct"] == 100.0
    assert m["cavity"]["count"] == 2 * 64
    assert m["cavity"]["max_height"] == 1
    assert m["cavity"]["percentiles"]["p50"] == 1 and m["cavity"]["percentiles"]["p90"] == 1
    assert m["walkable"]["void_count"] == 0

    roomy = _flat_band(8, 8, 12, surface_i=9)
    roomy[:, :, 3:7] = 0                      # one continuous 4-tall room across all 64 columns
    gt = compute_overhang_metrics(roomy, band_height=12)
    assert gt["walkable"]["void_count"] == 1

    gate = evaluate_baseline_gate(m, {"multi_run_column_pct": 0.0}, 5.0, gt_metrics=gt)
    assert gate["verdict"] == "FAIL"
    assert gate["conditions"] == {"multi_run_margin": True,
                                  "walkable_void_retention": False,
                                  "p90_cavity_height": False}
    assert gate["walkable_void_retention"] == 0.0
    assert gate["observed_p50_cavity_height"] == 1
    assert gate["observed_p90_cavity_height"] == 1
    assert gate["is_gate2_verdict"] is True

    # OVERHANG CELLS DO NOT DISCRIMINATE and must never be substituted for these conditions. The
    # speckle band scores a substantial overhang percentage while containing nothing enterable —
    # the metric measures how much air has solid above it, which says nothing about whether the air
    # is a room. (Whether speckle scores ABOVE or BELOW a structured band is purely a function of
    # the two densities: on this 12-deep fixture the 4-tall room wins, on a 64-deep band at 18%
    # pocket density the speckle wins 14.06% to 6.07%. Neither ordering is a property to rely on.)
    assert m["overhang_cell_pct"] > 10.0
    assert m["walkable"]["void_count"] == 0


def test_baseline_gate_speckle_is_not_rescued_by_one_room() -> None:
    """An earlier version of this check keyed on `walkable voids == 0`, which one small room
    silences while the band is still overwhelmingly speckle. Condition 3 is a PERCENTILE precisely
    so it cannot be bought that cheaply: adding a 2x2 room to 4 of 1024 columns (0.39%) leaves p90
    at 1 and the verdict at FAIL."""
    speckle = _flat_band(32, 32, 12, surface_i=9)
    speckle[:, :, 3] = 0
    speckle[:, :, 6] = 0
    speckle[0:2, 0:2, 3:7] = 0                # one 2x2 room, 4 columns out of 1024
    m = compute_overhang_metrics(speckle, band_height=12)
    assert m["walkable"]["void_count"] == 1           # the room exists, so a `voids == 0` test dies
    assert m["walkable"]["floor_cells"] == 4
    assert m["cavity"]["percentiles"]["p90"] == 1     # ...and it moves nothing at p90

    gt = compute_overhang_metrics(_carve(_flat_band(32, 32, 12, surface_i=9), 3, 7), band_height=12)
    gate = evaluate_baseline_gate(dict(m, multi_run_column_pct=100.0),
                                  {"multi_run_column_pct": 0.0}, 5.0, gt_metrics=gt)
    assert gate["conditions"]["p90_cavity_height"] is False
    assert gate["verdict"] == "FAIL"


def test_baseline_gate_min_void_retention_is_applied() -> None:
    """Condition 2's threshold is a parameter and it is USED — Blocker 2 was a parameter echoed but
    never applied, so every threshold in this file is now pinned by flipping it across the value."""
    gt = compute_overhang_metrics(_carve(_flat_band(8, 8, 12, surface_i=9), 3, 7), band_height=12)
    model = compute_overhang_metrics(_carve(_flat_band(8, 8, 12, surface_i=9), 3, 7), band_height=12)
    obs = dict(model, multi_run_column_pct=100.0)
    base = {"multi_run_column_pct": 0.0}

    g = evaluate_baseline_gate(obs, base, 5.0, gt_metrics=gt, min_void_retention=1.0)
    assert g["walkable_void_retention"] == pytest.approx(1.0)
    assert g["conditions"]["walkable_void_retention"] is True and g["verdict"] == "PASS"
    g = evaluate_baseline_gate(obs, base, 5.0, gt_metrics=gt, min_void_retention=1.5)
    assert g["conditions"]["walkable_void_retention"] is False and g["verdict"] == "FAIL"


def test_baseline_gate_undefined_retention_is_not_met() -> None:
    """Ground truth with no walkable voids cannot say whether a model retained them. An undefined
    condition is NOT met: this gate authorizes a 9-hour run and must not do so on an unmeasurable
    criterion."""
    gt = compute_overhang_metrics(_flat_band(8, 8, 12, surface_i=9), band_height=12)
    assert gt["walkable"]["voids_per_1000_columns"] == 0.0
    model = compute_overhang_metrics(_carve(_flat_band(8, 8, 12, surface_i=9), 3, 7), band_height=12)
    g = evaluate_baseline_gate(dict(model, multi_run_column_pct=100.0),
                               {"multi_run_column_pct": 0.0}, 5.0, gt_metrics=gt)
    assert g["walkable_void_retention"] is None
    assert g["conditions"]["walkable_void_retention"] is False
    assert g["verdict"] == "FAIL"


def test_baseline_gate_gate0_is_not_a_gate2_verdict() -> None:
    """Without ground truth there is no retention to evaluate, so condition 2 reports `None` and the
    result is flagged as not being the Gate 2 verdict — the report prints 'DATA CHECK', not
    'VERDICT', on that path."""
    m = compute_overhang_metrics(_carve(_flat_band(8, 8, 12, surface_i=9), 3, 7), band_height=12)
    g = evaluate_baseline_gate(dict(m, multi_run_column_pct=100.0), {"multi_run_column_pct": 0.0}, 5.0)
    assert g["is_gate2_verdict"] is False
    assert g["conditions"]["walkable_void_retention"] is None
    assert g["verdict"] == "PASS"          # conditions 1 and 3 only


def test_percentiles_from_counts_are_exact_nearest_rank() -> None:
    """Cavity heights are integers, so percentiles are exact and are always an observed height —
    never an interpolated 4.5 blocks of headroom."""
    counts = np.zeros(11, dtype=np.int64)
    counts[1] = 1
    counts[2] = 1
    counts[3] = 1                                   # heights {1, 2, 3}
    p = _percentiles_from_counts(counts, (50, 90, 99))
    assert p == {"p50": 2, "p90": 3, "p99": 3}
    assert _percentiles_from_counts(np.zeros(11, dtype=np.int64), (50,)) == {"p50": 0}


def test_aggregate_pools_counts_and_recomputes_percentages() -> None:
    """Gate 3 runs this over thousands of regions. Counts add; percentages are recomputed from the
    pooled counts, NEVER averaged.

    The regions here differ in BOTH size and rate, deliberately. An earlier version of this test
    pooled two identical regions, where pooling and averaging give the same answer — so replacing
    the pooled percentage with `np.mean(per_region_percentages)` passed a test named after exactly
    that distinction. Region A is 4x4 with 2 of 16 columns multi-run (12.5%); region B is 8x8 with
    32 of 64 columns multi-run (50%). The pooled rate is 34/80 = 42.5%; the mean of the rates is
    31.25%. Only one of those is right."""
    a = _flat_band(4, 4, 10, surface_i=6)
    a[1, 1:3, 3:6] = 0                                   # 2 multi-run columns of 16
    ma = compute_overhang_metrics(a, band_height=10)
    b = _flat_band(8, 8, 10, surface_i=6)
    b[:4, :, 3:6] = 0                                    # 32 multi-run columns of 64
    mb = compute_overhang_metrics(b, band_height=10)
    assert ma["multi_run_columns"] == 2 and mb["multi_run_columns"] == 32

    agg = aggregate_overhang_metrics([ma, mb])
    assert agg["num_regions_pooled"] == 2
    assert agg["num_columns"] == 16 + 64 == 80
    assert agg["multi_run_columns"] == 34
    assert agg["multi_run_column_pct"] == pytest.approx(100.0 * 34 / 80)      # 42.5, pooled
    assert agg["multi_run_column_pct"] != pytest.approx(
        (ma["multi_run_column_pct"] + mb["multi_run_column_pct"]) / 2.0)      # 31.25, averaged
    assert agg["overhang_cells"] == ma["overhang_cells"] + mb["overhang_cells"]
    assert agg["overhang_cell_pct"] == pytest.approx(
        100.0 * agg["overhang_cells"] / (80 * 10))
    assert agg["cavity"]["count"] == ma["cavity"]["count"] + mb["cavity"]["count"]
    assert agg["walkable"]["void_count"] == ma["walkable"]["void_count"] + mb["walkable"]["void_count"]
    assert agg["solid_fraction"] == pytest.approx(
        (ma["solid_fraction"] * 160 + mb["solid_fraction"] * 640) / 800)
    # Same schema for one region as for many.
    assert aggregate_overhang_metrics([ma]) == dict(ma, num_regions_pooled=1)


def test_aggregate_pooled_cavity_extremes_come_from_the_right_end() -> None:
    """Pooled `max_height` must be the LAST nonzero bin of the pooled histogram and pooled
    percentiles must come from the pooled counts. Every previous fixture had cavities of a single
    height, so taking the FIRST nonzero bin passed — and this number is what §1.2's band-size
    decision is made from."""
    short = _flat_band(4, 4, 20, surface_i=16)
    short[0, 0:2, 3:6] = 0                       # two cavities of height 3
    m_short = compute_overhang_metrics(short, band_height=20)
    tall = _flat_band(4, 4, 20, surface_i=16)
    tall[0, 0:2, 3:14] = 0                       # two cavities of height 11
    m_tall = compute_overhang_metrics(tall, band_height=20)
    assert m_short["cavity"]["max_height"] == 3 and m_tall["cavity"]["max_height"] == 11

    agg = aggregate_overhang_metrics([m_short, m_tall])
    assert agg["cavity"]["max_height"] == 11                 # last bin, not first
    assert agg["cavity"]["count"] == 4
    assert agg["cavity"]["height_counts"][3] == 2
    assert agg["cavity"]["height_counts"][11] == 2
    assert agg["cavity"]["mean_height"] == pytest.approx(7.0)
    # Nearest-rank over the pooled {3,3,11,11}: p50 is the 2nd value, p90 the 4th.
    assert agg["cavity"]["percentiles"]["p50"] == 3
    assert agg["cavity"]["percentiles"]["p90"] == 11
    assert agg["cavity"][f"taller_than_{BAND_SIZE_WARNING_HEIGHT}"] == 0


def test_aggregate_refuses_to_pool_mixed_floor_spaces() -> None:
    """Void counts measured in world y and in band index differ by up to 24x, so pooling them would
    produce a total that means nothing."""
    band = _carve(_flat_band(4, 4, 12, surface_i=9), 3, 7)
    world = compute_overhang_metrics(band, heightfield=np.zeros((4, 4), dtype=np.float32),
                                     band_height=12)
    index = compute_overhang_metrics(band, band_height=12)
    with pytest.raises(ValueError, match="floor spaces"):
        aggregate_overhang_metrics([world, index])


def test_clipped_at_band_floor_counts_every_clipped_cavity() -> None:
    """`clipped_at_band_floor` is the number that says "the band is too shallow" rather than "the
    voids are big", so it is pinned at an exact count above 1 — a `<= 1` mutant passed before."""
    band = np.zeros((2, 3, 10), dtype=np.uint8)
    band[:, :, 4:8] = 1                    # air 0..3 (clipped at the floor), solid 4..7, air 8..9
    m = compute_overhang_metrics(band, band_height=10)
    assert m["cavity"]["count"] == 6                       # one per column
    assert m["cavity"]["clipped_at_band_floor"] == 6       # all six touch band index 0
    assert m["cavity"]["clipped_at_band_floor_pct"] == pytest.approx(100.0)

    partial = np.zeros((2, 3, 10), dtype=np.uint8)
    partial[:, :, 4:8] = 1
    partial[0, :, 0:1] = 1                 # three columns now have a solid cell at index 0
    m2 = compute_overhang_metrics(partial, band_height=10)
    assert m2["cavity"]["count"] == 6
    assert m2["cavity"]["clipped_at_band_floor"] == 3
    assert m2["cavity"]["clipped_at_band_floor_pct"] == pytest.approx(50.0)


def test_walkable_floor_cells_can_exceed_one_per_column() -> None:
    """One column can hold several tall cavities and contributes a floor cell for each, so the floor
    total is NOT bounded by the column count. It was once reported as a "% of columns" and printed
    300%; it is now a rate per 1000 columns, which cannot be misread as a fraction."""
    band = np.zeros((2, 2, 20), dtype=np.uint8)
    for lo in (3, 8, 13):
        band[:, :, lo:lo + 1] = 1          # solid separators, so each column has three tall cavities
    band[:, :, 18:20] = 1
    m = compute_overhang_metrics(band, band_height=20)
    # Air runs per column: 0-2, 4-7, 9-12, 14-17 — four roofed cavities, all >= 3 tall.
    assert m["cavity"]["count"] == 16                       # 4 cavities x 4 columns
    assert m["walkable"]["void_count"] == 4                 # one component per floor level
    assert m["walkable"]["floor_cells"] == 16               # 4 floor cells per column
    assert m["walkable"]["floor_cells_per_1000_columns"] == pytest.approx(4000.0)
    assert "floor_cell_pct" not in m["walkable"]


def test_heightfield_band_is_column_independent_and_reads_zero() -> None:
    """`heightfield_band` is the v1.1.1 fallback, at the real BAND_HEIGHT. The anchor cancels out of
    the band-space form of `y <= h`, so every column is identical — which is exactly why a
    heightfield world has no 3D structure to measure."""
    h = np.array([[10.0, -40.0], [200.0, 0.0]], dtype=np.float32)
    band = heightfield_band(h)
    assert band.shape == (2, 2, BAND_HEIGHT)
    assert int(band[0, 0].sum()) == BAND_HEIGHT - BAND_TOP_OFFSET
    for z in range(2):
        for x in range(2):
            assert np.array_equal(band[z, x], band[0, 0])   # the anchor cancels

    m = compute_overhang_metrics(band)
    assert m["overhang_cells"] == 0
    assert m["multi_run_columns"] == 0
    assert m["cavity"]["count"] == 0
    assert m["walkable"]["void_count"] == 0


def test_normalize_band_rejects_bad_rank() -> None:
    with pytest.raises(ValueError, match="must be 3D"):
        _normalize_band(np.ones((4, 4), dtype=np.uint8), band_height=8)
    with pytest.raises(ValueError, match="axis 1"):
        _normalize_band(np.ones((4, 7, 2, 2), dtype=np.uint8), band_height=8)


def test_overhang_diagnostic_plot_writes_a_file(tmp_path) -> None:
    """The figure is driven entirely by metrics dicts, so it must render for a Gate 0 run (no model)
    as well as a paired one."""
    band = _flat_band(6, 6, 12, surface_i=8)
    band[2, 2:5, 3:7] = 0
    gt = compute_overhang_metrics(band, band_height=12)
    baseline = compute_overhang_metrics(marginal_baseline_band(band, band_height=12), band_height=12)

    p0 = tmp_path / "gate0.png"
    save_diagnostic_plot(gt, None, baseline, str(p0))
    assert p0.is_file() and p0.stat().st_size > 0

    p2 = tmp_path / "gate2.png"
    save_diagnostic_plot(gt, gt, baseline, str(p2))
    assert p2.is_file() and p2.stat().st_size > 0


def test_exact_half_boundaries_are_inclusive() -> None:
    """A DEAD OCCUPANCY HEAD EMITS EXACTLY 0.0, and an untrained one emits values that sit on the
    threshold constantly, so the `>=` vs `>` boundary is the realistic case and not a corner case.
    Both `>=`-vs-`>` mutants (the float threshold and the baseline marginal) passed before this."""
    # Float band exactly at the threshold counts as SOLID.
    at_half = np.full((2, 2, 8), 0.5, dtype=np.float32)
    m = compute_overhang_metrics(at_half, band_height=8, threshold=0.5)
    assert m["solid_fraction"] == 1.0
    assert compute_overhang_metrics(at_half, band_height=8, threshold=0.5000001)["solid_fraction"] == 0.0
    # Logits of exactly 0.0 at the logit threshold: the same rule, the same direction.
    zeros = np.zeros((2, 2, 8), dtype=np.float32)
    assert compute_overhang_metrics(zeros, band_height=8, threshold=0.0)["solid_fraction"] == 1.0

    # Baseline marginal of exactly 0.5 also counts as solid. Two columns solid, two air, at every
    # index: the marginal is exactly 0.5 everywhere and the template must be all-solid.
    band = np.zeros((2, 2, 4), dtype=np.uint8)
    band[0, :, :] = 1
    m2 = compute_overhang_metrics(band, band_height=4)
    assert m2["solid_fraction_by_band_index"] == [0.5, 0.5, 0.5, 0.5]
    assert marginal_baseline_band(band, band_height=4).all()


def test_marginal_baseline_from_profile_is_what_main_uses() -> None:
    """`main` builds the baseline from the POOLED profile via this function, not from the first
    region via `marginal_baseline_band` — so this is the one that has to be right."""
    tiled = marginal_baseline_from_profile([0.9, 0.6, 0.5, 0.4, 0.0], (3, 2))
    assert tiled.shape == (3, 2, 5)
    expected = np.array([True, True, True, False, False])
    for z in range(3):
        for x in range(2):
            assert np.array_equal(tiled[z, x], expected)
    # Agrees with the whole-band entry point on the same data.
    band = np.zeros((2, 2, 4), dtype=np.uint8)
    band[:, :, 0:2] = 1
    band[1, 1, :] = 1
    m = compute_overhang_metrics(band, band_height=4)
    assert np.array_equal(
        marginal_baseline_from_profile(m["solid_fraction_by_band_index"], (2, 2)),
        marginal_baseline_band(band, band_height=4),
    )


def _write_shard(path, band=None, heightfield=None) -> str:
    arrays = {}
    if band is not None:
        arrays["band"] = band
    if heightfield is not None:
        arrays["heightfield"] = heightfield
    np.savez_compressed(str(path), **arrays)
    return str(path)


def test_load_band_returns_the_heightfield_it_used_to_discard(tmp_path) -> None:
    """The heightfield sits in the shard next to the band and is what makes the walkable-void metric
    world-y-correct instead of 24x inflated. It was being opened and thrown away."""
    band = _carve(_flat_band(4, 4, BAND_HEIGHT, surface_i=40), 20, 24)
    hf = np.full((4, 4), 30.0, dtype=np.float32)
    loaded = load_band(_write_shard(tmp_path / "region_00000.npz", band, hf))
    assert loaded.heightfield is not None
    assert np.array_equal(loaded.heightfield, hf)
    assert loaded.is_heightfield_fallback is False
    assert "key 'band'" in loaded.source
    # ...and it is actually used, i.e. the metric reads world_y rather than band_index.
    m = compute_overhang_metrics(loaded.band, heightfield=loaded.heightfield)
    assert m["walkable"]["floor_space"] == "world_y"


def test_load_band_flags_the_heightfield_fallback_as_a_boolean(tmp_path) -> None:
    """A shard with no band array yields the degenerate v1.1.1 volume. The flag is a BOOLEAN, not a
    substring of the source string: the substring check only ever ran on the first ground-truth
    shard, so a stale shard later in a directory printed no warning, and a stale shard passed as
    --model printed 0.00% retention that reads as "the model learned nothing"."""
    hf = np.full((4, 4), 30.0, dtype=np.float32)
    loaded = load_band(_write_shard(tmp_path / "region_00000.npz", None, hf))
    assert loaded.is_heightfield_fallback is True
    assert "DERIVED-HEIGHTFIELD-FALLBACK" in loaded.source
    m = compute_overhang_metrics(loaded.band, heightfield=loaded.heightfield)
    assert m["overhang_cells"] == 0 and m["multi_run_columns"] == 0

    # A raw .npy has no heightfield at all, and says so rather than pretending.
    npy = tmp_path / "model_band.npy"
    np.save(str(npy), _flat_band(4, 4, BAND_HEIGHT, surface_i=40))
    raw = load_band(str(npy))
    assert raw.heightfield is None and raw.is_heightfield_fallback is False
    assert "no heightfield" in raw.source

    with pytest.raises(KeyError):
        load_band(_write_shard(tmp_path / "region_00001.npz", None, None))


def test_resolve_band_paths_does_not_sweep_npy_out_of_a_shard_directory(tmp_path) -> None:
    """A diagnostic array dropped in a shard directory would silently change the denominator of
    every percentage in the report."""
    hf = np.full((2, 2), 10.0, dtype=np.float32)
    _write_shard(tmp_path / "region_00000.npz", None, hf)
    _write_shard(tmp_path / "region_00001.npz", None, hf)
    np.save(str(tmp_path / "some_diagnostic.npy"), np.zeros((2, 2, BAND_HEIGHT), dtype=np.uint8))

    paths = resolve_band_paths(str(tmp_path))
    assert len(paths) == 2
    assert all(p.endswith(".npz") for p in paths)
    assert resolve_band_paths(str(tmp_path), max_regions=1) == paths[:1]
    assert resolve_band_paths(str(tmp_path / "region_00000.npz")) == [str(tmp_path / "region_00000.npz")]
    with pytest.raises(FileNotFoundError):
        resolve_band_paths(str(tmp_path / "nope"))


def test_normalize_band_honours_a_non_square_chunk_grid() -> None:
    """B alone cannot tell a 2x8 strip from a 4x4 block — both have B=16 — and assembling a strip as
    a block fabricates adjacency at four seams inside a connectivity metric."""
    rng = np.random.default_rng(3)
    chunks = rng.integers(0, 2, size=(16, 8, 2, 2)).astype(np.uint8)
    strip = _normalize_band(chunks, band_height=8, chunk_grid=(2, 8))
    assert strip.shape == (4, 16, 8)
    block = _normalize_band(chunks, band_height=8)
    assert block.shape == (8, 8, 8)
    # Same cells, different neighbours: the metric that depends on adjacency must differ.
    assert strip.sum() == block.sum()
    m_strip = compute_overhang_metrics(chunks, band_height=8, chunk_grid=(2, 8))
    assert m_strip["grid_shape"] == [4, 16]
    with pytest.raises(ValueError, match="does not describe"):
        _normalize_band(chunks, band_height=8, chunk_grid=(3, 5))


def test_print_report_runs_for_gate0_and_gate2(capsys) -> None:
    """`print_report` was entirely untested while carrying every number a human reads. Smoke-check
    both paths and pin the two labels that must not be confused."""
    band = _carve(_flat_band(6, 6, 12, surface_i=9), 3, 7)
    gt = compute_overhang_metrics(band, band_height=12)
    baseline = compute_overhang_metrics(marginal_baseline_band(band, band_height=12), band_height=12)

    gate0 = evaluate_baseline_gate(gt, baseline, 5.0)
    print_report(gt, None, baseline, compute_retention(gt, None), gate0)
    out = capsys.readouterr().out
    assert "GATE 0 DATA CHECK (not a Gate 2 verdict)" in out
    assert "DATA CHECK:" in out
    assert "VERDICT:" not in out              # the skimmer must not read a Gate 2 verdict here
    assert "undefined" in out                 # retention column with no model

    gate2 = evaluate_baseline_gate(gt, baseline, 5.0, gt_metrics=gt)
    print_report(gt, gt, baseline, compute_retention(gt, gt), gate2)
    out2 = capsys.readouterr().out
    assert "GATE 2 GO/NO-GO" in out2
    assert "VERDICT:" in out2
    assert "relief retention" in out2         # §7's fourth condition, measured elsewhere
    assert "1. multi-run columns" in out2 and "2. walkable-void retention" in out2
    assert "3. p90 cavity height" in out2


def test_band_index_run_warns_in_the_report(capsys) -> None:
    """A band-index void count is an inflated upper bound and the report has to say so where the
    number is printed, not only in a docstring."""
    band = _carve(_flat_band(6, 6, 12, surface_i=9), 3, 7)
    m = compute_overhang_metrics(band, band_height=12)
    assert m["walkable"]["floor_space"] == "band_index"
    baseline = compute_overhang_metrics(marginal_baseline_band(band, band_height=12), band_height=12)
    print_report(m, None, baseline, compute_retention(m, None),
                 evaluate_baseline_gate(m, baseline, 5.0))
    out = capsys.readouterr().out
    assert "INFLATED UPPER BOUND" in out

    hf = np.zeros((6, 6), dtype=np.float32)
    m_world = compute_overhang_metrics(band, heightfield=hf, band_height=12)
    print_report(m_world, None, baseline, compute_retention(m_world, None),
                 evaluate_baseline_gate(m_world, baseline, 5.0))
    assert "INFLATED UPPER BOUND" not in capsys.readouterr().out


def test_main_gate0_and_gate2_end_to_end(tmp_path, capsys, monkeypatch) -> None:
    """`main` was untested, and every should-fix the audit found lived in it. Drives the two
    documented invocations end to end and checks the JSON they write."""
    import sys as _sys
    from mc_imagine_model.scripts import diagnose_overhangs as tool

    band = _carve(_flat_band(8, 8, BAND_HEIGHT, surface_i=40), 20, 24)
    hf = np.full((8, 8), 30.0, dtype=np.float32)
    shard = _write_shard(tmp_path / "region_00000.npz", band, hf)
    out_json = tmp_path / "gate0.json"

    monkeypatch.setattr(_sys, "argv", ["diagnose_overhangs.py", "--ground-truth", shard,
                                       "--output", str(out_json)])
    assert tool.main() == 0
    text = capsys.readouterr().out
    assert "GATE 0 DATA CHECK" in text
    report = json.loads(out_json.read_text())
    assert report["floor_space"] == "world_y"
    assert report["ground_truth_is_heightfield_fallback"] is False
    assert report["model"] is None
    assert report["ground_truth"]["num_regions_pooled"] == 1
    assert set(report["retention"].values()) == {None}
    assert report["gate"]["is_gate2_verdict"] is False

    # Gate 2: the same command with a --model argument. The model band is a .npy with no
    # heightfield, so both sides must be re-measured in band-index space rather than mixed.
    model_npy = tmp_path / "model_band.npy"
    np.save(str(model_npy), band)
    out2 = tmp_path / "gate2.json"
    monkeypatch.setattr(_sys, "argv", ["diagnose_overhangs.py", "--ground-truth", shard,
                                       "--model", str(model_npy), "--output", str(out2)])
    assert tool.main() == 0
    text2 = capsys.readouterr().out
    assert "MODEL OUTPUT HAS NO HEIGHTFIELD" in text2
    assert "GATE 2 GO/NO-GO" in text2
    r2 = json.loads(out2.read_text())
    assert r2["floor_space_downgraded_to_match_model"] is True
    assert r2["ground_truth"]["walkable"]["floor_space"] == "band_index"
    assert r2["model"]["walkable"]["floor_space"] == "band_index"
    # Same band on both sides, measured the same way -> retention is exactly 1.0 everywhere.
    assert all(v == pytest.approx(1.0) for v in r2["retention"].values())

    # ...and with --model-heightfield both sides stay in world y.
    mh = tmp_path / "model_h.npy"
    np.save(str(mh), hf)
    out3 = tmp_path / "gate2_world.json"
    monkeypatch.setattr(_sys, "argv", ["diagnose_overhangs.py", "--ground-truth", shard,
                                       "--model", str(model_npy), "--model-heightfield", str(mh),
                                       "--output", str(out3)])
    assert tool.main() == 0
    capsys.readouterr()
    r3 = json.loads(out3.read_text())
    assert r3["floor_space_downgraded_to_match_model"] is False
    assert r3["model"]["walkable"]["floor_space"] == "world_y"


def test_main_reports_a_stale_model_shard_instead_of_blaming_the_model(tmp_path, capsys, monkeypatch) -> None:
    """A v1.1.1 shard passed as --model has no band and derives the degenerate one, giving 0%
    retention that reads as 'the model learned nothing'. That is docs/phase4-plan.md §5.1's silent
    downgrade arriving through the diagnostic, and it must be labelled loudly."""
    import sys as _sys
    from mc_imagine_model.scripts import diagnose_overhangs as tool

    band = _carve(_flat_band(8, 8, BAND_HEIGHT, surface_i=40), 20, 24)
    hf = np.full((8, 8), 30.0, dtype=np.float32)
    gt_shard = _write_shard(tmp_path / "region_00000.npz", band, hf)
    stale = _write_shard(tmp_path / "stale.npz", None, hf)

    monkeypatch.setattr(_sys, "argv", ["diagnose_overhangs.py", "--ground-truth", gt_shard,
                                       "--model", stale])
    assert tool.main() == 0
    text = capsys.readouterr().out
    assert "NO BAND ARRAY IN AT LEAST ONE MODEL SHARD" in text
    assert "silent downgrade" in text


def test_logit_band_at_the_probability_threshold_warns(capsys) -> None:
    """Thresholding logits at 0.5 biases toward air, i.e. toward a PASS — the one direction a
    diagnostic must never be quietly wrong in."""
    logits = np.linspace(-7.2, 8.4, 2 * 2 * 8, dtype=np.float32).reshape(2, 2, 8)
    compute_overhang_metrics(logits, band_height=8)
    assert "look like LOGITS" in capsys.readouterr().err
    compute_overhang_metrics(logits, band_height=8, threshold=0.0)
    assert "look like LOGITS" not in capsys.readouterr().err
    probs = np.full((2, 2, 8), 0.7, dtype=np.float32)
    compute_overhang_metrics(probs, band_height=8)
    assert capsys.readouterr().err == ""


def test_render_band_area_returns_known_layout_and_heightmap() -> None:
    """REGRESSION: `--checkpoint` at its own default `--chunk-span 4` assembles a
    [64, 64, 64] volume, which layout auto-detection correctly calls ambiguous — so the inference
    path died with an uncaught ValueError before reaching a single metric. The layout is KNOWN here
    and is passed explicitly; the heightmap comes back too, so the model side can be measured in
    world y like the ground-truth side."""
    from mc_imagine_model.scripts.diagnose_overhangs import render_band_area

    class _StubNet:
        """Emits the occupancy head's shapes without the head existing yet (Task B1)."""
        def eval(self):
            return self

        def __call__(self, tokens, cx, cz, seeds):
            b = int(tokens.shape[0])
            logits = torch.full((b, BAND_HEIGHT, 16, 16), -1.0)
            logits[:, :40, :, :] = 1.0                     # solid below, air above
            return {"occupancy_logits": logits,
                    "heightmap": torch.full((b, 16, 16), 42.0)}

    band, hf = render_band_area(_StubNet(), "a cliff", seed=1, chunk_span=4)
    assert band.shape == (64, 64, BAND_HEIGHT)             # the ambiguous cube, handled
    assert hf is not None and hf.shape == (64, 64)
    m = compute_overhang_metrics(band, heightfield=hf, layout="zxi")
    assert m["walkable"]["floor_space"] == "world_y"
    assert m["multi_run_columns"] == 0                     # a heightfield volume, correctly read

    class _OldNet(_StubNet):
        def __call__(self, tokens, cx, cz, seeds):
            return {"heightmap": torch.zeros((int(tokens.shape[0]), 16, 16))}

    with pytest.raises(KeyError, match="no occupancy output"):
        render_band_area(_OldNet(), "a cliff", seed=1, chunk_span=2)
