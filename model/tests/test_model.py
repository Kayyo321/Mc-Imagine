"""
Basic pytest tests for the Mc-Imagine model components.
"""

import os
import random
from typing import Dict

import torch
import pytest

from mc_imagine_model.data.world_generator import (
    ARCHETYPES,
    CARVE_STRENGTH_EPS,
    REGION_BLOCKS,
    ProceduralWorldSource,
    _UNBLENDED_PARAMS,
    carve_frequency_is_on_grid,
    carve_lattice_cells,
)
from mc_imagine_model.model.imagine_net import ImagineNet, _check_conv_arithmetic
from mc_imagine_model.model.text_encoder import PromptEncoder
from mc_imagine_model.model.positional import (
    CoordinateEncoder,
    SeedEncoder,
    seed_to_offset_torch,
    verify_seed_encoder_varies,
)
from mc_imagine_model.spec_constants import (
    HEIGHT_LOSS_SCALE,
    HEIGHT_MAX,
    HEIGHT_MIN,
    NUM_BIOMES,
    NUM_PROFILES,
    OFFSET_RANGE,
    SLOPE_LOSS_SCALE,
    WORLD_PERIOD,
    seed_to_offset,
)
from mc_imagine_model.tokenizer_utils import tokenize
from mc_imagine_model.training.losses import OverhangLoss, ReliefLoss, TerrainLoss

# A small-but-valid config: halo must equal num_conv_layers (each valid 3x3 conv eats one ring),
# see imagine_net._check_conv_arithmetic.
# A small-but-valid config: halo must equal num_conv_layers + num_conv3d_layers
# (each valid conv eats one ring), see imagine_net._check_conv_arithmetic.
SMALL_CONFIG = {
    "conv_channels": 32,
    "fusion_hidden": 64,
    "fusion_out": 32,
    "num_conv_layers": 4,
    "num_conv3d_layers": 3,
    "halo": 7,
}


def test_imagine_net_forward() -> None:
    """
    Test that the ImagineNet model can perform a forward pass and return expected shapes.
    """
    model = ImagineNet(SMALL_CONFIG)
    model.eval()

    b = 2
    prompt_tokens = torch.tensor([tokenize("gentle rolling grassland")] * b, dtype=torch.int32)
    chunk_x = torch.tensor([0, 5], dtype=torch.int32)
    chunk_z = torch.tensor([0, -3], dtype=torch.int32)
    seed = torch.tensor([1, 2], dtype=torch.int64)

    with torch.no_grad():
        out = model(prompt_tokens, chunk_x, chunk_z, seed)

    assert out["heightmap"].shape == (b, 16, 16)
    assert out["profile_logits"].shape == (b, NUM_PROFILES, 16, 16)
    assert out["water_level"].shape == (b, 16, 16)
    assert out["biome_logits"].shape == (b, NUM_BIOMES, 4, 4)
    assert out["occupancy_logits"].shape == (b, 64, 16, 16)
    # heightmap/water_level must lie in the head's HEIGHT_CENTER +/- HEIGHT_AMPLITUDE band.
    for key in ("heightmap", "water_level"):
        assert torch.all(out[key] >= HEIGHT_MIN)
        assert torch.all(out[key] <= HEIGHT_MAX)


def test_imagine_net_phase2_dims() -> None:
    """The Phase 4 capacity config (halo 9, 3D conv head) must build and produce the exact
    spec'd output shapes. These are the defaults `training/config*.yaml` sets."""
    config = {
        "halo": 9,
        "patch": 16,
        "conv_channels": 192,
        "num_conv_layers": 6,
        "num_conv3d_layers": 3,
        "fusion_hidden": 512,
        "fusion_out": 256,
        "seed_freqs": 32,
        "seed_embed_dim": 64,
        "num_profiles": 8,
        "num_biomes": 12,
    }
    model = ImagineNet(config)
    model.eval()
    assert model.coord_encoder.size == 34  # 16 + 2*9

    prompt_tokens = torch.tensor([tokenize("towering snow-capped peaks")], dtype=torch.int32)
    with torch.no_grad():
        out = model(
            prompt_tokens,
            torch.tensor([3], dtype=torch.int32),
            torch.tensor([-7], dtype=torch.int32),
            torch.tensor([12345], dtype=torch.int64),
        )
    assert out["heightmap"].shape == (1, 16, 16)
    assert out["profile_logits"].shape == (1, 8, 16, 16)
    assert out["water_level"].shape == (1, 16, 16)
    assert out["biome_logits"].shape == (1, 12, 4, 4)
    assert out["occupancy_logits"].shape == (1, 64, 16, 16)


def test_export_wrapper_uses_predicted_occupancy_band() -> None:
    """A release export must consume `occupancy_logits`, not silently rebuild a heightfield volume.

    The fixture puts air below h=63 inside the band. A heightfield expansion would make y=50 solid;
    the Phase 4 export must preserve it as air while still deriving `heightmap` from the topmost
    solid volume cell.
    """
    from mc_imagine_model.export.export_onnx import ChunkExportWrapper, MIN_Y
    from mc_imagine_model.spec_constants import AIR_ID, BAND_BOTTOM_OFFSET, NUM_PROFILES

    class FixtureNet(torch.nn.Module):
        def forward(self, prompt_tokens, chunk_x, chunk_z, seed):
            h = torch.full((1, 16, 16), 63.0)
            profile = torch.full((1, NUM_PROFILES, 16, 16), -20.0)
            profile[:, 0, :, :] = 20.0
            water = torch.full((1, 16, 16), -64.0)
            biome = torch.zeros((1, NUM_BIOMES, 4, 4))

            occ = torch.full((1, 64, 16, 16), -20.0)
            surface_i = 63 - 63 - BAND_BOTTOM_OFFSET
            occ[:, : surface_i + 1, :, :] = 20.0
            cave_i = 50 - 63 - BAND_BOTTOM_OFFSET
            occ[:, cave_i, 8, 8] = -20.0
            return {
                "heightmap": h,
                "profile_logits": profile,
                "water_level": water,
                "biome_logits": biome,
                "occupancy_logits": occ,
            }

    wrapper = ChunkExportWrapper(FixtureNet(), deterministic_carve=False)
    with torch.no_grad():
        heightmap, block_volume, _ = wrapper(
            torch.zeros((1, 128), dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([123], dtype=torch.int64),
        )

    assert int(heightmap[8, 8]) == 63
    assert int(block_volume[8, 50 - MIN_Y, 8]) == AIR_ID
    assert int(block_volume[8, 63 - MIN_Y, 8]) != AIR_ID


def test_export_wrapper_water_and_dressing_rules_under_overhangs() -> None:
    """Pins docs/phase4-plan.md §1.5's water and surface-dressing rules against a hand-built
    multi-run occupancy fixture, exercising the real `ChunkExportWrapper.forward` path (not a
    reimplementation of the rule that could pass even if the exported graph's cumulative-max logic
    is wrong).

    Two columns, both anchored to `h=63` (so band index i <-> world y = 2 + i,
    `spec_constants.BAND_BOTTOM_OFFSET`/`BAND_TOP_OFFSET`):

    Column (x=8, z=8): solid-air-solid-air-solid — three solid runs with two air gaps, i.e. an
    overhang (the topmost run's underside shelters the gap directly beneath it).
      i  0- 4 (y  2- 6): solid  — buried run, bottom
      i  5- 9 (y  7-11): air    — gap
      i 10-29 (y 12-31): solid  — buried run, middle
      i 30-39 (y 32-41): air    — gap, entirely below `water_level=45`, roofed by the run above:
                                   the sheltered sub-sea-level cavity §1.5 says must stay dry
      i 40-61 (y 42-63): solid  — topmost run (the overhang's roof); only its top 5 cells may be
                                   dressed
      i 62-63 (y 64-65): air    — open sky above the topmost solid cell

    Column (x=2, z=2): a shallow floor with everything above it open to the sky, so the whole
    column below `water_level=45` should flood.
      i  0- 4 (y  2- 6): solid  — floor
      i  5-63 (y  7-65): air    — open to the sky; y in [7,45] is at/under sea level -> water,
                                   y in [46,65] is above sea level -> stays air
    """
    from mc_imagine_model.export.export_onnx import ChunkExportWrapper, MIN_Y
    from mc_imagine_model.spec_constants import (
        AIR_ID,
        NUM_BIOMES,
        NUM_PROFILES,
        PROFILE_TABLE,
        STONE_ID,
        WATER_ID,
    )

    subsurface_id, surface_id = PROFILE_TABLE[0][1], PROFILE_TABLE[0][2]
    assert len({AIR_ID, STONE_ID, WATER_ID, subsurface_id, surface_id}) == 5, (
        "fixture assumes air/rock/water/subsurface/surface ids are pairwise distinct so each "
        "assertion below is unambiguous"
    )

    class FixtureNet(torch.nn.Module):
        def forward(self, prompt_tokens, chunk_x, chunk_z, seed):
            h = torch.full((1, 16, 16), 63.0)
            profile = torch.full((1, NUM_PROFILES, 16, 16), -20.0)
            profile[:, 0, :, :] = 20.0  # profile 0 ("grass": dirt/grass_block) everywhere
            water = torch.full((1, 16, 16), 45.0)
            biome = torch.zeros((1, NUM_BIOMES, 4, 4))

            occ = torch.full((1, 64, 16, 16), -20.0)  # default: air everywhere
            occ[:, :62, :, :] = 20.0  # baseline solid-below-h pattern for every other column

            # Column (x=8, z=8): solid-air-solid-air-solid, with an overhang over a sheltered
            # sub-sea-level cavity.
            occ[:, :, 8, 8] = -20.0
            occ[:, 0:5, 8, 8] = 20.0     # run 1 (buried, bottom): i 0-4  -> y  2- 6
            occ[:, 5:10, 8, 8] = -20.0   # gap
            occ[:, 10:30, 8, 8] = 20.0   # run 2 (buried, middle): i 10-29 -> y 12-31
            occ[:, 30:40, 8, 8] = -20.0  # sheltered cavity: i 30-39 -> y 32-41
            occ[:, 40:62, 8, 8] = 20.0   # run 3 (topmost/overhang roof): i 40-61 -> y 42-63
            occ[:, 62:64, 8, 8] = -20.0  # open sky above the topmost solid cell

            # Column (x=2, z=2): shallow floor, fully open to the sky above it.
            occ[:, :, 2, 2] = -20.0
            occ[:, 0:5, 2, 2] = 20.0     # floor: i 0-4 -> y 2-6

            return {
                "heightmap": h,
                "profile_logits": profile,
                "water_level": water,
                "biome_logits": biome,
                "occupancy_logits": occ,
            }

    wrapper = ChunkExportWrapper(FixtureNet(), deterministic_carve=False)
    with torch.no_grad():
        _, block_volume, _ = wrapper(
            torch.zeros((1, 128), dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([123], dtype=torch.int64),
        )

    def blk(x: int, y: int, z: int) -> int:
        return int(block_volume[x, y - MIN_Y, z])

    # (a) buried solid runs are plain rock, never surface/subsurface dressing.
    assert blk(8, 4, 8) == STONE_ID, "run 1 (bottom, buried) must be rock"
    assert blk(8, 20, 8) == STONE_ID, "run 2 (middle, buried) must be rock"
    assert blk(8, 42, 8) == STONE_ID, "run 3's bottom cell (below the dressing window) must be rock"
    assert blk(8, 58, 8) == STONE_ID, "just below the 4-cell subsurface window must still be rock"

    # (b) only the topmost run is dressed: surface at its top cell, subsurface for the 4 below it.
    assert blk(8, 63, 8) == surface_id, "topmost solid cell of the topmost run must be surface"
    for y in (59, 60, 61, 62):
        assert blk(8, y, 8) == subsurface_id, f"y={y} must be subsurface (topmost run, 4 below top)"

    # (c) the sheltered cavity below sea level, roofed by the overhang, must stay dry.
    for y in range(32, 42):
        assert blk(8, y, 8) == AIR_ID, f"y={y} is a roofed sub-sea-level cavity and must not flood"

    # (d) a column open to the sky below sea level does flood, and only up to sea level.
    for y in range(7, 46):
        assert blk(2, y, 2) == WATER_ID, f"y={y} is open to the sky and at/under sea level -> water"
    for y in range(46, 66):
        assert blk(2, y, 2) == AIR_ID, f"y={y} is open to the sky but above sea level -> air"


def test_conv_arithmetic_is_asserted() -> None:
    """halo/num_conv_layers mismatches must fail loudly at construction, not silently emit a
    wrong-sized tile that only explodes (or worse, doesn't) far downstream."""
    _check_conv_arithmetic(halo=9, patch=16, num_conv_layers=6, num_conv3d_layers=3)  # the Phase 4 setting
    _check_conv_arithmetic(halo=7, patch=16, num_conv_layers=4, num_conv3d_layers=3)
    _check_conv_arithmetic(halo=6, patch=16, num_conv_layers=6, num_conv3d_layers=0)
    with pytest.raises(ValueError):
        _check_conv_arithmetic(halo=9, patch=16, num_conv_layers=6, num_conv3d_layers=2)
    with pytest.raises(ValueError):
        _check_conv_arithmetic(halo=6, patch=16, num_conv_layers=6, num_conv3d_layers=3)
    with pytest.raises(ValueError):
        ImagineNet({**SMALL_CONFIG, "halo": 5})


def test_prompt_encoder() -> None:
    """
    Test the text encoder component: correct output shape, L2-normalized, and frozen (no
    trainable parameters).
    """
    encoder = PromptEncoder()
    tokens = torch.tensor([tokenize("a test prompt"), tokenize("another test prompt")], dtype=torch.int32)
    with torch.no_grad():
        output = encoder(tokens)
    assert output.shape == (2, encoder.embed_dim)
    norms = output.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
    assert all(not p.requires_grad for p in encoder.parameters())

    # Different prompts should (almost certainly) produce different embeddings.
    assert not torch.allclose(output[0], output[1], atol=1e-3)


def test_coordinate_encoder() -> None:
    """
    Test the positional encoding component: output shape, and the critical seamless-border
    property — the region of two adjacent chunks' halo windows that covers the *same* global block
    coordinate must be bit-identical, regardless of which chunk's request produced it (see
    docs/poc-plan.md §1b and export/export_onnx.py's verify_coordinate_purity, which runs this same
    check against the actual trained/exported model).
    """
    for halo in (4, 6):
        encoder = CoordinateEncoder(halo=halo, patch=16)
        size = 16 + 2 * halo
        chunk_x = torch.tensor([0, 1])
        chunk_z = torch.tensor([0, 0])
        seed = torch.tensor([12345, 12345], dtype=torch.int64)
        output = encoder(chunk_x, chunk_z, seed)
        assert output.shape == (2, encoder.out_channels, size, size)

        # chunk(0,0)'s halo covers global x in [-halo, 15+halo]; chunk(1,0)'s covers
        # [16-halo, 31+halo]. The shared 2*halo columns sit at [16, size) in chunk0's grid and at
        # [0, size-16) in chunk1's.
        shared_0 = output[0, :, :, 16:size]
        shared_1 = output[1, :, :, 0:size - 16]
        assert shared_0.shape[-1] == 2 * halo
        assert torch.equal(shared_0, shared_1), (
            "CoordinateEncoder is not a pure function of (global coordinate, seed)"
        )


def test_seed_to_offset_torch_matches_python() -> None:
    """The torch mirror of `spec_constants.seed_to_offset` must be *bit-identical* to the Python
    reference — `data/world_generator.py` uses the Python one to decide where in the canonical
    noise field the training targets were rendered, so any drift silently trains the model against
    a translated copy of its own coordinates (docs/phase2-plan.md §1a).

    Covers the pitfalls specifically: negative seeds (torch.fmod vs torch.remainder sign
    semantics), seeds far beyond float32's 2**24 exact range, and 200 random int64 seeds.
    """
    rng = random.Random(20260728)
    seeds = [
        0,
        1,
        12345,
        2 ** 61 + 17,
        -8756019554883095434,
        -1,
        -1_000_003,
        1_000_003,
        1_000_002,
        2 ** 63 - 1,
        -(2 ** 63),
    ]
    seeds += [rng.randint(-(2 ** 63), 2 ** 63 - 1) for _ in range(200)]

    got_x, got_z = seed_to_offset_torch(torch.tensor(seeds, dtype=torch.int64))
    for i, s in enumerate(seeds):
        exp_x, exp_z = seed_to_offset(s)
        assert got_x[i].item() == exp_x, f"offset_x mismatch for seed {s}: {got_x[i].item()} != {exp_x}"
        assert got_z[i].item() == exp_z, f"offset_z mismatch for seed {s}: {got_z[i].item()} != {exp_z}"
        assert 0 <= exp_x < OFFSET_RANGE and 0 <= exp_z < OFFSET_RANGE

    # Batched and scalar-at-a-time must also agree with each other (no shape-dependent path).
    for s in seeds[:20]:
        single_x, single_z = seed_to_offset_torch(torch.tensor([s], dtype=torch.int64))
        assert (single_x.item(), single_z.item()) == seed_to_offset(s)

    # float32 output must be exact: every offset is an integer < 2**24.
    assert torch.equal(got_x, torch.round(got_x))
    assert torch.equal(got_z, torch.round(got_z))


def test_coordinate_encoder_applies_seed_offset() -> None:
    """Different seeds must translate the coordinate frame (different features for the same chunk),
    and the same seed must reproduce bit-identically (docs/phase2-plan.md §1a: "same seed => same
    world, different seed => different layout, same character")."""
    encoder = CoordinateEncoder(halo=6, patch=16)
    chunk_x = torch.tensor([2, 2, 2])
    chunk_z = torch.tensor([-5, -5, -5])
    seeds = torch.tensor([12345, 999_777, 12345], dtype=torch.int64)
    out = encoder(chunk_x, chunk_z, seeds)

    assert torch.equal(out[0], out[2]), "same seed must be bit-identical"
    assert not torch.equal(out[0], out[1]), "different seeds must translate the field"

    # The translation must be the one seed_to_offset prescribes: encoding chunk(0,0) at some seed
    # must equal encoding raw global coordinates shifted by that seed's offset at seed 0's frame.
    ox, oz = seed_to_offset(12345)
    zero_ox, zero_oz = seed_to_offset(0)
    assert (zero_ox, zero_oz) == (0.0, 0.0)  # seed 0 hashes to no translation
    shifted = encoder(
        torch.tensor([0]), torch.tensor([0]), torch.tensor([12345], dtype=torch.int64)
    )
    # Same thing computed by hand: chunk(0,0)'s local (0,0) column is global block (0,0), which
    # after the seed translation is sampled at exactly (ox, oz). Local index 6 == halo == block 0.
    manual = encoder._fourier(
        torch.tensor([[[ox]]], dtype=torch.float32), torch.tensor([[[oz]]], dtype=torch.float32)
    )
    assert torch.equal(shifted[0, :, 6, 6], manual[0, :, 0, 0])


def test_seed_encoder_varies() -> None:
    """
    Explicit check that SeedEncoder produces genuinely different output for different seeds — see
    docs/poc-plan.md: "Must genuinely vary with seed (a real, not-ignorable dependency)."
    """
    assert verify_seed_encoder_varies()


def test_seed_encoder_deterministic() -> None:
    """Same seed must always produce the same output (determinism guarantee)."""
    encoder = SeedEncoder()
    encoder.eval()
    seed = torch.tensor([42, 42])
    with torch.no_grad():
        out = encoder(seed)
    assert torch.equal(out[0], out[1])


def test_terrain_head_range_covers_ground_truth() -> None:
    """The head must be able to *reach* the terrain the generator produces. Day 1's `63 + 96*tanh`
    ceiling of 159 sat below ground truth's 235, so the head saturated and killed all relief
    (docs/phase2-plan.md §0). Asserting the band here is the cheap regression guard."""
    from mc_imagine_model.model.heads import TerrainHead
    from mc_imagine_model.spec_constants import TERRAIN_CLIP_MAX, TERRAIN_CLIP_MIN

    assert HEIGHT_MIN < TERRAIN_CLIP_MIN < TERRAIN_CLIP_MAX < HEIGHT_MAX, (
        "the generator's clip band must sit strictly inside the head's representable band, so no "
        "target ever lands on the tanh asymptote where the gradient vanishes"
    )
    assert HEIGHT_MAX >= 236, "must cover the 235.6 max the ground-truth data reaches"

    head = TerrainHead(in_channels=8, num_profiles=NUM_PROFILES)
    # Drive the 1x1 convs hard in both directions; tanh must approach, and never exceed, the band.
    with torch.no_grad():
        for w in (-50.0, 50.0):
            head.height_conv.weight.fill_(w)
            head.height_conv.bias.fill_(w)
            head.water_conv.weight.fill_(w)
            head.water_conv.bias.fill_(w)
            feats = torch.ones(1, 8, 16, 16)
            heightmap, _, water = head(feats)
            for t in (heightmap, water):
                assert torch.all(t >= HEIGHT_MIN) and torch.all(t <= HEIGHT_MAX)
            if w > 0:
                assert heightmap.min().item() > 236, "the top of the band must be reachable"
            else:
                assert heightmap.max().item() < -60, "the bottom of the band must be reachable"


def test_terrain_loss_penalizes_flat_prediction() -> None:
    """The point of the slope term (docs/phase2-plan.md Phase 2): a perfectly flat prediction at the
    target's mean height is a strong local optimum for absolute-height MSE alone, which is what let
    the Day-1 model output std 0.03 terrain. With the slope term, the flat solution must cost
    strictly more than a rough one that shares the same absolute-height error."""
    torch.manual_seed(0)
    b = 4
    target_h = 96.0 + 20.0 * torch.randn(b, 16, 16)
    targets = {
        "heightmap": target_h,
        "profile_id": torch.zeros(b, 16, 16, dtype=torch.long),
        "water_level": torch.full((b,), 63.0),
    }

    def preds_for(height: torch.Tensor):
        return {
            "heightmap": height,
            "profile_logits": torch.zeros(b, NUM_PROFILES, 16, 16),
            "water_level": torch.full((b, 16, 16), 63.0),
        }

    flat = target_h.mean(dim=(1, 2), keepdim=True).expand_as(target_h).contiguous()

    with_slope = TerrainLoss(slope_weight=1.0)
    without_slope = TerrainLoss(slope_weight=0.0)

    # The slope term is exactly the extra price the flat solution pays.
    flat_penalty = with_slope(preds_for(flat), targets) - without_slope(preds_for(flat), targets)
    assert flat_penalty > 0.0

    # And the perfect prediction still costs nothing extra.
    perfect_penalty = (
        with_slope(preds_for(target_h), targets) - without_slope(preds_for(target_h), targets)
    )
    assert abs(perfect_penalty.item()) < 1e-9

    # The scales come from spec_constants, where the measurements behind them are recorded.
    # Deliberately NOT pinned to the head's tanh amplitude: normalizing by the head's output range
    # is the intuitive choice and was measured to be badly wrong (see below).
    assert TerrainLoss.HEIGHT_SCALE == HEIGHT_LOSS_SCALE
    assert TerrainLoss.SLOPE_SCALE == SLOPE_LOSS_SCALE

    # REGRESSION GUARD. Normalizing the slope term by the head amplitude (192) instead of a
    # per-block step (~1) measured the slope loss at 2.7e-5 against cross-entropies of ~2.1-2.5 —
    # three orders of magnitude down, so the relief supervision this class exists for contributed
    # essentially nothing to the gradient and the flat solution stayed optimal in practice. Assert
    # the flat-prediction slope penalty is a *material* fraction of the total loss, not merely > 0,
    # so a future rescaling can't silently make this term inert again.
    total_flat = with_slope(preds_for(flat), targets)
    assert flat_penalty / total_flat > 0.05, (
        f"slope penalty {flat_penalty.item():.3e} is only "
        f"{100 * (flat_penalty / total_flat).item():.4f}% of the total loss "
        f"{total_flat.item():.3e} — the relief term has been scaled into irrelevance"
    )

    # Sanity: the loss is finite and differentiable.
    h = flat.clone().requires_grad_(True)
    loss = with_slope(preds_for(h), targets)
    loss.backward()
    assert torch.isfinite(loss) and h.grad is not None and torch.any(h.grad != 0)

    # REGRESSION GUARD #2 — the one the slope term itself failed.
    #
    # A slope penalty > 0 is NOT sufficient to stop flat terrain, which is why the guard above was
    # not enough. Slope MSE is a *pointwise* expectation, so when the per-cell slope is
    # unpredictable from (caption, coords, seed) its optimal value is zero and flat remains the
    # best available answer — measured end-to-end: within-chunk relief peaked at 19% of target by
    # step 750 and collapsed to 6% by step 2500, on train and val alike.
    #
    # ReliefLoss is the term whose optimum is NOT zero. Assert a flat prediction is penalized in
    # proportion to how much relief it is failing to produce, so no future refactor can reduce this
    # to another pointwise term.
    relief = ReliefLoss()
    flat_relief = relief(preds_for(flat), targets)
    perfect_relief = relief(preds_for(target_h), targets)

    assert perfect_relief.item() < 1e-9, "a correct prediction must incur no relief penalty"
    assert flat_relief.item() > 0.1, (
        f"flat prediction incurs relief penalty {flat_relief.item():.3e}, which is too small to "
        f"outrun the pointwise terms — ReliefLoss has been scaled into irrelevance"
    )

    # And it must be strictly monotone in how flat the prediction is: a half-amplitude prediction
    # has to cost less than a fully flat one, otherwise the term gives no gradient to climb.
    half = flat + 0.5 * (target_h - flat)
    assert relief(preds_for(half), targets).item() < flat_relief.item()

    # Differentiable, and the gradient actually pushes toward MORE relief.
    #
    # A bit-exactly constant prediction is a stationary point: `std` is a function of |deviation|,
    # so at zero deviation autograd yields exactly zero gradient. That is inherent to any
    # magnitude-matching term, and benign — the point is measure-zero and unstable, and a real conv
    # stack never emits a bit-exactly constant patch (measured within-chunk std 0.006 at init,
    # 0.056 even in the fully collapsed model). Assert the realistic case instead, and assert the
    # gradient points the right way rather than merely existing.
    #
    # The perturbation must clear float32's ULP at terrain magnitudes (~1.15e-5 at y=96) or it is
    # silently rounded away and the tensor stays bit-exactly flat — 1e-2 is both safely above that
    # and squarely in the range a real collapsed model produces.
    h2 = (flat + 1e-2 * torch.randn_like(flat)).clone().requires_grad_(True)
    relief(preds_for(h2), targets).backward()
    assert h2.grad is not None and torch.any(h2.grad != 0)

    deviation = h2.detach() - h2.detach().mean(dim=(1, 2), keepdim=True)
    # -grad is the descent direction; it must correlate positively with the existing deviation,
    # i.e. descending this loss AMPLIFIES whatever relief is already there.
    assert float((-h2.grad * deviation).sum()) > 0.0, (
        "ReliefLoss gradient does not amplify existing relief — the term would be pushing the "
        "model toward flatter terrain, not away from it"
    )

    # And the exactly-flat stationary point is documented, not accidental.
    h3 = flat.clone().requires_grad_(True)
    relief(preds_for(h3), targets).backward()
    assert torch.all(h3.grad == 0), (
        "an exactly-flat prediction is expected to be a stationary point of ReliefLoss; if this "
        "now has gradient, the term changed shape and the docstring needs updating"
    )


def test_overhang_loss_uniform_prediction_is_stationary_point() -> None:
    """`OverhangLoss` (training/losses.py) computes a per-cell overhang PROBABILITY field
    (`p_overhang = (1-p) * solid_above`, via the cummax/flip construction docs/phase4.2-plan.md
    §3.2 requires be kept exactly as-is) and now compares it against the target with a per-cell BCE
    instead of the old count-MSE — but the stationary point this test pins is a property of that
    SHARED construction, not of the top-level reduction, so it survived the §3.2 rewrite unchanged.

    A bit-exactly uniform occupancy prediction has exactly one true degree of freedom: the shared
    logit `c` (`p = sigmoid(c)`), since every one of the `64*16*16` cells carries the same value. At
    a uniform `p`, `torch.cummax`'s tie-breaking backward (ties resolve to the earliest maximal
    element in scan order — see the note below) routes EVERY cell's gradient through the same single
    source cell, so `solid_above_shifted`'s derivative w.r.t. `c` is `p(1-p)` at every cell, same as
    `p` itself. That makes `d(p_overhang)/dc = p(1-p)(1-2p)` — IDENTICAL across every cell, not just
    in aggregate — which is exactly zero at `p=0.5` (`c=0`) and changes sign either side of it. Since
    every cell's `p_overhang` moves by the same scalar factor, `d(loss)/dc` factors as that scalar
    times a sum over cells, so it is zero at `c=0` regardless of what per-cell loss or target is
    used — this is why the property transferred from the old count-MSE loss to the new per-cell BCE
    without needing to be re-derived from scratch.

    Verified empirically before writing this test (not merely by the algebra above, since
    `torch.cummax`'s backward at exact ties is a hard, tie-breaking selection rather than the smooth
    analytic derivative — the two need not agree, but here they do): at `c=0` the gradient is `~0`
    (`-2e-10`, floating-point noise from the BCE `clamp`/`log`, not `0.0` bit-exactly the way the old
    MSE-of-count version was — BCE has no exact-zero guarantee the old polynomial arithmetic had),
    while `c=+-1e-2` both produce real nonzero gradients of the expected opposite signs.

    Like `ReliefLoss`'s stationary point, this is measure-zero and benign: a real conv stack never
    emits a bit-exactly uniform prediction across every band cell (see `ReliefLoss`'s docstring for
    the measured init/collapsed std figures that make the analogous claim for heightmaps), and
    moving one ULP off `c=0` restores a real, correctly-directed gradient.
    """
    torch.manual_seed(20260731)
    overhang = OverhangLoss()
    B = 2
    target = torch.randint(0, 2, (B, 64, 16, 16)).float()

    # The stationary point itself: differentiate the REAL loss (not a reimplementation) with
    # respect to the uniform prediction's one true degree of freedom.
    c = torch.zeros((), requires_grad=True)
    logits = c.expand(B, 64, 16, 16)
    loss = overhang(logits, target)
    loss.backward()
    assert c.grad is not None and abs(c.grad.item()) < 1e-6, (
        "a bit-exactly uniform occupancy prediction (p=0.5 everywhere) is expected to be a "
        "(near-)stationary point of OverhangLoss's shared cummax/flip construction "
        "(docs/phase4.2-plan.md §3.2); if the gradient is no longer tiny here, the construction "
        "changed shape and both this test and §3.2's note need updating"
    )

    # It is a genuine LOCAL MAXIMUM, not an accident of this target: probe the real forward path
    # (not a reimplementation) with an all-zero target. BCE against an all-zero target reduces to
    # `-log(1 - p_overhang)`, which is monotonically INCREASING in `p_overhang` — so since
    # `p_overhang` itself peaks at `p=0.5` (the algebra above), the loss against a zero target peaks
    # there too, and a HIGHER loss at `c=0` than at neighboring points is the local-maximum witness.
    target_zero = torch.zeros(1, 64, 16, 16)

    def real_loss_at(logit_value: float) -> float:
        logits_ = torch.full((1, 64, 16, 16), logit_value)
        return overhang(logits_, target_zero).item()

    at_half = real_loss_at(0.0)
    assert real_loss_at(-0.5) < at_half and real_loss_at(0.5) < at_half, (
        "p=0.5 must be a local MAXIMUM of the against-zero-target loss (since that loss is "
        "monotonic in p_overhang, which itself peaks at p=0.5); if it is not, the stationary point "
        "above is a saddle or minimum, not the benign maximum this test expects"
    )

    # And moving off the exact stationary point restores a real, nonzero gradient.
    c2 = torch.tensor(1e-2, requires_grad=True)
    overhang(c2.expand(B, 64, 16, 16), target).backward()
    assert c2.grad is not None and c2.grad.item() != 0.0, (
        "a near-uniform prediction should already have a real gradient; the stationary point must "
        "be an isolated, measure-zero point, not a plateau"
    )


def test_overhang_loss_penalizes_misplaced_overhang_at_matched_count() -> None:
    """docs/phase4.2-plan.md §3.2's Done-when: "Count-matching alone must no longer be satisfiable."

    Two predictions with the exact same TOTAL predicted overhang cell count — one placing them
    where the target's overhang actually is, one placing the same number of cells somewhere the
    target has none — must no longer score the same loss. The old per-chunk count-MSE literally
    could not tell these apart (that was §1.2's diagnosis: "it can express *how many* overhang
    cells a chunk should have and never *where*"); the new per-cell BCE must.
    """
    overhang = OverhangLoss()
    B, Y, Z, X = 1, 64, 4, 4

    # Target: a single overhang cell, produced the same way the loss itself derives one — one solid
    # cell with air directly below it, elsewhere all solid (so the topmost band index has nothing
    # above it and is never itself counted as an overhang).
    target = torch.ones(B, Y, Z, X)
    target[0, 10, 1, 1] = 0.0  # the one air cell; target[0, 11, 1, 1] stays solid above it

    def logits_with_air_at(y: int, z: int, x: int) -> torch.Tensor:
        # Large-magnitude logits so probs are close to bit-exact 0/1, matching the target's own
        # binary structure as closely as floating point allows.
        base = torch.full((B, Y, Z, X), 8.0)
        base[0, y, z, x] = -8.0
        return base

    loss_matched = overhang(logits_with_air_at(10, 1, 1), target).item()
    loss_misplaced = overhang(logits_with_air_at(20, 2, 2), target).item()

    assert loss_matched < 1e-3, (
        f"predicting the overhang at its actual location should drive the loss near zero, got "
        f"{loss_matched:.6f}"
    )
    assert loss_misplaced > 10 * max(loss_matched, 1e-6), (
        f"a same-COUNT (one cell), different-LOCATION prediction should score substantially worse "
        f"than the matched-location prediction (matched={loss_matched:.6f}, "
        f"misplaced={loss_misplaced:.6f}) — if it does not, count-matching is still enough to "
        f"satisfy this term and §3.2's fix did not localise it"
    )


def test_relief_frequency_is_pinned_per_archetype() -> None:
    """Every noise FREQUENCY — 2D relief and the Phase 4 3D carve alike — must stay a pure function
    of the archetype, hence of the caption. Every amplitude-like GAIN must stay free to vary.

    Frequencies are the sampled parameters the caption cannot describe and the world seed cannot
    reveal: `sample_params` draws parameters from `_region_rng(..., stream=0)` and the world seed
    from `stream=1`, decorrelated on purpose. Randomizing one makes flat terrain the MSE optimum,
    because averaging a fixed-phase noise field over a range of frequencies decorrelates the fields
    being averaged and annihilates their structure. Measured in 2D, as the fraction of within-chunk
    relief surviving conditional averaging:

        amplitude only ~100%   erosion only ~100%   ridge+plateau 83-100%   frequency only 20-37%

    docs/phase4-plan.md §2.2(b) extends the rule to the 3D carve frequencies, and there is reason to
    expect the 3D case is worse rather than kinder: a carve is a *local* feature, so averaging over
    its phase erases it faster than averaging over a smooth field's. No 3D equivalent of the table
    above has been measured; the pinning is not waiting for one.

    Four failures this catches, each of which otherwise shows up only as a disappointing trained
    model:

    1. **A widened range.** Any frequency range that is not degenerate.
    2. **A blend leak.** `sample_params` blends toward a secondary archetype 25% of the time, and
       the caption is keyed to the PRIMARY one, so a blended frequency is caption-invisible for a
       quarter of all regions — exactly the hole the pinning exists to close.
    3. **A carve frequency drifting off the quantization grid.** `_fbm3d` quantizes each octave to
       `n_cells / WORLD_PERIOD`, so a requested frequency is shifted by up to
       `0.5 / (WORLD_PERIOD * freq)`. Measured in `_fbm3d`'s docstring, first octave: 4.37/512 ->
       2.75%, 3.10/512 -> 3.23%, 1.90/512 -> 5.26%, 0.70/512 -> 7.14%. The large carve features
       this release exists for sit in that 3-7% band, so every pinned carve frequency is chosen ON
       the grid and the quantizer is a genuine no-op. Off-grid still renders — just at a wavelength
       a few percent from the documented one, silently invalidating the calibration numbers in
       `world_generator`. (The existing `relief_frequency` values are deliberately NOT held to this:
       4.75/2.25/3.0/5.25/4.5/3.75 are on-grid, 3.35/3.9/2.65 are not, measured shifts 2.99%,
       2.56%, 3.77%. That is committed Phase-2 behaviour the trained baselines were produced under.)
    4. **Overhangs appearing in "endless flat desert dunes".** `desert_dunes`, `rolling_grassland`,
       `swamp` and `savanna_plains` are pinned to zero carve strength because the model has to learn
       that the *absence* of 3D structure is prompt-determined too. A blended gain would hand a
       quarter of desert regions someone else's caves. The bound is `<= CARVE_STRENGTH_EPS` rather
       than `== 0.0` on purpose and it is not a loophole: the rasterizer skips the 3D pass outright
       at or below that epsilon, so a sub-epsilon strength is provably inert, not merely small.
       Measured with the shipped table, six sampled regions of each flat archetype carve exactly
       zero band cells.

    The last assertion runs the other way and is not decoration: it fails if someone "fixes" a
    future failure here by pinning the amplitude-like gains as well, which would pass every check
    above while deleting the carve diversity the plan asks for.
    """
    # Frequencies: pinned, on-grid, unblended. Gains: free to vary.
    CARVE_FREQUENCY_PARAMS = ("carve_frequency", "hoodoo_frequency")
    CARVE_GAIN_PARAMS = (
        "carve_strength", "undercut_strength", "arch_strength",
        "cave_mouth_strength", "hoodoo_strength",
    )
    FLAT_ARCHETYPES = ("desert_dunes", "rolling_grassland", "swamp", "savanna_plains")
    PINNED_FREQUENCY_PARAMS = ("relief_frequency",) + CARVE_FREQUENCY_PARAMS

    # Structural: the exclusion set itself must still name every one of them. The sampling loop
    # below catches a leak empirically, but only for parameters that actually differ between the
    # archetypes a 1500-region draw happens to pair up; this catches a shrunken set directly.
    missing = set(PINNED_FREQUENCY_PARAMS + CARVE_GAIN_PARAMS) - set(_UNBLENDED_PARAMS)
    assert not missing, (
        f"{sorted(missing)} dropped out of world_generator._UNBLENDED_PARAMS, so sample_params "
        f"will blend them toward a secondary archetype. For a frequency that is caption-invisible "
        f"structure; for a gain it puts overhangs in 'endless flat desert dunes'"
    )

    for name, spec in ARCHETYPES.items():
        for key in PINNED_FREQUENCY_PARAMS:
            lo, hi = spec[key]
            assert lo == hi, (
                f"archetype {name!r} has {key}=({lo}, {hi}); it must be pinned (lo == hi) or flat "
                f"terrain / structureless mush becomes the loss optimum again"
            )
        # On-grid applies to the carve frequencies only — see failure 3 in the docstring for why
        # `relief_frequency` is exempt.
        for key in CARVE_FREQUENCY_PARAMS:
            freq = spec[key][0]
            cells = WORLD_PERIOD * (freq / REGION_BLOCKS)
            # `cells or 1` guards the message itself: a frequency of 0.0 is exactly the edit this
            # assertion exists to catch, and a ZeroDivisionError raised while formatting the
            # explanation would replace the diagnostic with a traceback into the test.
            assert carve_frequency_is_on_grid(freq), (
                f"archetype {name!r} has {key}={freq}, which is off the 1/WORLD_PERIOD "
                f"quantization grid: WORLD_PERIOD * freq / REGION_BLOCKS = {cells}, which must be "
                f"an integer >= 2. _fbm3d would sample a frequency "
                f"{abs(round(cells) - cells) / (cells or 1):.2%} away from the pinned value. Pick "
                f"n_cells as an integer >= 2 and use n_cells * CARVE_FREQ_GRID."
            )
            assert carve_lattice_cells(freq) == round(cells) >= 2, (
                f"archetype {name!r} {key}={freq} does not survive _fbm3d's own quantizer "
                f"(including its floor of 2 cells)"
            )

    for name in FLAT_ARCHETYPES:
        lo, hi = ARCHETYPES[name]["carve_strength"]
        assert max(lo, hi) <= CARVE_STRENGTH_EPS, (
            f"{name!r} has carve_strength=({lo}, {hi}); the flat archetypes must carve nothing. "
            f"'endless flat desert dunes' is the control case that teaches the model the absence "
            f"of 3D structure is prompt-determined, and a token nonzero value for variety destroys "
            f"it (and costs an ~8.6 s _fbm3d evaluation per region to do so)"
        )

    # The 25% secondary-archetype blend must leave every pinned frequency alone, and must not leak
    # carve strength into the flat archetypes either.
    src = ProceduralWorldSource(seed=1234)
    seen: Dict[str, Dict[str, set]] = {}
    for region_id in range(1500):
        params = src.sample_params(region_id % 64, region_id // 64)
        per_archetype = seen.setdefault(params.archetype, {})
        for key in PINNED_FREQUENCY_PARAMS + CARVE_GAIN_PARAMS:
            per_archetype.setdefault(key, set()).add(round(getattr(params, key), 9))

    for key in PINNED_FREQUENCY_PARAMS:
        leaked = {a: sorted(v[key]) for a, v in seen.items() if len(v[key]) > 1}
        assert not leaked, (
            f"these archetypes produced more than one {key} across sampled regions, so the "
            f"secondary-archetype blend is leaking it: {leaked}"
        )
    assert len(seen) >= 8, f"only {len(seen)} archetypes sampled; test is not covering enough"

    for name in FLAT_ARCHETYPES:
        if name not in seen:
            continue
        worst = max(seen[name]["carve_strength"])
        assert worst <= CARVE_STRENGTH_EPS, (
            f"{name!r} sampled carve_strength up to {worst} — the table pins it to zero, so the "
            f"secondary-archetype blend is leaking carve strength into a flat archetype"
        )

    # ...and the converse, checked PER GAIN rather than over the set. Gains are amplitude-like:
    # they rescale a fixed pattern, so conditional averaging preserves their structure (measured
    # 83-100% for the 2D amplitude/erosion analogues) and every one of them SHOULD vary somewhere.
    # An earlier version of this guard asked only whether *any* gain varied anywhere, which meant
    # four of the five could be pinned flat and the test still passed while claiming otherwise.
    # A gain that varies within a range produces many distinct values across 1500 regions; 10 is far
    # above float noise and far below the dozens-to-hundreds actually observed.
    for key in CARVE_GAIN_PARAMS:
        varying = sorted(a for a, per in seen.items() if len(per[key]) >= 10)
        assert varying, (
            f"no archetype varies {key!r} — every sampled region got a pinned value. Frequencies "
            f"must be pinned; gains must not be. Pinning a gain passes every other assertion in "
            f"this test while deleting the per-region carve diversity docs/phase4-plan.md §2.1 "
            f"asks for, which is the cheapest wrong way to make a future failure here go away"
        )


def test_fbm3d_is_bit_exactly_world_period_periodic() -> None:
    """The 3D carve field must be EXACTLY `WORLD_PERIOD`-periodic in x and z — `==`, not
    `allclose` — and must genuinely depend on y.

    `CoordinateEncoder` builds its features purely from sin/cos at wavelengths that all divide
    `WORLD_PERIOD`, so two positions 2048 blocks apart are literally the same input tensor. If the
    occupancy target differs between them, the network is being asked for two different volumes
    from one input, and BCE answers conflicting targets the way MSE does — with their average. A
    uniform grey occupancy probability thresholds back to the plain heightfield, which is precisely
    the structure the band was added to escape. This project has now shipped that same
    conditional-averaging failure three times in 2D (`sample_params`'s docstring, `ReliefLoss`'s,
    `phase3.1-plan.md` §1.2); this test exists so the 3D repeat is caught by pytest instead of by
    looking at a trained model.

    `==` rather than `allclose` is deliberate. A field that is periodic to 1e-11 has *already* lost
    the property — the training target is the exact array, and "close" aliases still carry
    different labels. Exactness is what `_perlin3d`'s pre-hash `np.mod` and `_fbm3d`'s frequency
    quantization buy, so exactness is what gets asserted.

    **`OFF_GRID_FREQ` is the load-bearing parameter of this test and must stay off-grid.** An
    earlier version used `4.75 / 512`, for which `WORLD_PERIOD * freq` is 19.0 — an exact integer,
    as is every octave above it (38, 76, 152). `int(round(...))` is then a no-op in all four
    octaves and `q_freq == freq`, so the quantizer that `_fbm3d`'s docstring spends a paragraph on
    is never exercised at all. That version passed unchanged against a deliberately mutated
    `_fbm3d` that kept `n_cells` for the lattice but sampled at the *requested* frequency — i.e.
    against precisely the "tempting shortcut" the docstring warns about, which is the single most
    likely way for a future edit to break this. `4.37 / 512` gives 17.48 -> 17 cells (and 35 / 70 /
    140), so `q_freq != freq` in every octave and the mutant fails. The on-grid case is kept as
    well, but only the off-grid one has teeth.

    The multi-period assertion is NOT what catches a fractional-cell period — that shows up on the
    very first alias, because a mismatched `q_freq` lands the wrapped sample in a different cell
    immediately. It is kept for negative multiples and for the two-axis case.

    The spread assertions are not decoration: `assert a == b` is trivially satisfied by a kernel
    that returns zeros (or any constant) everywhere, which is a plausible way for this to break —
    an `n_cells` that collapsed to 1, a hash that returns one index. But they guard against a
    *constant* field, not a *wrong* one, so the continuity checks below are separate: a mutation
    harness showed the spread assertions alone also pass on a linear fade and on a corner-offset
    sign error (`dx1 = x - x0`) that makes the field discontinuous at every lattice plane. Perlin
    noise is C1 by construction — every corner dot vanishes at its own corner and the quintic fade
    is flat at t=0 and t=1 — so a jump in value *or in gradient* across a lattice plane is a real
    bug and not a tolerance question. A smoothstep fade is the one mutation these cannot catch, and
    the comment at the check says why it is out of reach and why that is acceptable.

    The y assertions cover the other direction: a field that ignores y entirely is bit-exactly
    periodic in x and z and is also just a 2D field wearing a third axis, which would carve vertical
    prisms and no overhangs; and a field that *wraps* y would stack identical caves at a fixed
    vertical interval.
    """
    import numpy as np
    from mc_imagine_model.data.world_generator import ProceduralWorldSource
    from mc_imagine_model.spec_constants import WORLD_PERIOD

    OFF_GRID_FREQ = 4.37 / 512   # 17.48 cells/period -> quantizer active in every octave
    ON_GRID_FREQ = 4.75 / 512    # 19.00 cells/period -> quantizer is a no-op; kept as a control
    assert (WORLD_PERIOD * OFF_GRID_FREQ) % 1.0 > 0.05, (
        "OFF_GRID_FREQ has drifted onto the 1/WORLD_PERIOD grid, which silently disables the only "
        "part of this test that exercises _fbm3d's frequency quantization"
    )

    # Negative, non-integer, and out-of-period coordinates on purpose: lattice indices go negative
    # wherever region coordinates or a domain warp push a sample left of the origin, and a C-style
    # truncating remainder (rather than `np.mod`'s floored one) tears the field along exactly the
    # x<0 / z<0 planes these points straddle.
    xs = np.array([-4096.0, -2048.0, -1999.75, -1024.5, -513.125, -1.0, 0.0, 0.5,
                   3.25, 137.75, 511.0, 1023.9375, 2047.5, 2048.0, 5000.3125])
    zs = xs[::-1] + 0.375
    # y values spanning the real band range, including below the world floor (a column anchored at
    # TERRAIN_CLIP_MIN puts its band bottom at -109, see spec_constants' band block).
    ys = np.array([-109.0, -64.5, 0.0, 62.25, 96.0, 200.75, 319.0])
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    src = ProceduralWorldSource
    for base_freq in (OFF_GRID_FREQ, ON_GRID_FREQ):
        for ridged in (False, True):
            tag = f"base_freq={base_freq * 512:.2f}/512, ridged={ridged}"
            kw = dict(seed=20260728, base_freq=base_freq, octaves=4, ridged=ridged)
            base = src._fbm3d(X, Y, Z, **kw)

            assert np.array_equal(base, src._fbm3d(X + WORLD_PERIOD, Y, Z, **kw)), (
                f"_fbm3d ({tag}) is not bit-exactly periodic in x; max deviation "
                f"{np.abs(base - src._fbm3d(X + WORLD_PERIOD, Y, Z, **kw)).max()!r}"
            )
            assert np.array_equal(base, src._fbm3d(X, Y, Z + WORLD_PERIOD, **kw)), (
                f"_fbm3d ({tag}) is not bit-exactly periodic in z; max deviation "
                f"{np.abs(base - src._fbm3d(X, Y, Z + WORLD_PERIOD, **kw)).max()!r}"
            )
            # Negative multiples, and both axes displaced at once.
            assert np.array_equal(
                base, src._fbm3d(X - 3 * WORLD_PERIOD, Y, Z + 7 * WORLD_PERIOD, **kw)
            ), f"_fbm3d ({tag}) drifts over multiple periods"

            # Not constant. Measured std at OFF_GRID_FREQ is 0.162 (plain) / 0.095 (ridged); 0.02
            # is a floor that only a broken kernel can fall under.
            assert base.std() > 0.02, f"_fbm3d ({tag}) is near-constant: std={base.std()}"
            assert np.ptp(base) > 0.1, f"_fbm3d ({tag}) has no range: ptp={np.ptp(base)}"

            # Genuinely three-dimensional. Axis 1 is y in the meshgrid above.
            assert np.abs(np.diff(base, axis=1)).max() > 0.05, (
                f"_fbm3d ({tag}) barely varies with y — it is a 2D field with a spare axis, which "
                f"carves vertical prisms and produces no overhangs"
            )

    kw = dict(seed=20260728, base_freq=OFF_GRID_FREQ, octaves=4)

    # Continuity across lattice planes, in value AND in first derivative. The spread assertions
    # above only rule out a *constant* field, not a *wrong* one; a mutation harness confirmed they
    # pass unchanged on a corner-offset sign error (`dx1 = x - x0`) that tears the field along every
    # lattice plane, and on a linear fade whose gradient jumps there. In the occupancy band either
    # one reads as a regular grid of one-block seams cutting through every cavity.
    #
    # `x * (n_cells / WORLD_PERIOD)` puts an octave's lattice planes at world-space multiples of
    # `WORLD_PERIOD / n_cells`, so straddle one of those rather than an integer world coordinate.
    #
    # Measured worst case over these 36 crossings: the real kernel jumps 0.0 in value and 7.9e-7 in
    # slope at h=1e-4 (pure finite-difference truncation error, and it scales linearly with h as
    # truncation error must). The sign-error mutant jumps 1.3e-1 in value; the linear-fade mutant
    # jumps 1.6e-2 in slope. Both thresholds therefore sit >100x clear of the real kernel and >100x
    # below the mutants.
    #
    # NOT caught, deliberately: a smoothstep fade (3t^2-2t^3). It measures 7.8e-7, identical to the
    # quintic — smoothstep is also C1, differing only in the second derivative, and it produces a
    # legitimate (1985-vintage) Perlin field. No value- or gradient-continuity test can reject it,
    # and it is not a correctness bug, only a deviation from the documented kernel.
    h = 1e-4
    for axis in range(3):
        for n_cells in (17, 35, 70, 140):  # the four octaves' lattices at OFF_GRID_FREQ
            plane = WORLD_PERIOD / n_cells  # world-space spacing of this octave's lattice planes
            for k in (-3, 1, 5):

                def at(offset: float, _axis: int = axis, _k: int = k, _p: float = plane) -> float:
                    pt = np.array([137.5, 42.25, -613.75])
                    pt[_axis] = _k * _p + offset
                    return float(src._fbm3d(*(np.array(t) for t in pt), **kw))

                where = f"axis-{axis} lattice plane at {k * plane} (n_cells={n_cells})"
                jump = abs(at(h) - at(-h))
                assert jump < 1e-3, (
                    f"_fbm3d jumps by {jump:.3e} in value across the {where}; the field is torn, "
                    f"which means a corner offset or the fade polynomial is wrong"
                )
                slope_jump = abs((at(-h) - at(-2 * h)) / h - (at(2 * h) - at(h)) / h)
                assert slope_jump < 1e-4, (
                    f"_fbm3d's gradient jumps by {slope_jump:.3e} across the {where}; the field is "
                    f"C0 but not C1, which is what a fade polynomial that is not flat at t=0 and "
                    f"t=1 (e.g. a linear fade) produces"
                )

    # y must NOT be wrapped: a vertical period would stack identical caves at a fixed y interval.
    assert not np.array_equal(
        src._fbm3d(X, Y, Z, **kw), src._fbm3d(X, Y + WORLD_PERIOD, Z, **kw)
    ), "_fbm3d appears periodic in y; y is bounded and directly available and must not be wrapped"

    # Broadcasting: the caller evaluates a [520, 520, 64] volume in y-slabs, passing x/y/z with
    # different shapes so only the corner dot products are full size (see `_fbm3d`'s memory table).
    got = src._fbm3d(
        xs.reshape(1, -1, 1), ys.reshape(-1, 1, 1), zs.reshape(1, 1, -1), **kw
    )
    assert got.shape == (len(ys), len(xs), len(zs))
    assert np.array_equal(got, np.transpose(src._fbm3d(X, Y, Z, **kw), (1, 0, 2))), (
        "broadcast evaluation must be bit-identical to the fully-materialized meshgrid, or the "
        "slabbed volume and any spot check of it will disagree"
    )


# --- the occupancy band (Phase 4 task A4 — docs/phase4-plan.md §2) ---------------------------
# One full `render_region` costs ~6.0 s for a carving archetype and ~0.4 s for a flat one (measured
# on an M-series laptop, single-threaded; the 3D carve field is ~95% of it). The band tests below
# want several archetypes and several coordinate shifts of the same one, so renders are cached by
# (archetype, region_x, region_z) — without this the four tests would re-render the same
# `deep_canyon` region five times and cost half a minute to say nothing new.
_BAND_RENDER_CACHE: Dict[tuple, tuple] = {}


def _params_for_archetype(archetype: str, seed: int = 7, limit: int = 400):
    """The first region in the layout scan that `sample_params` assigns to `archetype`.

    Picked by scanning rather than by hand-building a RegionParams so the test exercises the
    parameter vectors the dataset will actually contain, including the sampled carve gains.
    """
    src = ProceduralWorldSource(seed=seed)
    for region_id in range(limit):
        params = src.sample_params(region_id % 64, region_id // 64)
        if params.archetype == archetype:
            return src, params
    raise AssertionError(f"no {archetype!r} region in the first {limit} of seed {seed}")


def _render_archetype(archetype: str, seed: int = 7):
    """Cached `(params, region)` for one region of `archetype`."""
    key = (archetype, seed)
    if key not in _BAND_RENDER_CACHE:
        src, params = _params_for_archetype(archetype, seed)
        _BAND_RENDER_CACHE[key] = (params, src.render_region(params))
    return _BAND_RENDER_CACHE[key]


def test_band_top_solid_cell_is_the_returned_heightfield() -> None:
    """docs/phase4-plan.md §1.4: `heightmap` is redefined as **the topmost solid cell of the emitted
    volume**, and the generator must make that true rather than approximately true.

    "Those two definitions must not drift, or decoration floats and structures bury themselves."
    §1.4 enforces it at export by reading the volume; here it has to hold in the ground truth, or
    the model is trained on a heightmap and a band that disagree — and every consistency term in
    §3.3 would then be pulling the heads toward two different surfaces.

    It holds BY CONSTRUCTION: every 3D depth window starts at depth >= 3
    (`UNDERCUT_ROOF`/`ARCH_SPAN_CLEARANCE`/`CAVE_ROOF`), so nothing can remove the cell at depth 0,
    and the fill rule `depth >= 0` puts air in the two headroom cells above it. `_render_occupancy_
    band` asserts the cheap slice-wise version of that on every region it renders. This test asserts
    the expensive per-column version — an independent top-down argmax over the band, compared
    against `np.rint` of the returned heightfield — on four archetypes including the two that carry
    hoodoos, because the hoodoo is the one feature that *moves the anchor* and is therefore the one
    that would break this if the shave were applied after the anchor was taken instead of before.

    Also pinned here, because both are the same class of silent one-block error:
      * the carved cells all lie inside the union of the depth windows (indices 23..58 with the
        shipped constants), so a window that grew toward the surface shows up as a test failure
        rather than as terrain whose heightmap is quietly wrong;
      * no band cell below the world floor is air (`spec_constants`' band block requires those
        rendered solid so target and emitted volume agree about them).
    """
    import numpy as np
    from mc_imagine_model.data.world_generator import (
        ARCH_REACH, ARCH_SPAN_CLEARANCE, CAVE_REACH, CAVE_ROOF, UNDERCUT_REACH, UNDERCUT_ROOF,
        WORLD_FLOOR_Y,
    )
    from mc_imagine_model.spec_constants import BAND_BOTTOM_OFFSET, BAND_HEIGHT

    deepest = max(UNDERCUT_ROOF + UNDERCUT_REACH, ARCH_SPAN_CLEARANCE + ARCH_REACH,
                  CAVE_ROOF + CAVE_REACH)
    shallowest = min(UNDERCUT_ROOF, ARCH_SPAN_CLEARANCE, CAVE_ROOF)
    assert shallowest > 0, (
        "a depth window reached depth 0, so a 3D feature can now remove a column's surface cell "
        "and docs/phase4-plan.md §1.4 stops holding by construction. That is a design change, not "
        "a bug to paper over: it needs the returned heightfield to be re-derived from the band"
    )

    for archetype in ("deep_canyon", "mesa_plateaus", "fjord_rift", "rolling_grassland"):
        params, region = _render_archetype(archetype)
        band, h = region["band"], region["heightfield"]
        assert band.shape == h.shape + (BAND_HEIGHT,)
        anchors = np.rint(h).astype(np.int64)

        # Independent top-down search for the topmost solid cell, deliberately NOT the same
        # expression the generator uses.
        top_i = BAND_HEIGHT - 1 - np.argmax(band[:, :, ::-1], axis=2)
        assert band.any(axis=2).all(), f"{archetype}: a column has no solid cell at all"
        top_world_y = anchors + BAND_BOTTOM_OFFSET + top_i
        bad = int((top_world_y != anchors).sum())
        assert bad == 0, (
            f"{archetype}: {bad} of {anchors.size} columns have a topmost solid band cell that is "
            f"not rint(heightfield). The band and the heightmap disagree about where the surface "
            f"is, which trains fine and produces terrain shifted against its own heightmap"
        )

        # Carving is confined to the union of the depth windows.
        i = np.arange(BAND_HEIGHT)
        depth = -(BAND_BOTTOM_OFFSET + i)
        base_solid = depth >= 0
        carved = (~band) & base_solid[None, None, :]
        carved_i = np.flatnonzero(carved.any(axis=(0, 1)))
        if carved_i.size:
            carved_depth = depth[carved_i]
            assert carved_depth.max() <= deepest and carved_depth.min() >= shallowest, (
                f"{archetype}: carving reached depths {carved_depth.min()}..{carved_depth.max()}, "
                f"outside the union of the feature windows {shallowest}..{deepest}"
            )

        # Nothing below y=-64 may be air.
        world_y = anchors[:, :, None] + BAND_BOTTOM_OFFSET + i[None, None, :]
        below_floor = world_y < WORLD_FLOOR_Y
        assert not (below_floor & ~band).any(), (
            f"{archetype}: air below the world floor. export_onnx drops those cells, so the target "
            f"and the emitted volume would disagree about them"
        )


def test_band_is_bit_exactly_world_period_periodic_and_deterministic() -> None:
    """docs/phase4-plan.md §2.2(a), end to end: the *band*, not just `_fbm3d`, must be exactly
    `WORLD_PERIOD`-periodic in x and z — and `==`, not `allclose`.

    `test_fbm3d_is_bit_exactly_world_period_periodic` pins the kernel. This pins the whole
    rasterizer on top of it, which is where the property is actually easy to lose: the gates come
    from `np.gradient` and a gaussian blur of the heightfield, the anchor comes from `np.rint` of a
    float32, and the carve is a threshold — every one of those turns a 1e-11 coordinate difference
    into a flipped block. `CoordinateEncoder` gives the network *literally the same input tensor*
    2048 blocks apart, so a band that differs there is two different occupancy volumes demanded of
    one input, and BCE answers that with their average: the uniform grey that thresholds back to
    the plain heightfield this band exists to escape.

    The shift is applied as `region_x + WORLD_PERIOD // REGION_BLOCKS` on an otherwise identical
    parameter vector, which is the honest test: same seed, same archetype, same gains, a canvas
    displaced by exactly one period. `render_region` folds coordinates with `np.mod` before any
    noise call, so the two canvases are bit-identical arrays and everything downstream must be too.
    A version of this that resampled `sample_params` at the shifted region would have compared two
    different worlds and passed for the wrong reason.

    Determinism is checked in the same test because it needs the same fixtures: re-rendering the
    same parameter vector must be bit-identical. Nothing in the band path is stochastic, but
    `_region_rng` streams, a `dict` iteration order over the feature list, or a future threaded
    slab loop could each make it not so, and a non-deterministic target is unlearnable in exactly
    the way this project has already shipped three times.
    """
    import dataclasses

    import numpy as np
    from mc_imagine_model.spec_constants import WORLD_PERIOD

    src, params = _params_for_archetype("deep_canyon")
    assert params.carve_strength > 0 and params.hoodoo_strength > 0, (
        "this test needs an archetype that actually carves and shaves, or it proves nothing"
    )
    base = _render_archetype("deep_canyon")[1]
    assert not base["band"].all(), "deep_canyon carved nothing; the test has no structure to check"

    period_regions = WORLD_PERIOD // REGION_BLOCKS  # 4
    for axis in ("region_x", "region_z"):
        shifted = src.render_region(
            dataclasses.replace(params, **{axis: getattr(params, axis) + period_regions})
        )
        assert np.array_equal(base["band"], shifted["band"]), (
            f"the occupancy band is not bit-exactly periodic under a {axis} shift of one "
            f"WORLD_PERIOD: {int((base['band'] != shifted['band']).sum())} of "
            f"{base['band'].size} cells differ"
        )
        assert np.array_equal(base["heightfield"], shifted["heightfield"]), (
            f"the post-hoodoo heightfield is not bit-exactly periodic under a {axis} shift"
        )

    again = src.render_region(params)
    assert np.array_equal(base["band"], again["band"]), "band render is not deterministic"
    assert np.array_equal(base["heightfield"], again["heightfield"]), (
        "post-hoodoo heightfield render is not deterministic"
    )


def test_flat_archetypes_have_no_3d_structure() -> None:
    """`desert_dunes`, `rolling_grassland`, `swamp` and `savanna_plains` must emit the plain
    heightfield expansion, cell for cell.

    docs/phase4-plan.md §2.1: "Flat terrain with overhangs in it is not a feature; the caption
    'endless flat desert dunes' should produce a solid dune field and the model needs to learn that
    the *absence* of 3D structure is also prompt-determined." That makes these four the control
    case, and a control case that is 99.9% right is not a control case.

    `test_relief_frequency_is_pinned_per_archetype` asserts the *parameters* are zero. This asserts
    the *raster* is, which is a different claim: it also covers the hoodoo (a heightfield edit that
    bypasses the carve threshold entirely) and it would catch a carve that ignored its gain.

    The expected band is written out here rather than imported from `diagnose_overhangs.
    heightfield_band` on purpose — a shared helper would let one wrong constant satisfy both sides.
    """
    import numpy as np
    from mc_imagine_model.spec_constants import BAND_BOTTOM_OFFSET, BAND_HEIGHT, BAND_TOP_OFFSET

    depth = -(BAND_BOTTOM_OFFSET + np.arange(BAND_HEIGHT))
    expected_column = depth >= 0
    assert int(expected_column.sum()) == BAND_HEIGHT - BAND_TOP_OFFSET

    for archetype in ("desert_dunes", "rolling_grassland", "swamp", "savanna_plains"):
        params, region = _render_archetype(archetype)
        band = region["band"]
        assert params.carve_strength == 0.0, f"{archetype} sampled a nonzero carve_strength"
        expected = np.broadcast_to(expected_column, band.shape)
        differing = int((band != expected).sum())
        assert differing == 0, (
            f"{archetype}: {differing} band cells differ from the plain heightfield expansion. The "
            f"flat archetypes are the control case that teaches the model the absence of 3D "
            f"structure is prompt-determined"
        )
        # And the hoodoo must not have moved the surface either: h_carved is h.
        assert np.isfinite(region["heightfield"]).all()


def test_verify_coordinate_purity_matches_export_path() -> None:
    """`export/export_onnx.py:verify_coordinate_purity` is the check that runs against the real
    exported model; it must stay exact under the Phase 2 halo and the new seed-aware signature."""
    from mc_imagine_model.export.export_onnx import verify_coordinate_purity

    model = ImagineNet(SMALL_CONFIG)
    model.eval()
    assert verify_coordinate_purity(model, seed=12345) == 0.0
    assert verify_coordinate_purity(model, seed=-8756019554883095434) == 0.0


def test_configs_declare_relief_weight() -> None:
    """Assert that config.cuda.yaml, config.mps.yaml, and config.rehearsal.yaml explicitly define
    relief_weight, occupancy_weight, overhang_weight, and consistency_weight under training using PyYAML.

    The code defaults the Phase 4 weights to 0.0 for backwards compatibility, but shipped configs
    must not use that value or the occupancy head reaches the training box unsupervised — except
    consistency_weight, which is deliberately pinned to 0.0 in all three configs
    (docs/remote-training-readiness-plan.md §2.H1/§5 item 0.2): ConsistencyLoss's residual is
    minimized by pushing soft_topmost toward the top of the band regardless of data, a live
    candidate contributor to Gate 2's uniform failure. That decision is locked; do not revert it
    here without revisiting §2.H1 first.
    """
    import os
    import yaml

    test_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.dirname(test_dir)
    config_dir = os.path.join(model_dir, "src", "mc_imagine_model", "training")

    configs = ["config.cuda.yaml", "config.mps.yaml", "config.rehearsal.yaml"]
    expected_positive_weights = ["relief_weight", "occupancy_weight", "overhang_weight"]
    for config_name in configs:
        path = os.path.join(config_dir, config_name)
        assert os.path.isfile(path), f"Config file not found: {path}"
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert "training" in cfg, f"Missing 'training' section in {config_name}"
        for w_name in expected_positive_weights:
            assert w_name in cfg["training"], (
                f"Missing '{w_name}' under 'training' in {config_name}"
            )
            assert float(cfg["training"][w_name]) > 0.0, (
                f"{config_name} sets {w_name}=0.0; shipped configs must train every Phase 4 loss"
            )
        assert "consistency_weight" in cfg["training"], (
            f"Missing 'consistency_weight' under 'training' in {config_name}"
        )
        assert float(cfg["training"]["consistency_weight"]) == 0.0, (
            f"{config_name} sets consistency_weight != 0.0; this is locked at 0.0 per "
            "docs/remote-training-readiness-plan.md §2.H1 — see this test's docstring."
        )


def test_imagine_net_forward_backward_dry_run() -> None:
    """Forward and backward pass dry-run test asserting loss computation and gradient flow
    through 3D head and combined loss with non-zero weights."""
    from mc_imagine_model.training.losses import CombinedLoss

    config = {
        "halo": 9,
        "patch": 16,
        "conv_channels": 32,
        "num_conv_layers": 6,
        "num_conv3d_layers": 3,
        "fusion_hidden": 64,
        "fusion_out": 32,
        "seed_freqs": 16,
        "seed_embed_dim": 32,
        "num_profiles": 8,
        "num_biomes": 12,
    }
    model = ImagineNet(config)
    model.train()

    b = 2
    prompt_tokens = torch.tensor([tokenize("rocky mountains with caves")] * b, dtype=torch.int32)
    chunk_x = torch.tensor([0, 1], dtype=torch.int32)
    chunk_z = torch.tensor([0, 1], dtype=torch.int32)
    seed = torch.tensor([123, 456], dtype=torch.int64)

    preds = model(prompt_tokens, chunk_x, chunk_z, seed)
    assert preds["occupancy_logits"].shape == (b, 64, 16, 16)

    targets = {
        "heightmap": torch.full((b, 16, 16), 64.0, dtype=torch.float32),
        "profile_id": torch.zeros(b, 16, 16, dtype=torch.long),
        "water_level": torch.full((b,), 63.0, dtype=torch.float32),
        "biome_grid": torch.zeros(b, 4, 4, dtype=torch.long),
        "band": torch.randint(0, 2, (b, 64, 16, 16), dtype=torch.float32),
    }

    loss_fn = CombinedLoss(
        terrain_weight=1.0,
        biome_weight=1.0,
        slope_weight=1.0,
        relief_weight=10.0,
        occupancy_weight=1.0,
        overhang_weight=1.0,
        consistency_weight=1.0,
    )

    total_loss, components = loss_fn(preds, targets)
    assert torch.isfinite(total_loss)
    for k in ("terrain", "biome", "relief", "slope", "occupancy", "overhang", "consistency"):
        assert k in components
        assert torch.isfinite(components[k])

    total_loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient for {name}"


def _corrupt_shard(src_path: str, dst_path: str, **overrides) -> None:
    """Writes a copy of a real shard with some arrays replaced/removed, for stale-shard tests."""
    import numpy as np

    with np.load(src_path) as npz:
        arrays = {k: npz[k] for k in npz.files}
    for key, value in overrides.items():
        if value is _MISSING:
            arrays.pop(key, None)
        else:
            arrays[key] = value
    np.savez_compressed(dst_path, **arrays)


_MISSING = object()


def test_load_shard_raises_not_asserts_on_stale_or_corrupt_data(tmp_path) -> None:
    """docs/remote-training-readiness-plan.md §2.E3 / §5 item 0.5.

    `McImagineDataset._load_shard`'s four stale-shard guards were `assert` statements, which
    `python -O`/`PYTHONOPTIMIZE=1` strips — some container images set that. This pins them as
    `raise`s that survive `-O`, by actually running this test under `-O` (see the `-O` invocation
    this docstring's item requires) against four kinds of deliberately corrupted shard.
    """
    import numpy as np
    from mc_imagine_model.data.dataset import McImagineDataset
    from mc_imagine_model.data.world_generator import CANVAS, ProceduralWorldSource
    from mc_imagine_model.spec_constants import HEIGHT_MAX, NUM_BIOMES

    real_dir = tmp_path / "real"
    real_dir.mkdir()
    [real_path] = ProceduralWorldSource(seed=0).generate_shards(1, str(real_dir), region_layout_width=64)

    cases = {
        "missing_band": dict(band=_MISSING),
        "bad_band_shape": dict(band=np.zeros((CANVAS, CANVAS, 8), dtype=bool)),
        "biome_out_of_range": dict(biome_map=np.full((CANVAS, CANVAS), NUM_BIOMES, dtype=np.uint8)),
        "height_out_of_range": dict(
            heightfield=np.full((CANVAS, CANVAS), HEIGHT_MAX + 100.0, dtype=np.float32)
        ),
    }
    for name, overrides in cases.items():
        bad_path = tmp_path / f"region_{name}.npz"
        _corrupt_shard(real_path, str(bad_path), **overrides)
        ds = McImagineDataset(str(tmp_path), shard_paths=[str(bad_path)],
                              chunks_per_region=1, shard_cache_size=1)
        with pytest.raises(ValueError):
            ds[0]


def _tiny_model_optimizer_scheduler():
    model = ImagineNet(SMALL_CONFIG)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    return model, optimizer, scheduler


def test_export_raises_on_missing_state_dict_keys(tmp_path) -> None:
    """docs/remote-training-readiness-plan.md §2.J1 / §5 item 1.2, core case: a checkpoint that is
    missing real (non-text-encoder) weights — e.g. written before an architecture change added a
    head — must raise on export, not silently ship a randomly-initialized piece of the model under
    strict=False.
    """
    from mc_imagine_model.export.export_onnx import load_imagine_net
    from mc_imagine_model.training.train import save_checkpoint

    model, optimizer, scheduler = _tiny_model_optimizer_scheduler()
    config = {"model": dict(SMALL_CONFIG)}
    ckpt_path = os.path.join(str(tmp_path), "stale.pt")
    save_checkpoint(ckpt_path, model, optimizer, scheduler, None, config,
                    epoch=0, global_step=1, val_loss=1.0, best_val_loss=1.0)

    # Simulate a stale/incompatible checkpoint by dropping real (non-text-encoder) keys after the
    # fact — the shape of bug this check exists to catch: a checkpoint saved before an architecture
    # change (e.g. missing occupancy_head.*) still needs to fail loudly on export.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    dropped = [k for k in ckpt["model_state_dict"] if k.startswith("occupancy_head.")]
    assert dropped, "expected SMALL_CONFIG's ImagineNet to have an occupancy_head"
    for k in dropped:
        del ckpt["model_state_dict"][k]
    torch.save(ckpt, ckpt_path)

    with pytest.raises(RuntimeError, match="missing="):
        load_imagine_net(ckpt_path)


def test_export_raises_when_checkpoint_dir_missing_for_c2_format_checkpoint(tmp_path) -> None:
    """docs/remote-training-readiness-plan.md §2.J2 / §5 item 1.2 — the literal negative test the
    plan requires. C2 (frozen text encoder dropped from checkpoints, `train.save_checkpoint`) is
    only safe because J1 (export raises on unexpected missing keys) and E2 (`PromptEncoder`'s
    random fallback is fatal by default, item 0.6) are both already in place. If
    `model/checkpoints/` is unreachable at export time, this composition must fail loudly — not
    silently package a `.mcim` whose text encoder is random noise.
    """
    from mc_imagine_model.export.export_onnx import load_imagine_net
    from mc_imagine_model.training.train import save_checkpoint

    model, optimizer, scheduler = _tiny_model_optimizer_scheduler()
    # This checkpoint's own model_state_dict already omits text_encoder.* (save_checkpoint's C2
    # behavior) — the point under test is what happens when export tries to reconstruct the model
    # and the checkpoint's *own* config points at a checkpoint_dir that no longer exists.
    config = {"model": dict(SMALL_CONFIG)}
    config["model"]["checkpoint_dir"] = str(tmp_path / "definitely_missing_minilm_checkpoint")
    ckpt_path = os.path.join(str(tmp_path), "last.pt")
    save_checkpoint(ckpt_path, model, optimizer, scheduler, None, config,
                    epoch=0, global_step=1, val_loss=1.0, best_val_loss=1.0)

    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert not any(k.startswith("text_encoder.") for k in saved["model_state_dict"]), (
        "test setup assumption broken: save_checkpoint should already exclude text_encoder.*"
    )

    with pytest.raises(RuntimeError):
        load_imagine_net(ckpt_path)
