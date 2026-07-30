"""
Basic pytest tests for the Mc-Imagine model components.
"""

import random
from typing import Dict

import torch
import pytest

from mc_imagine_model.data.world_generator import ARCHETYPES, ProceduralWorldSource
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
    seed_to_offset,
)
from mc_imagine_model.tokenizer_utils import tokenize
from mc_imagine_model.training.losses import ReliefLoss, TerrainLoss

# A small-but-valid config: halo must equal num_conv_layers (each valid 3x3 conv eats one ring),
# see imagine_net._check_conv_arithmetic.
SMALL_CONFIG = {
    "conv_channels": 32,
    "fusion_hidden": 64,
    "fusion_out": 32,
    "num_conv_layers": 4,
    "halo": 4,
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
    # heightmap/water_level must lie in the head's HEIGHT_CENTER +/- HEIGHT_AMPLITUDE band.
    for key in ("heightmap", "water_level"):
        assert torch.all(out[key] >= HEIGHT_MIN)
        assert torch.all(out[key] <= HEIGHT_MAX)


def test_imagine_net_phase2_dims() -> None:
    """The Phase 2 capacity config (docs/phase2-plan.md Phase 2) must build and produce the exact
    spec'd output shapes. These are the defaults `training/config*.yaml` sets."""
    config = {
        "halo": 6,
        "patch": 16,
        "conv_channels": 192,
        "num_conv_layers": 6,
        "fusion_hidden": 512,
        "fusion_out": 256,
        "seed_freqs": 32,
        "seed_embed_dim": 64,
        "num_profiles": 8,
        "num_biomes": 12,
    }
    model = ImagineNet(config)
    model.eval()
    assert model.coord_encoder.size == 28  # 16 + 2*6

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


def test_conv_arithmetic_is_asserted() -> None:
    """halo/num_conv_layers mismatches must fail loudly at construction, not silently emit a
    wrong-sized tile that only explodes (or worse, doesn't) far downstream."""
    _check_conv_arithmetic(halo=6, patch=16, num_conv_layers=6)  # the Phase 2 setting
    _check_conv_arithmetic(halo=4, patch=16, num_conv_layers=4)  # the Day 1 setting
    with pytest.raises(ValueError):
        _check_conv_arithmetic(halo=6, patch=16, num_conv_layers=4)
    with pytest.raises(ValueError):
        _check_conv_arithmetic(halo=4, patch=16, num_conv_layers=6)
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


def test_relief_frequency_is_pinned_per_archetype() -> None:
    """`relief_frequency` must stay a pure function of the archetype (hence of the caption).

    It is the one sampled parameter the caption cannot describe and the world seed cannot reveal —
    `sample_params` draws parameters from `_region_rng(..., stream=0)` and the world seed from
    `stream=1`, decorrelated on purpose. Randomizing it makes flat terrain the MSE optimum:
    averaging a fixed-phase noise field over a range of frequencies decorrelates the fields being
    averaged and annihilates their structure. Measured, as the fraction of within-chunk relief
    surviving conditional averaging:

        amplitude only ~100%   erosion only ~100%   ridge+plateau 83-100%   frequency only 20-37%

    So amplitude/erosion/ridge/plateau may keep varying freely; frequency may not. This test fails
    loudly if a future edit widens a range or blends the value away, instead of the world silently
    going flat again.
    """
    for name, spec in ARCHETYPES.items():
        lo, hi = spec["relief_frequency"]
        assert lo == hi, (
            f"archetype {name!r} has relief_frequency=({lo}, {hi}); it must be pinned (lo == hi) "
            f"or flat terrain becomes the loss optimum again"
        )

    # The 25% secondary-archetype blend in `sample_params` must also leave it alone: the caption is
    # keyed to the PRIMARY archetype, so a blended frequency is caption-invisible and reopens the
    # same hole for a quarter of all regions.
    src = ProceduralWorldSource(seed=1234)
    seen: Dict[str, set] = {}
    for region_id in range(1500):
        params = src.sample_params(region_id % 64, region_id // 64)
        seen.setdefault(params.archetype, set()).add(round(params.relief_frequency, 9))

    leaked = {a: sorted(v) for a, v in seen.items() if len(v) > 1}
    assert not leaked, (
        f"these archetypes produced more than one relief_frequency across sampled regions, so the "
        f"secondary-archetype blend is leaking it: {leaked}"
    )
    assert len(seen) >= 8, f"only {len(seen)} archetypes sampled; test is not covering enough"


def test_verify_coordinate_purity_matches_export_path() -> None:
    """`export/export_onnx.py:verify_coordinate_purity` is the check that runs against the real
    exported model; it must stay exact under the Phase 2 halo and the new seed-aware signature."""
    from mc_imagine_model.export.export_onnx import verify_coordinate_purity

    model = ImagineNet(SMALL_CONFIG)
    model.eval()
    assert verify_coordinate_purity(model, seed=12345) == 0.0
    assert verify_coordinate_purity(model, seed=-8756019554883095434) == 0.0
