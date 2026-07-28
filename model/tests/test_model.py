"""
Basic pytest tests for the Mc-Imagine model components.
"""

import torch
import pytest

from mc_imagine_model.model.imagine_net import ImagineNet
from mc_imagine_model.model.text_encoder import PromptEncoder
from mc_imagine_model.model.positional import CoordinateEncoder, SeedEncoder, verify_seed_encoder_varies
from mc_imagine_model.spec_constants import NUM_BIOMES, NUM_PROFILES
from mc_imagine_model.tokenizer_utils import tokenize


def test_imagine_net_forward() -> None:
    """
    Test that the ImagineNet model can perform a forward pass and return expected shapes.
    """
    config = {"conv_channels": 32, "fusion_hidden": 64, "fusion_out": 32, "num_conv_layers": 4}
    model = ImagineNet(config)
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
    # heightmap must be within the 63 +/- 96*tanh band
    assert torch.all(out["heightmap"] >= 63.0 - 96.0)
    assert torch.all(out["heightmap"] <= 63.0 + 96.0)


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
    encoder = CoordinateEncoder(halo=4, patch=16)
    chunk_x = torch.tensor([0, 1])
    chunk_z = torch.tensor([0, 0])
    output = encoder(chunk_x, chunk_z)
    assert output.shape == (2, encoder.out_channels, 24, 24)

    # chunk(0,0)'s halo covers global x in [-4, 19]; chunk(1,0)'s halo covers global x in [12, 35].
    # Shared range [12, 19] -> local index 16..23 in chunk0's grid, local index 0..7 in chunk1's grid.
    shared_0 = output[0, :, :, 16:24]
    shared_1 = output[1, :, :, 0:8]
    assert torch.equal(shared_0, shared_1), "CoordinateEncoder is not a pure function of global coordinate"


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
