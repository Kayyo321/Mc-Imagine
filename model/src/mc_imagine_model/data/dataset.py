"""
PyTorch Dataset class for training the Mc-Imagine model.
"""

import glob
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from mc_imagine_model.data.labeler import ChunkLabeler
from mc_imagine_model.data.world_generator import CANVAS, HALO_MARGIN, REGION_BLOCKS
from mc_imagine_model.spec_constants import HEIGHT_MAX, HEIGHT_MIN, NUM_BIOMES
from mc_imagine_model.tokenizer_utils import MAX_PROMPT_TOKENS, tokenize

CHUNKS_PER_REGION_SIDE = REGION_BLOCKS // 16  # 32
QUARTER = 4  # biome_grid quarter-resolution factor


class McImagineDataset(Dataset):
    """
    Dataset for loading prompt-conditioned chunk data, sourced from `.npz` macro-region shards
    written by `data/world_generator.py`'s `ProceduralWorldSource.generate_shards`.

    Each `__getitem__` call slices one 16x16 chunk (by `[cx, cz]` within its containing macro-region)
    out of the region's (CANVAS x CANVAS) heightfield/profile/biome rasters. `world_generator.py`
    renders those rasters with a HALO_MARGIN border specifically so every chunk position — including
    edge chunks — can be sliced without extra padding; see its module docstring and
    docs/poc-plan.md §1b. Note what this dataset does *not* need to materialize: `block_volume`.
    Per docs/poc-plan.md's Phase 6 export design, `block_volume` is a deterministic function of
    (heightmap, profile, water_level, biome) computed by `torch.where`-style expansion *inside* the
    exported graph — the network only ever predicts the ~350 scalars per chunk that determine it,
    not the full 98,304-cell volume. Training therefore only needs heightmap/profile/water/biome
    ground truth; `block_volume` stays a zero placeholder here (matching `structure_markers`/
    `structure_graph_*`, which are also zeroed since this is a `structure_support: "none"` model).
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Any] = None,
        chunks_per_region: int = 100,
        max_tokens: int = MAX_PROMPT_TOKENS,
        index_seed: int = 0,
        shard_cache_size: int = 32,
        shard_paths: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            data_dir (str): Directory containing `region_*.npz` shards (see
                `data/world_generator.py`'s `ProceduralWorldSource.generate_shards`).
            transform (callable, optional): Optional transform to be applied on a sample.
            chunks_per_region (int): How many of a region's 32x32=1024 chunk positions to include
                in the dataset index (subsampled, deterministically, per region) — see
                docs/poc-plan.md's Phase 4 target of "~200k chunk samples across ~2k parameter draws".
            max_tokens (int): Prompt token sequence length (must match `capabilities.max_prompt_tokens`).
            index_seed (int): Seed for the deterministic per-region chunk-position subsample and for
                `ChunkLabeler`'s template/synonym choice.
            shard_cache_size (int): Number of decompressed region shards to keep resident (simple
                LRU) — avoids re-decompressing the same `.npz` array repeatedly across the ~100
                samples drawn from each region.
            shard_paths (list[str], optional): Explicit list of shard paths to use instead of
                globbing `data_dir` — lets a caller (see `training/train.py`) split shards by
                *region* into train/val sets (holding out whole regions, not random chunks within
                them, since chunks from the same region share a caption and are highly correlated).
        """
        self.data_dir = data_dir
        self.transform = transform
        self.max_tokens = max_tokens

        if shard_paths is not None:
            shard_paths = list(shard_paths)
        else:
            shard_paths = sorted(glob.glob(os.path.join(data_dir, "region_*.npz")))
            if not shard_paths:
                # Also allow pointing directly at a directory containing shards nested one level down
                shard_paths = sorted(glob.glob(os.path.join(data_dir, "**", "region_*.npz"), recursive=True))

        labeler = ChunkLabeler(rng_seed=index_seed)
        self._region_meta: Dict[str, Dict[str, Any]] = {}
        index: List[Tuple[str, int, int]] = []

        for shard_idx, path in enumerate(shard_paths):
            with np.load(path) as npz:
                params = {
                    k[len("param_"):]: (npz[k].item() if npz[k].shape == () else npz[k])
                    for k in npz.files
                    if k.startswith("param_")
                }
            caption = labeler.label_region(params)
            token_ids = tokenize(caption, max_tokens=max_tokens)
            self._region_meta[path] = {
                "params": params,
                "caption": caption,
                "token_ids": np.array(token_ids, dtype=np.int32),
            }

            rng = np.random.default_rng(index_seed + shard_idx * 7919 + 1)
            n_total = CHUNKS_PER_REGION_SIDE * CHUNKS_PER_REGION_SIDE
            n_pick = min(chunks_per_region, n_total)
            chosen_flat = rng.choice(n_total, size=n_pick, replace=False)
            for flat in chosen_flat:
                cx, cz = divmod(int(flat), CHUNKS_PER_REGION_SIDE)
                index.append((path, cx, cz))

        self.shard_paths = shard_paths
        self.index = index
        self._shard_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._shard_cache_size = shard_cache_size

    def __len__(self) -> int:
        return len(self.index)

    def _load_shard(self, path: str) -> Dict[str, Any]:
        cached = self._shard_cache.get(path)
        if cached is not None:
            self._shard_cache.move_to_end(path)
            return cached
        with np.load(path) as npz:
            data = {
                "heightfield": npz["heightfield"],
                "profile_map": npz["profile_map"],
                "biome_map": npz["biome_map"],
                "water_level": float(npz["water_level"]),
            }
        # Cheap once-per-shard contract check (not per-sample). Phase 2 changed both the biome
        # palette (8 -> 12 ids) and the height band, so a directory left over from an older
        # generator would otherwise train silently against out-of-contract targets — exactly the
        # failure mode docs/phase2-plan.md §1b wants to make impossible. Heights are legitimately
        # negative now, so this is a two-sided bound, not a floor at 0.
        biome_max = int(data["biome_map"].max())
        assert biome_max < NUM_BIOMES, (
            f"{path}: biome id {biome_max} >= NUM_BIOMES ({NUM_BIOMES}) — stale shard, regenerate "
            f"with ProceduralWorldSource.generate_shards"
        )
        h_min, h_max = float(data["heightfield"].min()), float(data["heightfield"].max())
        assert HEIGHT_MIN < h_min and h_max < HEIGHT_MAX, (
            f"{path}: heights [{h_min:.1f}, {h_max:.1f}] fall outside the head's representable band "
            f"({HEIGHT_MIN}, {HEIGHT_MAX}) — TerrainHead's tanh would saturate on these targets"
        )
        self._shard_cache[path] = data
        if len(self._shard_cache) > self._shard_cache_size:
            self._shard_cache.popitem(last=False)
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a data sample.

        Tensor shapes/fields follow docs/model-spec.md. `capability_tier` here refers to
        structure_support ("none" | "basic" | "intricate") and selects which structure fields are
        meaningful for this sample: "none" samples carry zeroed/unused structure_markers and
        structure_graph_* fields; "basic" samples populate structure_markers only; "intricate"
        samples populate structure_graph_nodes/_edges/_origin only (see StructureLoss vs.
        StructureGraphLoss in training/losses.py, which each read one or the other). intensity
        ("low" | "medium" | "high") is a separate, independent field controlling detail density and
        isn't gated the same way.

        Args:
            idx (int): Sample index.

        Returns:
            dict: Sample containing prompt_tokens, chunk_x, chunk_z, seed, capability_tier,
                  heightmap, block_volume, biome_grid, structure_markers, structure_graph_nodes,
                  structure_graph_edges, and structure_origin.
        """
        path, cx, cz = self.index[idx]
        shard = self._load_shard(path)
        meta = self._region_meta[path]
        params = meta["params"]

        R = HALO_MARGIN
        x0, x1 = cx * 16, cx * 16 + 16 + 2 * R
        z0, z1 = cz * 16, cz * 16 + 16 + 2 * R
        assert x1 <= CANVAS and z1 <= CANVAS, "chunk halo window exceeds region canvas"

        # Core 16x16 crop (no halo) — matches model.onnx's actual output tensor shapes. The halo
        # margin baked into the region raster (see world_generator.py) only exists so this slice is
        # always valid, including at region edges; the network's own cross-chunk-continuity
        # mechanism (CoordinateEncoder + valid-padded convs, docs/poc-plan.md §1b) is a pure
        # function of chunk_x/chunk_z at inference time and needs no ground-truth halo data here.
        height_core = shard["heightfield"][z0 + R : z0 + R + 16, x0 + R : x0 + R + 16]
        profile_core = shard["profile_map"][z0 + R : z0 + R + 16, x0 + R : x0 + R + 16]
        biome_core = shard["biome_map"][z0 + R : z0 + R + 16, x0 + R : x0 + R + 16]

        # biome_grid ground truth at quarter resolution (4x4), majority-vote per 4x4 block.
        # `minlength=NUM_BIOMES` pins the histogram to the palette rather than to whichever ids
        # happen to occur in this 4x4 window — behaviourally identical for argmax, but it means the
        # palette size lives in exactly one place (spec_constants) now that it is 12 and not 8.
        # Ties break toward the lower biome id, as before.
        biome_quarter = np.zeros((QUARTER, QUARTER), dtype=np.int64)
        for bz in range(QUARTER):
            for bx in range(QUARTER):
                block = biome_core[bz * 4 : bz * 4 + 4, bx * 4 : bx * 4 + 4]
                counts = np.bincount(block.reshape(-1).astype(np.int64), minlength=NUM_BIOMES)
                biome_quarter[bz, bx] = int(np.argmax(counts))

        water_level = float(shard["water_level"])

        chunk_x_global = int(params["region_x"]) * CHUNKS_PER_REGION_SIDE + cx
        chunk_z_global = int(params["region_z"]) * CHUNKS_PER_REGION_SIDE + cz

        sample = {
            "prompt_tokens": torch.from_numpy(meta["token_ids"].copy()).to(torch.int32),
            "chunk_x": torch.tensor(chunk_x_global, dtype=torch.int32),
            "chunk_z": torch.tensor(chunk_z_global, dtype=torch.int32),
            "seed": torch.tensor(int(params["seed"]), dtype=torch.int64),
            "capability_tier": "none",
            "heightmap": torch.from_numpy(height_core.copy()).to(torch.float32),
            "profile_id": torch.from_numpy(profile_core.copy().astype(np.int64)),
            "water_level": torch.tensor(water_level, dtype=torch.float32),
            "block_volume": torch.zeros(0),  # see class docstring — derived at export time, not trained
            "biome_grid": torch.from_numpy(biome_quarter.copy()),
            "structure_markers": torch.zeros(0),
            "structure_graph_nodes": torch.zeros(0),
            "structure_graph_edges": torch.zeros(0),
            "structure_origin": torch.zeros(0),
            "caption": meta["caption"],
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class McImagineMacroDataset(Dataset):
    """
    Dataset for training MacroFieldNet (model/macro_net.py) — only needed for models with
    capabilities.requires_macro_field = true. Samples are per 32x32-chunk macro-region (matching
    vanilla's .mca region file boundary), not per-chunk, which is why this is a separate class from
    McImagineDataset rather than extra fields on it: MacroFieldNet trains on its own, at a coarser
    granularity, with its own loss (MacroFieldLoss in training/losses.py).

    Note the natural alignment with data sourcing: world_generator.py + chunk_extractor.py already
    operate on .mca region files (see PROJECT.md "Model Training Pipeline" data sourcing section),
    so a macro-region training sample and a raw extracted region file cover the same ground.

    Not implemented in this pass: `imaginator-low_intensity-no_structures` (the only model this PoC
    trains) declares `requires_macro_field: false` and never trains or exports `macro.onnx` — see
    docs/poc-plan.md's scope.
    """

    def __init__(self, data_dir: str, transform: Optional[Any] = None) -> None:
        """
        Args:
            data_dir (str): Directory containing the processed macro-region data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        # TODO: Load dataset index/metadata

    def __len__(self) -> int:
        # TODO: Return actual dataset size
        return 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a macro-region data sample. Tensor shapes/fields follow docs/model-spec.md's
        macro.onnx contract.

        Returns:
            dict: Sample containing prompt_tokens, region_x, region_z, seed, region_heightfield,
                  flavor_zone_ids, flavor_zone_weights, and structure_candidate.
        """
        # TODO: Load and return tensor data for the given index
        return {
            "prompt_tokens": torch.zeros(0),
            "region_x": torch.tensor(0),
            "region_z": torch.tensor(0),
            "seed": torch.tensor(0),
            "region_heightfield": torch.zeros(0),
            "flavor_zone_ids": torch.zeros(0),
            "flavor_zone_weights": torch.zeros(0),
            "structure_candidate": torch.zeros(0),
        }
