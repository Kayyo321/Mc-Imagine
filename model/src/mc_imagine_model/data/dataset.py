"""
PyTorch Dataset class for training the Mc-Imagine model.
"""

from typing import Dict, Any, Optional
import torch
from torch.utils.data import Dataset


class McImagineDataset(Dataset):
    """
    Dataset for loading prompt-conditioned chunk data.
    """

    def __init__(self, data_dir: str, transform: Optional[Any] = None) -> None:
        """
        Args:
            data_dir (str): Directory containing the processed chunk data.
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
        # TODO: Load and return tensor data for the given index
        return {
            "prompt_tokens": torch.zeros(0),
            "chunk_x": torch.tensor(0),
            "chunk_z": torch.tensor(0),
            "seed": torch.tensor(0),
            "capability_tier": "none",
            "heightmap": torch.zeros(0),
            "block_volume": torch.zeros(0),
            "biome_grid": torch.zeros(0),
            "structure_markers": torch.zeros(0),
            "structure_graph_nodes": torch.zeros(0),
            "structure_graph_edges": torch.zeros(0),
            "structure_origin": torch.zeros(0),
        }


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
