"""
Module for auto-labeling extracted/generated regions with text descriptions.

Provides templated captioning from a *known* generation parameter vector (the "procedural
bootstrap" data-sourcing strategy from PROJECT.md's "Model Training Pipeline" section) — turning
e.g. `archetype="valley_mountains_craters", relief_amplitude=88, ridge_sharpness=0.8, water_level=64`
into "vast, plunging valleys overlooking majestic mountain peaks and water-filled craters."

Templates are keyed to the same `data/world_generator.py` archetype names so the caption vocabulary
and the actual rendered terrain never drift apart — this is exactly the "check the labeler's
template-to-parameter mapping is actually discriminative" property docs/poc-plan.md's Phase 5 gate
calls out as the fix if the trained model collapses to the caption prior.

Roughly 40 templates (5-6 per archetype) x 3-5 synonym choices per slot gives thousands of distinct
surface forms without needing an LLM paraphrase pass. `ChunkLabeler.label_region` is the primary
entry point; `maybe_paraphrase` is a documented no-op standing in for the LLM-augmentation strategy
`config.yaml`'s `data.augmentation` flag names (see its docstring for why it's a no-op here).
"""

import random
from typing import Any, Dict, List


SYNONYMS: Dict[str, List[str]] = {
    "vast": ["vast", "endless", "sprawling", "immense", "far-reaching"],
    "endless": ["endless", "vast", "boundless", "unbroken"],
    "deep": ["deep", "plunging", "steep-sided", "yawning"],
    "valley": ["valley", "ravine", "canyon", "gorge"],
    "beautiful": ["beautiful", "stunning", "breathtaking", "majestic", "awe-inspiring"],
    "mountain": ["mountain", "peak", "summit", "ridge"],
    "overlook": ["overlooking", "surrounding", "cradling", "ringing"],
    "water": ["water", "lake", "meltwater", "pool"],
    "crater": ["crater", "basin", "hollow", "pit"],
    "desert": ["desert", "arid wasteland", "sun-scorched wasteland", "sandy waste"],
    "dune": ["dune", "sand dune", "sand ridge", "sand sea"],
    "flat": ["flat", "level", "endless flat"],
    "bone_dry": ["bone dry", "waterless", "parched"],
    "snow": ["snow-capped", "frost-covered", "ice-crowned", "glacier-streaked"],
    "towering": ["towering", "soaring", "sky-high", "colossal"],
    "peak": ["peak", "summit", "pinnacle", "spire"],
    "tropical": ["tropical", "sun-drenched", "palm-fringed"],
    "archipelago": ["archipelago", "island chain", "scatter of islands", "island cluster"],
    "shallow": ["shallow", "calm", "turquoise", "glassy"],
    "mesa": ["mesa", "plateau", "tableland", "butte"],
    "plateau": ["plateau", "tableland", "terrace", "mesa"],
    "red": ["red", "rust-colored", "ochre", "sun-baked red"],
    "eroded": ["eroded", "weathered", "wind-carved", "time-worn"],
    "rolling": ["rolling", "gently rolling", "undulating", "softly rolling"],
    "grassland": ["grassland", "meadow", "prairie", "grassy plain"],
    "gentle": ["gentle", "soft", "mild", "easy"],
    "swamp": ["swamp", "marsh", "bog", "mire"],
    "murky": ["murky", "muddy", "waterlogged", "sodden"],
    "forest": ["forest", "woodland", "timberland"],
    "taiga": ["taiga", "boreal forest", "conifer forest", "pine forest"],
    "savanna": ["savanna", "open grassland", "dry plain"],
    "scattered": ["scattered", "sparse", "widely spaced"],
}


def _fill(rng: random.Random, template: str, **overrides: str) -> str:
    class _Filler(dict):
        def __missing__(self, key: str) -> str:
            return rng.choice(SYNONYMS[key])

    filler = _Filler(overrides)
    return template.format_map(filler)


# --- templates, keyed by data/world_generator.py archetype name ---------------------------------
_TEMPLATES: Dict[str, List[str]] = {
    "valley_mountains_craters": [
        "{vast} {deep} {valley}s {overlook} {beautiful} {mountain}s and {water}-filled {crater}s",
        "{beautiful} {mountain} {peak}s {overlook} {vast}, {deep} {valley}s dotted with {water}-filled {crater}s",
        "a {vast} landscape of {deep} {valley}s beneath {towering} {mountain}s, {crater}s brimming with {water}",
        "{deep} {valley}s carved between {beautiful} {mountain} ranges, {water}-filled {crater}s scattered below",
        "{towering}, {beautiful} {mountain}s rising above {vast} {deep} {valley}s and shimmering {water}-filled {crater}s",
        "an {beautiful} range of {mountain}s cut by {deep} {valley}s, each {crater} pooling with {water}",
    ],
    "desert_dunes": [
        "{flat} {vast} {desert} {dune}s stretching to the horizon",
        "an {vast}, {flat} {desert} of rippling {dune}s, {bone_dry} and quiet",
        "{endless} {dune} fields across a {bone_dry} {desert}",
        "a {flat} {desert} landscape of wind-rippled {dune}s",
        "{vast} {bone_dry} {dune} seas beneath a relentless sun",
    ],
    "snow_peaks": [
        "{towering} {snow} {mountain} {peak}s piercing the sky",
        "a range of {beautiful} {snow} {peak}s, {towering} over the land below",
        "{snow} {mountain} {peak}s rising {towering} and {beautiful} into the clouds",
        "{towering}, {snow} {mountain} spires with sheer icy faces",
        "an {beautiful} row of {snow} {peak}s, jagged and {towering}",
    ],
    "tropical_archipelago": [
        "a {shallow} {tropical} {archipelago} of small islands",
        "{shallow}, {tropical} waters scattered with a {tropical} island {archipelago}",
        "a {tropical} {archipelago}, {shallow} turquoise water between sun-drenched islets",
        "{scattered} {tropical} islands rising from {shallow}, {beautiful} water",
        "a {beautiful} {tropical} island chain surrounded by {shallow} water",
    ],
    "mesa_plateaus": [
        "{eroded} {red} {mesa} {plateau}s",
        "{red}, {eroded} {mesa} country carved into flat-topped {plateau}s",
        "{eroded}, wind-carved {red} {mesa} {plateau}s stacked like terraces",
        "a {red} {mesa} landscape, {eroded} into stepped {plateau}s",
        "{towering} {red} {mesa} formations {eroded} into broad {plateau}s",
    ],
    "rolling_grassland": [
        "{gentle} {rolling} {grassland}",
        "a {gentle}, {rolling} {grassland} under open sky",
        "{rolling} {grassland} plains, {gentle} and open",
        "an {gentle} sweep of {rolling} {grassland}",
        "soft, {rolling} {grassland} hills stretching gently into the distance",
    ],
    "swamp": [
        "a {murky} {swamp} thick with still water",
        "a {vast} {murky} {swamp} of waterlogged ground",
        "{murky}, waterlogged {swamp} land dotted with pools",
        "a {swamp} of {murky} bogs and stagnant pools",
    ],
    "taiga_forest": [
        "a {vast} {taiga} of dense conifer {forest}",
        "rolling hills covered in dense {taiga} {forest}",
        "{vast}, {gentle} {taiga} {forest} country",
        "a cold {taiga} {forest}, quiet and dense",
    ],
    "savanna_plains": [
        "an open {savanna} plain with {scattered} trees",
        "{gentle} {savanna} plains under a wide sky",
        "{vast}, dry {savanna} grassland",
        "{rolling} {savanna} country, dry and open",
    ],
}


class ChunkLabeler:
    """
    Auto-labels macro-region/chunk data to provide text prompts for conditional generation.
    """

    def __init__(self, rng_seed: int = 0) -> None:
        self._rng = random.Random(rng_seed)

    def label_region(self, params: Dict[str, Any]) -> str:
        """
        Turns a known generation parameter vector (see `data/world_generator.py`'s `RegionParams`,
        passed here as a dict via `dataclasses.asdict`) into a templated caption.

        Args:
            params (dict): region parameter vector, must include "archetype" plus the numeric
                fields `RegionParams` declares.

        Returns:
            str: a natural-language caption describing the region's terrain.
        """
        archetype = params.get("archetype", "rolling_grassland")
        templates = _TEMPLATES.get(archetype, _TEMPLATES["rolling_grassland"])
        template = self._rng.choice(templates)
        caption = _fill(self._rng, template)

        # A couple of parameter-grounded modifiers, so captions aren't purely archetype-keyed —
        # the model also has to learn that *within* an archetype, more extreme parameters read as
        # more extreme language, not just "which of 9 buckets was this."
        extras: List[str] = []
        relief_amplitude = float(params.get("relief_amplitude", 0.0))
        erosion = float(params.get("erosion", 0.0))
        water_level = float(params.get("water_level", 0.0))
        base_height = float(params.get("base_height", 64.0))

        if archetype not in ("desert_dunes",) and relief_amplitude > 70:
            extras.append("with dramatic, extreme elevation changes")
        if archetype not in ("desert_dunes", "mesa_plateaus") and erosion > 0.6:
            extras.append("heavily eroded by ancient water flow")
        if water_level > base_height + 8 and archetype not in ("tropical_archipelago", "swamp"):
            extras.append("partially flooded by a large lake")

        if extras and self._rng.random() < 0.5:
            caption = caption + ", " + self._rng.choice(extras)

        return caption

    def label_chunk(self, chunk_data: Dict[str, Any]) -> str:
        """
        Generates a text label describing a single chunk. Chunks inherit their containing
        region's caption (terrain captions describe the region-scale landform the region was
        sampled to represent, not per-16x16-column detail), so this simply delegates to
        `label_region` when region parameters are available.

        Args:
            chunk_data (dict): Structured data for a single chunk; expected to carry a "params"
                key with the containing region's parameter vector, when available.

        Returns:
            str: Text description/prompt for the chunk.
        """
        params = chunk_data.get("params")
        if params:
            return self.label_region(params)
        return "a minecraft chunk"

    def label_region_batch(self, param_list: List[Dict[str, Any]]) -> List[str]:
        """
        Generates text labels for a list of regions.

        Args:
            param_list (list[dict]): List of region parameter-vector dictionaries.

        Returns:
            list[str]: Corresponding list of text descriptions.
        """
        return [self.label_region(p) for p in param_list]


def maybe_paraphrase(caption: str, augmentation_enabled: bool) -> str:
    """
    Stands in for data-sourcing strategy 4 ("Caption paraphrasing" in PROJECT.md's "Model Training
    Pipeline" section): an LLM-generated paraphrase pass over templated captions, gated by
    `config.yaml`'s `data.augmentation` flag.

    No LLM API is available in this training environment (no network-callable LLM endpoint is
    configured), so this is a **documented no-op pass-through**, not a fabricated network call:
    whether or not `augmentation_enabled` is set, the caption is returned unchanged. Wiring this up
    to an actual paraphrasing model is future work, not part of this PoC pass.
    """
    return caption
