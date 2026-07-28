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

Roughly 50 templates (4-6 per archetype) x 3-5 synonym choices per slot, plus the
parameter-grounded modifiers `label_region` appends, gives thousands of distinct surface forms
without needing an LLM paraphrase pass. `ChunkLabeler.label_region` is the primary entry point;
`maybe_paraphrase` is a documented no-op standing in for the LLM-augmentation strategy
the training config's `data.augmentation` flag names (see its docstring for why it's a no-op here).

Two import-time assertions below (see "coverage guards") tie this module to
`world_generator.ARCHETYPES`: a new archetype without templates, or a template slot without
synonyms, is a hard failure rather than a silent fallback to `rolling_grassland` captions.
"""

import random
import string
from typing import Any, Dict, List

from mc_imagine_model.data.world_generator import ARCHETYPES


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
    # --- Phase 2: vocabulary for the `deep_canyon` and `fjord_rift` archetypes ------------------
    "canyon": ["canyon", "gorge", "chasm", "rift"],
    "plunging": ["plunging", "sheer", "vertiginous", "precipitous"],
    "arid": ["arid", "parched", "sun-baked", "rainless"],
    "terrace": ["terrace", "ledge", "bench", "stepped shelf"],
    "floor": ["floor", "bed", "bottom"],
    "fjord": ["fjord", "inlet", "sound", "sea-loch"],
    "cliff": ["cliff", "rock wall", "escarpment", "cliff face"],
    "narrow": ["narrow", "slender", "tight", "hemmed-in"],
    "cold": ["cold", "icy", "frigid", "glacial"],
    "deep_water": ["deep water", "black water", "fathomless water", "dark cold water"],
}


def _fill(rng: random.Random, template: str, **overrides: str) -> str:
    class _Filler(dict):
        def __missing__(self, key: str) -> str:
            return rng.choice(SYNONYMS[key])

    filler = _Filler(overrides)
    return template.format_map(filler)


def _tidy(text: str) -> str:
    """Cleans up the two artefacts that fall out of filling independent synonym slots: indefinite
    articles that disagree with whatever word the slot happened to draw ("an vast desert"), and an
    immediately repeated stem when two slots share a synonym ("{mesa} {plateau}s" -> "plateau
    plateaus"). Both were present in the Day-1 captions and are pure noise in the training text.
    """
    words = text.split(" ")
    out: List[str] = []
    for word in words:
        if out and _stem(out[-1]) and _stem(out[-1]) == _stem(word):
            out[-1] = word  # keep the later form, which carries the suffix/punctuation
            continue
        out.append(word)
    for i in range(len(out) - 1):
        lowered = out[i].lower()
        if lowered in ("a", "an"):
            nxt = out[i + 1].lstrip("\"'(")
            article = "an" if nxt[:1].lower() in "aeiou" else "a"
            out[i] = article.capitalize() if out[i][:1].isupper() else article
    return " ".join(out)


def _stem(word: str) -> str:
    core = word.strip(",.;:!?\"'()").lower()
    return core[:-1] if core.endswith("s") and len(core) > 3 else core


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
    # Phase 2 archetype. The discriminative signal a reader should pick up is *extreme depth* plus
    # *stepped, arid, red rock* — deep_canyon is the only archetype combining the project's largest
    # relief_amplitude/erosion with high plateau_quantization, so terraces and canyon floors far
    # below the rim are what separates it from `valley_mountains_craters` (wetter, unterraced) and
    # from `mesa_plateaus` (terraced but shallow).
    "deep_canyon": [
        "{vast} {arid} {canyon}s cut {deep} into {red} rock, their {floor}s far below the {plateau} rim",
        "a {plunging} {red} {canyon} system, {terrace}d walls dropping hundreds of blocks to a dry {floor}",
        "{deep}, {plunging} {canyon}s carved through {arid} {red} {plateau} country",
        "an {arid} tableland split open by {plunging} {canyon}s, {terrace} upon {terrace} of {red} rock down to the {floor}",
        "{vast} {red} {plateau}s torn open by {deep} {canyon}s with sheer, stepped walls",
        "a {bone_dry} {canyon} land of flat {plateau} tops and {plunging} drops to the {canyon} {floor}",
    ],
    # Phase 2 archetype. Discriminative signal: *steep walls plunging straight into deep cold
    # water*, in narrow inlets — the only archetype with both near-max ridge_sharpness/erosion and a
    # water_level sitting inside the carved relief, so the valleys it cuts come out drowned.
    "fjord_rift": [
        "{narrow} {fjord}s where {plunging} {cliff}s drop straight into {deep_water}",
        "a {cold} coast of {narrow} {fjord}s, {plunging} rock walls meeting {deep_water}",
        "{plunging} {cliff}-walled {fjord}s reaching far inland from a {cold} sea",
        "a {cold}, {narrow} {fjord} whose {cliff}s fall sheer into {deep_water}",
        "{deep} glacial {fjord}s, {narrow} arms of {deep_water} hemmed in by {plunging} {cliff}s",
        "a maze of {narrow}, {cold} {fjord}s between {towering} {plunging} {cliff}s",
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


# --- coverage guards ----------------------------------------------------------------------------
# `label_region` falls back to the `rolling_grassland` templates for an unknown archetype, which is
# the right behaviour for a stray/legacy params dict but is *silent* — before Phase 2 added
# `deep_canyon`/`fjord_rift` there was nothing stopping a new archetype from shipping thousands of
# regions captioned as gentle grassland. These two import-time assertions make that impossible:
# templates must cover ARCHETYPES exactly, and every `{slot}` used must resolve in SYNONYMS
# (an unknown slot would otherwise raise KeyError from `_Filler.__missing__` at data-gen time).
_missing = sorted(set(ARCHETYPES) - set(_TEMPLATES))
_extra = sorted(set(_TEMPLATES) - set(ARCHETYPES))
assert not _missing and not _extra, (
    f"labeler._TEMPLATES must cover world_generator.ARCHETYPES exactly; "
    f"missing templates for {_missing}, templates for unknown archetypes {_extra}"
)

_unknown_slots = sorted({
    field
    for templates in _TEMPLATES.values()
    for template in templates
    for _, field, _, _ in string.Formatter().parse(template)
    if field
} - set(SYNONYMS))
assert not _unknown_slots, f"labeler templates use slots absent from SYNONYMS: {_unknown_slots}"
del _missing, _extra, _unknown_slots


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

        # Parameter-grounded modifiers, so captions aren't purely archetype-keyed — the model also
        # has to learn that *within* an archetype, more extreme parameters read as more extreme
        # language, not just "which of 11 buckets was this."
        #
        # Thresholds recalibrated in Phase 2 against 200 freshly rendered regions. The Day-1
        # thresholds were tuned for the old, narrower parameter ranges and had become pure
        # archetype restatements once `relief_amplitude` reached 132: `relief_amplitude > 70` fired
        # for 78-99% of the four high-relief archetypes and 0% of everything else, `erosion > 0.6`
        # fired for 94-98% of the three canyon-family archetypes and 0-2% elsewhere, and
        # `water_level > base_height + 8` fired for **0%** of all 4000 sampled regions — dead code.
        # A modifier that is always-on inside an archetype and always-off outside it carries no
        # information the archetype name did not already carry.
        #
        # Every ladder below now splits *within* the archetypes it applies to (measured firing
        # rates in the comments), and the depth/flood terms key off quantities that demonstrably
        # predict the rendered terrain rather than off raw parameters.
        extras: List[str] = []
        relief_amplitude = float(params.get("relief_amplitude", 0.0))
        erosion = float(params.get("erosion", 0.0))
        water_level = float(params.get("water_level", 0.0))
        base_height = float(params.get("base_height", 64.0))

        # Estimated valley-floor height. `shaped` bottoms out near -0.24 for the ridge-heavy
        # archetypes and the erosion carve then subtracts up to `0.60 * erosion * amplitude` on top
        # (see world_generator.render_region and its VALLEY_MASK_SCALE note). Measured correlation
        # with the actual rendered per-region minimum is 0.64, and regions with est_floor < 25 have
        # a mean true minimum of 49.8 vs 62.1 for the rest — so "deep"/"plunging" language really
        # does track the parameters that produce depth, which raw `relief_amplitude` alone did not.
        est_floor = base_height - 0.24 * relief_amplitude - 0.60 * erosion * relief_amplitude
        # Fraction of the region's vertical span that ends up under water. Separates cleanly:
        # regions passing this test average 31% of columns submerged, regions failing it 0%.
        floods_low_ground = water_level > est_floor + 0.45 * (base_height - est_floor)

        # Relief ladder. Splits deep_canyon 44/56, fjord_rift 50/50, snow_peaks 32/64 across the
        # top two rungs; `mesa_plateaus` lands almost entirely on the third; the "level" rung is the
        # first modifier the flat archetypes have ever been eligible for.
        if relief_amplitude > 100:
            extras.append("with colossal, sheer changes in elevation")
        elif relief_amplitude > 68:
            extras.append("with dramatic, extreme elevation changes")
        elif relief_amplitude > 40:
            extras.append("with pronounced ridges and hollows")
        elif relief_amplitude > 18 and archetype == "desert_dunes":
            extras.append("its dunes piled high and steep")
        elif relief_amplitude < 12 and archetype != "swamp":
            # `swamp` is 92% below this threshold, so the modifier would be pure archetype
            # restatement there; desert_dunes gets its own wording (70%/12% low/high split) so that
            # the one archetype excluded from every other ladder still has *some* parameter
            # grounding rather than identical captions at amplitude 3 and amplitude 20.
            extras.append(
                "its dunes low and shallow" if archetype == "desert_dunes"
                else "barely rising above level ground"
            )

        # Erosion ladder. Excluded for the two archetypes whose templates already carry the word
        # ({bone_dry} desert, {eroded} mesa). Splits deep_canyon 56/44 and fjord_rift 20/70.
        if archetype not in ("desert_dunes", "mesa_plateaus"):
            if erosion > 0.85:
                extras.append("deeply eroded by ancient water flow")
            elif erosion > 0.55:
                extras.append("worn down by long erosion")
            elif erosion < 0.15 and archetype not in ("tropical_archipelago", "swamp"):
                # would contradict the template for the water-dominated archetypes
                extras.append("untouched by running water")

        # Depth. The `< 25` rung fires for 22% of deep_canyon and 40% of fjord_rift; regions on it
        # have a mean rendered minimum of 33.5, and 81% of regions on the `25..55` rung really do
        # dip below sea level. Gated on relief_amplitude so a flat archetype whose base_height
        # happens to sit low can never be captioned as having valley floors it does not have.
        if relief_amplitude > 25:
            if est_floor < 25:
                extras.append("cut by chasms that plunge far below sea level")
            elif est_floor < 55:
                extras.append("with valley floors dipping below sea level")

        # Flooding — replaces the never-firing `water_level > base_height + 8`. Excluded where the
        # templates already describe the water. Fires for 60% of fjord_rift, 32% of
        # valley_mountains_craters.
        if floods_low_ground and archetype not in ("tropical_archipelago", "swamp", "fjord_rift"):
            extras.append("its low ground drowned under deep water")

        # One modifier most of the time, occasionally two — enough to expose the whole modifier
        # vocabulary without turning every caption into a parameter dump.
        if extras:
            self._rng.shuffle(extras)
            n_extras = 2 if (len(extras) > 1 and self._rng.random() < 0.22) else (
                1 if self._rng.random() < 0.62 else 0
            )
            for extra in extras[:n_extras]:
                caption = caption + ", " + extra

        return _tidy(caption)

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
    the training config's `data.augmentation` flag.

    No LLM API is available in this training environment (no network-callable LLM endpoint is
    configured), so this is a **documented no-op pass-through**, not a fabricated network call:
    whether or not `augmentation_enabled` is set, the caption is returned unchanged. Wiring this up
    to an actual paraphrasing model is future work, not part of this PoC pass.
    """
    return caption
