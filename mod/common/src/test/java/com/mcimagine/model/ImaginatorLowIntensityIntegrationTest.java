package com.mcimagine.model;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.mcimagine.api.ModelInfo;
import com.mcimagine.prompt.PromptTokenizer;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end integration test for the trained {@code imaginator-low_intensity-no_structures} model
 * produced by docs/poc-plan.md's Phase 4-6 (procedural data -> ImagineNet training -> ONNX export ->
 * .mcim packaging). This is the closest thing to an in-game verification that's automatable here: it
 * can't drive the actual world-creation GUI (no input-injection tool for the native LWJGL window,
 * per the Phase 0-3 agent's own report), but it drives the exact same code path
 * (ModelLoader -> ModelSession.generateChunk) the GUI would, against the real trained ONNX graph
 * rather than the dummy sine-wave model, and checks the properties that actually matter for the
 * PoC's stated goal ("same prompt+seed+model => same world; prompt actually reaches generation"):
 *
 * <ol>
 *   <li>the model is discovered, its capabilities parsed correctly, and it passes ModelLoader's
 *       load-time trial inference (proves the real ONNX graph, not just the dummy, loads cleanly);</li>
 *   <li>same prompt + same seed + same chunk -> bit-identical output (determinism);</li>
 *   <li>different prompts + same seed -> meaningfully different heightmaps (proves the prompt
 *       actually reaches the model - the exact bug docs/poc-plan.md opens with: "the prompt reaches
 *       nothing");</li>
 *   <li>different seeds + same prompt -> different output (proves SeedEncoder isn't ignored, the
 *       other bug the plan explicitly calls out).</li>
 * </ol>
 */
public class ImaginatorLowIntensityIntegrationTest {

    private static final String MODEL_FILE = "imaginator-low_intensity-no_structures-1.0.0.mcim";

    /**
     * Phase 2 grew {@code capabilities.biome_palette} from 8 to 12 entries (docs/phase2-plan.md §2
     * Phase 1.4) and simultaneously re-ranged the terrain head, so a {@code .mcim} declaring fewer than
     * this was trained against the old data contract entirely.
     */
    private static final int PHASE2_BIOME_PALETTE_SIZE = 12;

    private static Path findModelsDir() {
        Path[] candidates = new Path[]{
                Paths.get("fabric/run/mcimagine/models"),
                Paths.get("../fabric/run/mcimagine/models"),
                Paths.get("mod/fabric/run/mcimagine/models"),
                Paths.get("../../mod/fabric/run/mcimagine/models")
        };
        for (Path candidate : candidates) {
            if (Files.exists(candidate.resolve(MODEL_FILE))) {
                return candidate.toAbsolutePath();
            }
        }
        return null;
    }

    /**
     * Gate for every test in this class.
     *
     * <p>These assertions describe the <em>current-generation</em> model: its palette sizes, its height
     * band, and how far apart two prompts must land. Phase 2 invalidates the Day-1 {@code .mcim} by design
     * (docs/phase2-plan.md: "Phases 1-2 change the data contract... the Day-1 .mcim is invalidated by
     * design"), and that file is only replaced once retraining happens on the CUDA box. Failing here would
     * report a stale artefact as a Java defect, so the tests are skipped instead - via
     * {@link Assumptions#assumeTrue}, so they resurrect automatically the moment a retrained
     * {@code .mcim} is dropped into the models directory, with no code change and no risk of the coverage
     * being quietly deleted in the meantime.
     *
     * <p>The palette size is read straight out of the archive's {@code manifest.json} rather than through
     * {@code ModelLoader}, which would run a full 86 MB ONNX load plus trial inference just to decide
     * whether to skip.
     */
    private static Path requireCurrentGenerationModel() {
        Path modelsDir = findModelsDir();
        Assumptions.assumeTrue(modelsDir != null,
                MODEL_FILE + " is not present - run the export/package step to enable this test");

        int declaredBiomes = declaredBiomePaletteSize(modelsDir.resolve(MODEL_FILE));
        Assumptions.assumeTrue(declaredBiomes >= PHASE2_BIOME_PALETTE_SIZE,
                MODEL_FILE + " declares a " + declaredBiomes + "-entry biome_palette; this is the stale Day-1 "
                        + "artefact, which Phase 2's data-contract change invalidates. Retrain and re-export "
                        + "(>= " + PHASE2_BIOME_PALETTE_SIZE + " biomes) and this test runs again automatically.");
        return modelsDir;
    }

    /** Reads {@code capabilities.biome_palette}'s length from a {@code .mcim}, or {@code -1} if unreadable. */
    private static int declaredBiomePaletteSize(Path mcimPath) {
        try (ZipFile zip = new ZipFile(mcimPath.toFile())) {
            ZipEntry manifest = zip.getEntry("manifest.json");
            if (manifest == null) {
                return -1;
            }
            try (InputStream is = zip.getInputStream(manifest);
                 InputStreamReader reader = new InputStreamReader(is, StandardCharsets.UTF_8)) {
                JsonObject root = JsonParser.parseReader(reader).getAsJsonObject();
                if (!root.has("capabilities")) {
                    return -1;
                }
                JsonObject caps = root.getAsJsonObject("capabilities");
                return caps.has("biome_palette") ? caps.getAsJsonArray("biome_palette").size() : -1;
            }
        } catch (Exception e) {
            return -1;
        }
    }

    @Test
    public void trainedModelLoadsAndPassesTrialInference() {
        Path modelsDir = requireCurrentGenerationModel();

        ModelLoader loader = new ModelLoader(modelsDir);
        List<ModelInfo> models = loader.discoverModels();

        ModelInfo trained = models.stream()
                .filter(m -> "imaginator-low_intensity-no_structures".equals(m.id()))
                .findFirst()
                .orElseThrow(() -> new AssertionError("Trained model not found among discovered models: " + models));

        assertTrue(trained.available(), "Trained model should pass ModelLoader's load-time trial inference");
        assertEquals("0.5.0", trained.formatVersion());

        ModelInfo.ModelCapabilities caps = trained.capabilities();
        assertEquals("low", caps.intensity());
        assertEquals("none", caps.structureSupport());
        assertFalse(caps.requiresMacroField());
        assertEquals(0, caps.detailPasses());
        assertEquals(128, caps.maxPromptTokens());
        assertEquals(13, caps.blockPalette().size(), "block_palette should have 13 entries (air + 12 real blocks)");
        assertTrue(caps.blockPalette().get(0).equals("minecraft:air"), "index 0 of block_palette must be air");

        // Phase 2 (docs/phase2-plan.md §2 Phase 1.4) added badlands/eroded_badlands/stony_peaks/beach so
        // mesa terrain stops labelling itself savanna and attracting vanilla savanna lake decoration.
        assertTrue(caps.biomePalette().size() >= PHASE2_BIOME_PALETTE_SIZE,
                "biome_palette should have at least " + PHASE2_BIOME_PALETTE_SIZE + " entries, got " + caps.biomePalette().size());
        assertTrue(caps.biomePalette().containsAll(List.of("minecraft:badlands", "minecraft:eroded_badlands",
                        "minecraft:stony_peaks", "minecraft:beach")),
                "biome_palette must carry the four Phase 2 additions: " + caps.biomePalette());
    }

    @Test
    public void samePromptSameSeedProducesIdenticalOutput() throws Exception {
        Path modelsDir = requireCurrentGenerationModel();
        Path mcimPath = modelsDir.resolve(MODEL_FILE);

        ModelLoader loader = new ModelLoader(modelsDir);
        String tokenizerJson = loader.extractTokenizerJson(mcimPath);
        assertNotNull(tokenizerJson, "tokenizer.json should be present in the .mcim");
        PromptTokenizer tokenizer = PromptTokenizer.fromTokenizerJson(tokenizerJson);
        assertTrue(tokenizer.vocabSize() > 1000);

        byte[] modelBytes = loader.extractModelBytes(mcimPath, "model.onnx");
        int[] tokens = tokenizer.tokenize("vast deep valleys, beautiful mountains overlooking water-filled craters", 128);

        try (ModelSession session = new ModelSession(modelBytes)) {
            ChunkOutput a = session.generateChunk(3, -2, 12345L, tokens);
            ChunkOutput b = session.generateChunk(3, -2, 12345L, tokens);

            assertArrayEquals(a.heightmap(), b.heightmap(), "Same prompt+seed+chunk must produce identical heightmap");
            assertArrayEquals(a.blockVolume(), b.blockVolume(), "Same prompt+seed+chunk must produce identical block_volume");
            assertArrayEquals(a.biomeGrid(), b.biomeGrid(), "Same prompt+seed+chunk must produce identical biome_grid");

            assertEquals(256, a.heightmap().length);
            assertEquals(16 * 384 * 16, a.blockVolume().length);
            for (int h : a.heightmap()) {
                assertTrue(h >= -64 && h <= 319, "heightmap value out of Minecraft world bounds: " + h);
            }
        }
    }

    @Test
    public void differentPromptsProduceMeaningfullyDifferentTerrain() throws Exception {
        Path modelsDir = requireCurrentGenerationModel();
        Path mcimPath = modelsDir.resolve(MODEL_FILE);

        ModelLoader loader = new ModelLoader(modelsDir);
        PromptTokenizer tokenizer = PromptTokenizer.fromTokenizerJson(loader.extractTokenizerJson(mcimPath));
        byte[] modelBytes = loader.extractModelBytes(mcimPath, "model.onnx");

        int[] snowTokens = tokenizer.tokenize("towering snow-capped peaks", 128);
        int[] desertTokens = tokenizer.tokenize("endless flat desert dunes", 128);
        assertFalse(java.util.Arrays.equals(snowTokens, desertTokens), "sanity: prompts must tokenize differently");

        try (ModelSession session = new ModelSession(modelBytes)) {
            ChunkOutput snow = session.generateChunk(0, 0, 777L, snowTokens);
            ChunkOutput desert = session.generateChunk(0, 0, 777L, desertTokens);

            double snowMean = average(snow.heightmap());
            double desertMean = average(desert.heightmap());

            assertFalse(java.util.Arrays.equals(snow.heightmap(), desert.heightmap()),
                    "'towering snow-capped peaks' and 'endless flat desert dunes' must not generate identical terrain "
                            + "- this is the core PoC deliverable, the plan's opening bug: 'the prompt reaches nothing'");
            assertTrue(snowMean > desertMean + 20,
                    "snow peaks (mean=" + snowMean + ") should be substantially higher elevation than desert dunes (mean=" + desertMean + ")");
        }
    }

    @Test
    public void differentSeedsProduceDifferentOutputForSamePrompt() throws Exception {
        Path modelsDir = requireCurrentGenerationModel();
        Path mcimPath = modelsDir.resolve(MODEL_FILE);

        ModelLoader loader = new ModelLoader(modelsDir);
        PromptTokenizer tokenizer = PromptTokenizer.fromTokenizerJson(loader.extractTokenizerJson(mcimPath));
        byte[] modelBytes = loader.extractModelBytes(mcimPath, "model.onnx");
        int[] tokens = tokenizer.tokenize("gentle rolling grassland", 128);

        try (ModelSession session = new ModelSession(modelBytes)) {
            ChunkOutput seedA = session.generateChunk(10, 10, 111L, tokens);
            ChunkOutput seedB = session.generateChunk(10, 10, 222L, tokens);

            assertFalse(java.util.Arrays.equals(seedA.blockVolume(), seedB.blockVolume()),
                    "Different seeds with the same prompt must produce different output "
                            + "- proves SeedEncoder genuinely varies with seed (the other bug the plan calls out)");
        }
    }

    private static double average(int[] arr) {
        long sum = 0;
        for (int v : arr) sum += v;
        return (double) sum / arr.length;
    }
}
