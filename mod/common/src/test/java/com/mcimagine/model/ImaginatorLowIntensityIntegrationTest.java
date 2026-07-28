package com.mcimagine.model;

import com.mcimagine.api.ModelInfo;
import com.mcimagine.prompt.PromptTokenizer;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

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

    @Test
    public void trainedModelLoadsAndPassesTrialInference() {
        Path modelsDir = findModelsDir();
        assertNotNull(modelsDir, MODEL_FILE + " not found - run the Phase 6 export/package step first");

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
        assertEquals(8, caps.biomePalette().size(), "biome_palette should have 8 entries per the spec worked example");
        assertTrue(caps.blockPalette().get(0).equals("minecraft:air"), "index 0 of block_palette must be air");
    }

    @Test
    public void samePromptSameSeedProducesIdenticalOutput() throws Exception {
        Path modelsDir = findModelsDir();
        assertNotNull(modelsDir, MODEL_FILE + " not found");
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
        Path modelsDir = findModelsDir();
        assertNotNull(modelsDir, MODEL_FILE + " not found");
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
        Path modelsDir = findModelsDir();
        assertNotNull(modelsDir, MODEL_FILE + " not found");
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
