package com.mcimagine.model;

import com.mcimagine.api.ModelInfo;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class ModelSessionTest {

    @Test
    public void testModelDiscoveryAndInference() throws Exception {
        Path[] candidatePaths = new Path[]{
                Paths.get("fabric/run/mcimagine/models"),
                Paths.get("../fabric/run/mcimagine/models"),
                Paths.get("mod/fabric/run/mcimagine/models"),
                Paths.get("../../mod/fabric/run/mcimagine/models")
        };

        Path modelsDir = null;
        for (Path candidate : candidatePaths) {
            if (Files.exists(candidate) && Files.exists(candidate.resolve("dummy-test-v1.mcim"))) {
                modelsDir = candidate.toAbsolutePath();
                break;
            }
        }

        assertNotNull(modelsDir, "dummy-test-v1.mcim models directory should be found in project");

        ModelLoader loader = new ModelLoader(modelsDir);
        List<ModelInfo> models = loader.discoverModels();
        assertFalse(models.isEmpty(), "Discovered models list should not be empty");

        ModelInfo dummyModel = models.stream()
                .filter(m -> "dummy-test-v1".equals(m.id()) || (m.path() != null && m.path().contains("dummy-test-v1")))
                .findFirst()
                .orElseThrow(() -> new AssertionError("dummy-test-v1.mcim model not found in discovered models list"));

        assertEquals("dummy-test-v1", dummyModel.id());
        assertEquals("Dummy Test Model", dummyModel.name());
        assertEquals("1.0.0", dummyModel.version());
        assertEquals("Mc-Imagine Team", dummyModel.author());
        assertEquals("model.onnx", dummyModel.onnxFilename());

        byte[] modelBytes = loader.extractModelBytes(dummyModel);
        assertNotNull(modelBytes, "Extracted ONNX bytes should not be null");
        assertTrue(modelBytes.length > 0, "Extracted ONNX bytes length should be greater than zero");

        try (ModelSession session = new ModelSession(modelBytes)) {
            int chunkX = 5;
            int chunkZ = 10;
            long seed = 12345L;
            int[] promptTokens = new int[128];

            ChunkOutput output = session.generateChunk(chunkX, chunkZ, seed, promptTokens);
            assertNotNull(output, "ChunkOutput should not be null");
            assertNotNull(output.heightmap(), "Heightmap array should not be null");
            assertEquals(256, output.heightmap().length, "Heightmap array should contain 256 elements (16x16)");

            boolean hasNonZero = false;
            for (int h : output.heightmap()) {
                if (h != 0) {
                    hasNonZero = true;
                    break;
                }
            }
            assertTrue(hasNonZero, "ChunkOutput heightmap should contain non-zero valid data");

            assertNotNull(output.blockVolume(), "Block volume array should not be null");
            assertTrue(output.blockVolume().length > 0, "Block volume array should not be empty");
        }
    }
}
