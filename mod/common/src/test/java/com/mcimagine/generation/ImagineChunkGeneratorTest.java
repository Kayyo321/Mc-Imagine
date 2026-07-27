package com.mcimagine.generation;

import com.mcimagine.api.ModelInfo;
import com.mcimagine.model.ChunkOutput;
import com.mcimagine.model.ModelSession;
import net.minecraft.SharedConstants;
import net.minecraft.core.BlockPos;
import net.minecraft.server.Bootstrap;
import net.minecraft.world.level.ChunkPos;
import net.minecraft.world.level.LevelHeightAccessor;
import net.minecraft.world.level.NoiseColumn;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.chunk.ChunkAccess;
import net.minecraft.world.level.levelgen.Heightmap;
import net.minecraft.world.level.levelgen.blending.Blender;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Executor;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;

public class ImagineChunkGeneratorTest {

    @BeforeAll
    public static void setupMinecraft() {
        SharedConstants.tryDetectVersion();
        Bootstrap.bootStrap();
    }

    private Path findDummyModelPath() {
        Path[] candidatePaths = new Path[]{
                Paths.get("fabric/run/mcimagine/models"),
                Paths.get("../fabric/run/mcimagine/models"),
                Paths.get("mod/fabric/run/mcimagine/models"),
                Paths.get("../../mod/fabric/run/mcimagine/models")
        };
        for (Path candidate : candidatePaths) {
            if (Files.exists(candidate) && Files.exists(candidate.resolve("dummy-test-v1.mcim"))) {
                return candidate.toAbsolutePath();
            }
        }
        return null;
    }

    @Test
    public void testImagineChunkGeneratorModelInitialization() {
        Path modelsDir = findDummyModelPath();
        assertNotNull(modelsDir, "dummy-test-v1.mcim directory should be found in workspace");

        ImagineChunkGenerator generator = new ImagineChunkGenerator(null, modelsDir);
        assertTrue(generator.isModelLoaded(), "ImagineChunkGenerator should successfully load model");

        ModelInfo info = generator.getModelInfo();
        assertNotNull(info, "ModelInfo should not be null");
        assertEquals("dummy-test-v1", info.id());

        ModelSession session = generator.getModelSession();
        assertNotNull(session, "ModelSession should not be null");
    }

    @Test
    public void testGetBaseHeightDrivenByModelInference() {
        Path modelsDir = findDummyModelPath();
        assertNotNull(modelsDir, "dummy-test-v1.mcim directory should be found");

        ImagineChunkGenerator generator = new ImagineChunkGenerator(null, modelsDir);

        LevelHeightAccessor level = new LevelHeightAccessor() {
            @Override public int getHeight() { return 384; }
            @Override public int getMinBuildHeight() { return -64; }
        };

        int blockX = 8;
        int blockZ = 8;

        int height = generator.getBaseHeight(blockX, blockZ, Heightmap.Types.OCEAN_FLOOR_WG, level, null);
        assertTrue(height > 0, "Base height should be a positive height value");

        ChunkOutput output = generator.generateOrGetChunkOutput(0, 0);
        assertNotNull(output, "ChunkOutput should be generated from ModelSession");
        assertEquals(output.heightmap()[8 * 16 + 8], height, "getBaseHeight must match ModelSession output tensor height");
    }

    @Test
    public void testGetBaseColumnDrivenByModelInference() {
        Path modelsDir = findDummyModelPath();
        assertNotNull(modelsDir, "dummy-test-v1.mcim directory should be found");

        ImagineChunkGenerator generator = new ImagineChunkGenerator(null, modelsDir);

        LevelHeightAccessor level = new LevelHeightAccessor() {
            @Override public int getHeight() { return 384; }
            @Override public int getMinBuildHeight() { return -64; }
        };

        NoiseColumn column = generator.getBaseColumn(8, 8, level, null);
        assertNotNull(column, "NoiseColumn should not be null");

        int expectedHeight = generator.getBaseHeight(8, 8, Heightmap.Types.OCEAN_FLOOR_WG, level, null);

        BlockState topState = column.getBlock(expectedHeight);
        assertNotNull(topState, "Top block state should not be null");
        assertEquals(Blocks.GRASS_BLOCK.defaultBlockState(), topState, "Top ground block should be grass block");

        BlockState subState = column.getBlock(expectedHeight - 1);
        assertEquals(Blocks.DIRT.defaultBlockState(), subState, "Sub-surface block should be dirt");
    }

    @Test
    public void testFillFromNoiseWithModelInference() {
        Path modelsDir = findDummyModelPath();
        assertNotNull(modelsDir, "dummy-test-v1.mcim directory should be found");

        ImagineChunkGenerator generator = new ImagineChunkGenerator(null, modelsDir);

        ChunkAccess mockChunk = Mockito.mock(ChunkAccess.class);
        Map<BlockPos, BlockState> blockMap = new HashMap<>();

        Mockito.when(mockChunk.getPos()).thenReturn(new ChunkPos(1, 2));
        Mockito.when(mockChunk.getMinBuildHeight()).thenReturn(-64);
        Mockito.when(mockChunk.getMaxBuildHeight()).thenReturn(320);

        Mockito.when(mockChunk.setBlockState(any(BlockPos.class), any(BlockState.class), anyBoolean()))
                .thenAnswer(invocation -> {
                    BlockPos pos = invocation.getArgument(0);
                    BlockState state = invocation.getArgument(1);
                    blockMap.put(pos, state);
                    return state;
                });

        Executor directExecutor = Runnable::run;
        ChunkAccess result = generator.fillFromNoise(directExecutor, Blender.empty(), null, null, mockChunk).join();

        assertNotNull(result, "Resulting chunk should not be null");
        assertFalse(blockMap.isEmpty(), "Blocks should be set in chunk by fillFromNoise");

        ChunkOutput output = generator.generateOrGetChunkOutput(1, 2);
        int expectedHeight = output.heightmap()[5 * 16 + 5];
        BlockState topBlock = blockMap.get(new BlockPos(5, expectedHeight, 5));
        assertNotNull(topBlock, "Top block at local (5, 5) should be set");
        assertEquals(Blocks.GRASS_BLOCK.defaultBlockState(), topBlock, "Top block should be grass block driven by model inference");
    }

    @Test
    public void testFallbackChunkGenerationWithoutModel() {
        ImagineChunkGenerator generator = new ImagineChunkGenerator(null, new ModelSession());
        assertFalse(generator.isModelLoaded(), "Generator with empty ModelSession should report model not loaded");

        LevelHeightAccessor level = new LevelHeightAccessor() {
            @Override public int getHeight() { return 384; }
            @Override public int getMinBuildHeight() { return -64; }
        };

        int height = generator.getBaseHeight(10, 10, Heightmap.Types.OCEAN_FLOOR_WG, level, null);
        ChunkOutput output = generator.generateOrGetChunkOutput(0, 0);
        assertEquals(output.heightmap()[10 * 16 + 10], height, "Fallback height should match fallback heightmap output");

        NoiseColumn column = generator.getBaseColumn(10, 10, level, null);
        assertNotNull(column);
    }
}
