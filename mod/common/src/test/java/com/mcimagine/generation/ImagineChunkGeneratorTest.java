package com.mcimagine.generation;

import com.mcimagine.api.ModelInfo;
import com.mcimagine.model.ChunkOutput;
import com.mcimagine.model.FallbackTerrain;
import com.mcimagine.model.ModelSession;
import net.minecraft.SharedConstants;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Holder;
import net.minecraft.core.HolderLookup;
import net.minecraft.core.registries.Registries;
import net.minecraft.data.registries.VanillaRegistries;
import net.minecraft.server.Bootstrap;
import net.minecraft.world.level.ChunkPos;
import net.minecraft.world.level.LevelHeightAccessor;
import net.minecraft.world.level.NoiseColumn;
import net.minecraft.world.level.biome.Biomes;
import net.minecraft.world.level.biome.BiomeSource;
import net.minecraft.world.level.biome.FixedBiomeSource;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.chunk.ChunkAccess;
import net.minecraft.world.level.chunk.LevelChunkSection;
import net.minecraft.world.level.levelgen.Heightmap;
import net.minecraft.world.level.levelgen.NoiseGeneratorSettings;
import net.minecraft.world.level.levelgen.blending.Blender;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Executor;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;

public class ImagineChunkGeneratorTest {

    private static Holder<NoiseGeneratorSettings> overworldSettings;
    private static BiomeSource plainsBiomeSource;

    /**
     * {@code ImagineChunkGenerator} now requires a real {@code Holder<NoiseGeneratorSettings>} (it
     * delegates {@code applyCarvers}/{@code buildSurface}/{@code spawnOriginalMobs} to an internal
     * {@code NoiseBasedChunkGenerator} built from it - see that class's javadoc for why). Offline unit
     * tests obtain one, with no running server, via {@link VanillaRegistries#createLookup()} - the same
     * mechanism Minecraft's own datagen uses to bootstrap "world gen" registries.
     */
    @BeforeAll
    public static void setupMinecraft() {
        SharedConstants.tryDetectVersion();
        Bootstrap.bootStrap();

        HolderLookup.Provider registries = VanillaRegistries.createLookup();
        overworldSettings = registries.lookupOrThrow(Registries.NOISE_SETTINGS).getOrThrow(NoiseGeneratorSettings.OVERWORLD);
        Holder<net.minecraft.world.level.biome.Biome> plains = registries.lookupOrThrow(Registries.BIOME).getOrThrow(Biomes.PLAINS);
        plainsBiomeSource = new FixedBiomeSource(plains);
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

        ImagineChunkGenerator generator = new ImagineChunkGenerator(plainsBiomeSource, overworldSettings, modelsDir);
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

        ImagineChunkGenerator generator = new ImagineChunkGenerator(plainsBiomeSource, overworldSettings, modelsDir);

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

        ImagineChunkGenerator generator = new ImagineChunkGenerator(plainsBiomeSource, overworldSettings, modelsDir);

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

        ImagineChunkGenerator generator = new ImagineChunkGenerator(plainsBiomeSource, overworldSettings, modelsDir);

        // fillFromNoise now writes through LevelChunkSection.acquire()/setBlockState()/release() instead of
        // 98,304 individual ChunkAccess.setBlockState calls (docs/poc-plan.md Phase 3, bug #2/#4 fixes), so
        // the mock chunk needs its sections + heightmap-unprimed accessors stubbed rather than setBlockState.
        int minBuildHeight = -64;
        int maxBuildHeight = 320;
        int sectionCount = (maxBuildHeight - minBuildHeight) / 16;

        Map<Integer, BlockState[][][]> sectionWrites = new HashMap<>();
        LevelChunkSection[] sections = new LevelChunkSection[sectionCount];
        for (int i = 0; i < sectionCount; i++) {
            LevelChunkSection section = Mockito.mock(LevelChunkSection.class);
            BlockState[][][] local = new BlockState[16][16][16];
            Mockito.when(section.setBlockState(Mockito.anyInt(), Mockito.anyInt(), Mockito.anyInt(), any(BlockState.class), anyBoolean()))
                    .thenAnswer(invocation -> {
                        int lx = invocation.getArgument(0);
                        int ly = invocation.getArgument(1);
                        int lz = invocation.getArgument(2);
                        BlockState state = invocation.getArgument(3);
                        local[lx][ly][lz] = state;
                        return state;
                    });
            sectionWrites.put(i, local);
            sections[i] = section;
        }

        ChunkAccess mockChunk = Mockito.mock(ChunkAccess.class);
        Mockito.when(mockChunk.getPos()).thenReturn(new ChunkPos(1, 2));
        Mockito.when(mockChunk.getMinBuildHeight()).thenReturn(minBuildHeight);
        Mockito.when(mockChunk.getMaxBuildHeight()).thenReturn(maxBuildHeight);
        Mockito.when(mockChunk.getSections()).thenReturn(sections);
        Mockito.when(mockChunk.getSectionIndex(Mockito.anyInt()))
                .thenAnswer(invocation -> ((int) invocation.getArgument(0) - minBuildHeight) >> 4);
        Mockito.when(mockChunk.getOrCreateHeightmapUnprimed(Heightmap.Types.WORLD_SURFACE_WG))
                .thenReturn(Mockito.mock(Heightmap.class));
        Mockito.when(mockChunk.getOrCreateHeightmapUnprimed(Heightmap.Types.OCEAN_FLOOR_WG))
                .thenReturn(Mockito.mock(Heightmap.class));

        Executor directExecutor = Runnable::run;
        ChunkAccess result = generator.fillFromNoise(directExecutor, Blender.empty(), null, null, mockChunk).join();
        assertNotNull(result, "Resulting chunk should not be null");

        ChunkOutput output = generator.generateOrGetChunkOutput(1, 2);
        int expectedHeight = output.heightmap()[5 * 16 + 5];
        int sectionIndex = (expectedHeight - minBuildHeight) >> 4;
        int localY = (expectedHeight - minBuildHeight) & 15;

        BlockState topBlock = sectionWrites.get(sectionIndex)[5][localY][5];
        assertNotNull(topBlock, "Top block at local (5, 5) should be set");
        assertEquals(Blocks.GRASS_BLOCK.defaultBlockState(), topBlock, "Top block should be grass block driven by model inference");
    }

    @Test
    public void testFallbackChunkGenerationWithoutModel() {
        ImagineChunkGenerator generator = new ImagineChunkGenerator(plainsBiomeSource, overworldSettings, new ModelSession());
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

    // --- Sea-level water override (docs/phase2-plan.md §2 Phase 1.5) ---------------------------------
    // The `air && y <= seaLevel => water` rule is a compensation for FallbackTerrain's Java-side
    // synthesis, which only ever fills up to the surface. A spec-conformant model bakes water into its
    // own block_volume in-graph from its own predicted per-column water level (docs/model-spec.md), so
    // its air below sea level is authoritative - applying the override there floods every deep valley
    // and dry basin the model deliberately carved. These two tests pin both halves of that.

    private static final LevelHeightAccessor FULL_HEIGHT_LEVEL = new LevelHeightAccessor() {
        @Override public int getHeight() { return 384; }
        @Override public int getMinBuildHeight() { return -64; }
    };

    /** A model-shaped {@code block_volume}: solid up to {@code surfaceY}, air above, nothing else. */
    private static int[] solidUpTo(int surfaceY) {
        int[] volume = new int[16 * FallbackTerrain.HEIGHT_SPAN * 16];
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                for (int y = FallbackTerrain.MIN_Y; y <= surfaceY; y++) {
                    volume[FallbackTerrain.blockVolumeIndex(x, y - FallbackTerrain.MIN_Y, z)] = 1;
                }
            }
        }
        return volume;
    }

    private static ImagineChunkGenerator generatorReturning(ChunkOutput output) throws Exception {
        ModelSession session = Mockito.mock(ModelSession.class);
        Mockito.when(session.generateChunk(Mockito.anyInt(), Mockito.anyInt(), Mockito.anyLong(), any()))
                .thenReturn(output);
        return new ImagineChunkGenerator(plainsBiomeSource, overworldSettings, session);
    }

    @Test
    public void modelSuppliedAirBelowSeaLevelIsNotFloodedWithWater() throws Exception {
        int surfaceY = 40; // a whole chunk sitting 23 blocks below sea level, deliberately dry
        int[] heightmap = new int[256];
        Arrays.fill(heightmap, surfaceY);
        ChunkOutput modelOutput = new ChunkOutput(heightmap, solidUpTo(surfaceY), new int[4 * 96 * 4], new float[0][0]);

        ImagineChunkGenerator generator = generatorReturning(modelOutput);
        assertEquals(63, generator.getSeaLevel(), "sanity: overworld sea level is 63");

        NoiseColumn column = generator.getBaseColumn(8, 8, FULL_HEIGHT_LEVEL, null);

        assertNotEquals(Blocks.AIR.defaultBlockState(), column.getBlock(surfaceY), "sanity: solid terrain at the surface");
        for (int y = surfaceY + 1; y <= 63; y++) {
            assertEquals(Blocks.AIR.defaultBlockState(), column.getBlock(y),
                    "y=" + y + " is air in a model-authored block_volume below sea level and must stay air");
        }
    }

    @Test
    public void fallbackDerivedVolumeStillFillsToSeaLevelWithWater() {
        // The empty ModelSession routes through FallbackTerrain.generateFallbackChunkOutput, whose
        // synthesis stops at the surface - without the override its ocean basins would generate dry.
        ImagineChunkGenerator generator = new ImagineChunkGenerator(plainsBiomeSource, overworldSettings, new ModelSession());
        int seaLevel = generator.getSeaLevel();
        int[] fallbackHeights = FallbackTerrain.generateHeightmap(0, 0);

        int localX = -1;
        int localZ = -1;
        for (int x = 0; x < 16 && localX < 0; x++) {
            for (int z = 0; z < 16; z++) {
                if (fallbackHeights[z * 16 + x] < seaLevel - 1) {
                    localX = x;
                    localZ = z;
                    break;
                }
            }
        }
        assertTrue(localX >= 0, "fallback sine terrain should dip below sea level somewhere in chunk (0, 0)");

        int surfaceY = fallbackHeights[localZ * 16 + localX];
        NoiseColumn column = generator.getBaseColumn(localX, localZ, FULL_HEIGHT_LEVEL, null);

        assertNotEquals(Blocks.WATER.defaultBlockState(), column.getBlock(surfaceY), "the surface block itself stays solid");
        for (int y = surfaceY + 1; y <= seaLevel; y++) {
            assertEquals(Blocks.WATER.defaultBlockState(), column.getBlock(y),
                    "y=" + y + " is above fallback terrain and at/below sea level, so it must be water");
        }
        assertEquals(Blocks.AIR.defaultBlockState(), column.getBlock(seaLevel + 1), "nothing above sea level is flooded");
    }

    @Test
    public void malformedModelVolumeFallsBackAndRegainsTheWaterFill() throws Exception {
        // A wrong-length block_volume is replaced by FallbackTerrain.buildBlockVolumeFromHeightmap, so the
        // provenance flag must flip to "fallback" even though the ChunkOutput itself came from a model.
        int[] heightmap = new int[256];
        Arrays.fill(heightmap, 40);
        ChunkOutput malformed = new ChunkOutput(heightmap, new int[7], new int[4 * 96 * 4], new float[0][0]);

        ImagineChunkGenerator generator = generatorReturning(malformed);
        NoiseColumn column = generator.getBaseColumn(8, 8, FULL_HEIGHT_LEVEL, null);

        assertEquals(Blocks.WATER.defaultBlockState(), column.getBlock(50),
                "a heightmap-derived fallback volume must still be flooded up to sea level");
    }
}
