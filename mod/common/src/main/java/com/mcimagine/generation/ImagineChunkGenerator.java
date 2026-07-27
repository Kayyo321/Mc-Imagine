package com.mcimagine.generation;

import com.mojang.serialization.Codec;
import com.mojang.serialization.codecs.RecordCodecBuilder;
import net.minecraft.core.BlockPos;
import net.minecraft.server.level.WorldGenRegion;
import net.minecraft.world.level.LevelHeightAccessor;
import net.minecraft.world.level.NoiseColumn;
import net.minecraft.world.level.StructureManager;
import net.minecraft.world.level.biome.BiomeManager;
import net.minecraft.world.level.biome.BiomeSource;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.chunk.ChunkAccess;
import net.minecraft.world.level.chunk.ChunkGenerator;
import net.minecraft.world.level.levelgen.GenerationStep;
import net.minecraft.world.level.levelgen.Heightmap;
import net.minecraft.world.level.levelgen.RandomState;
import net.minecraft.world.level.levelgen.blending.Blender;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;

/**
 * Replaces vanilla chunk generation with AI-powered generation (PoC).
 */
public class ImagineChunkGenerator extends ChunkGenerator {

    public static final Codec<ImagineChunkGenerator> CODEC = RecordCodecBuilder.create(instance ->
            instance.group(
                    BiomeSource.CODEC.fieldOf("biome_source").forGetter(ImagineChunkGenerator::getBiomeSource)
            ).apply(instance, instance.stable(ImagineChunkGenerator::new))
    );

    public ImagineChunkGenerator(BiomeSource biomeSource) {
        super(biomeSource);
    }

    @Override
    protected Codec<? extends ChunkGenerator> codec() {
        return CODEC;
    }

    @Override
    public void applyCarvers(WorldGenRegion region, long seed, RandomState randomState, BiomeManager biomeManager, StructureManager structureManager, ChunkAccess chunk, GenerationStep.Carving step) {
        // No carvers for now
    }

    @Override
    public void buildSurface(WorldGenRegion region, StructureManager structureManager, RandomState randomState, ChunkAccess chunk) {
        // Surface is built during fillFromNoise for this PoC
    }

    @Override
    public void spawnOriginalMobs(WorldGenRegion region) {
        // No mob spawning in PoC
    }

    @Override
    public int getGenDepth() {
        return 384;
    }

    @Override
    public CompletableFuture<ChunkAccess> fillFromNoise(Executor executor, Blender blender, RandomState randomState, StructureManager structureManager, ChunkAccess chunk) {
        return CompletableFuture.supplyAsync(() -> {
            int chunkX = chunk.getPos().x;
            int chunkZ = chunk.getPos().z;

            for (int x = 0; x < 16; x++) {
                for (int z = 0; z < 16; z++) {
                    int globalX = chunkX * 16 + x;
                    int globalZ = chunkZ * 16 + z;

                    int baseHeight = 60;
                    double wave = Math.sin(globalX / 10.0) * 5.0 + Math.cos(globalZ / 10.0) * 5.0;
                    int height = baseHeight + (int) wave;

                    for (int y = chunk.getMinBuildHeight(); y <= height; y++) {
                        if (y < height - 1) {
                            chunk.setBlockState(new BlockPos(x, y, z), Blocks.STONE.defaultBlockState(), false);
                        } else if (y == height - 1) {
                            chunk.setBlockState(new BlockPos(x, y, z), Blocks.DIRT.defaultBlockState(), false);
                        } else if (y == height) {
                            chunk.setBlockState(new BlockPos(x, y, z), Blocks.GRASS_BLOCK.defaultBlockState(), false);
                        }
                    }
                }
            }
            return chunk;
        }, executor);
    }

    @Override
    public int getSeaLevel() {
        return 63;
    }

    @Override
    public int getMinY() {
        return -64;
    }

    @Override
    public int getBaseHeight(int x, int z, Heightmap.Types type, LevelHeightAccessor level, RandomState randomState) {
        return 60 + (int)(Math.sin(x / 10.0) * 5.0 + Math.cos(z / 10.0) * 5.0);
    }

    @Override
    public NoiseColumn getBaseColumn(int x, int z, LevelHeightAccessor level, RandomState randomState) {
        int height = getBaseHeight(x, z, Heightmap.Types.OCEAN_FLOOR_WG, level, randomState);
        BlockState[] states = new BlockState[height - level.getMinBuildHeight()];
        
        for (int i = 0; i < states.length; i++) {
            int y = level.getMinBuildHeight() + i;
            if (y < height - 1) states[i] = Blocks.STONE.defaultBlockState();
            else if (y == height - 1) states[i] = Blocks.DIRT.defaultBlockState();
            else if (y == height) states[i] = Blocks.GRASS_BLOCK.defaultBlockState();
            else states[i] = Blocks.AIR.defaultBlockState();
        }
        return new NoiseColumn(level.getMinBuildHeight(), states);
    }

    @Override
    public void addDebugScreenInfo(List<String> list, RandomState randomState, BlockPos pos) {
    }
}