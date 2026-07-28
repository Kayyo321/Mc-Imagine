package com.mcimagine.model;

/**
 * Shared fallback terrain generator used both when {@link ModelSession} inference throws
 * (its own {@code catch} block) and when {@code ImagineChunkGenerator} has no {@link ModelSession} at all
 * (e.g. the empty no-arg {@code ModelSession}). Previously this sine-wave terrain was duplicated in both
 * places (PROJECT.md §"Error Handling & Fallback Chain" flags this); consolidated here so the two callers
 * can't drift apart.
 *
 * <p>{@code block_volume} is laid out {@code [x][y][z]} with {@code y} indexed from {@code minY = -64},
 * matching docs/model-spec.md's 0.5.0-pinned axis order exactly, so downstream decoding
 * (ImagineChunkGenerator#fillFromNoise) can use one indexing scheme regardless of whether the data came
 * from a real model or this fallback.
 */
public final class FallbackTerrain {

    public static final int MIN_Y = -64;
    public static final int MAX_Y = 320;
    public static final int HEIGHT_SPAN = MAX_Y - MIN_Y; // 384

    private FallbackTerrain() {}

    /** Flat index into a {@code [16][HEIGHT_SPAN][16]} ({@code [x][y][z]}) block volume array. */
    public static int blockVolumeIndex(int localX, int yIndex, int localZ) {
        return (localX * HEIGHT_SPAN + yIndex) * 16 + localZ;
    }

    public static int[] generateHeightmap(int chunkX, int chunkZ) {
        int[] heightmap = new int[256];
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                int globalX = chunkX * 16 + x;
                int globalZ = chunkZ * 16 + z;
                int h = 60 + (int) (Math.sin(globalX / 10.0) * 5.0 + Math.cos(globalZ / 10.0) * 5.0);
                heightmap[z * 16 + x] = h;
            }
        }
        return heightmap;
    }

    /** Synthesizes a stone/dirt/grass block volume from a heightmap, e.g. for legacy single-output models. */
    public static int[] buildBlockVolumeFromHeightmap(int[] heightmap) {
        int[] blockVolume = new int[16 * HEIGHT_SPAN * 16];
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                int h = heightmap[z * 16 + x];
                for (int y = MIN_Y; y <= h && y < MAX_Y; y++) {
                    int yIdx = y - MIN_Y;
                    int index = blockVolumeIndex(x, yIdx, z);
                    if (index >= 0 && index < blockVolume.length) {
                        if (y < h - 1) blockVolume[index] = 1; // stone
                        else if (y == h - 1) blockVolume[index] = 2; // dirt
                        else blockVolume[index] = 3; // grass block
                    }
                }
            }
        }
        return blockVolume;
    }

    public static ChunkOutput generateFallbackChunkOutput(int chunkX, int chunkZ) {
        int[] heightmap = generateHeightmap(chunkX, chunkZ);
        int[] blockVolume = buildBlockVolumeFromHeightmap(heightmap);
        return new ChunkOutput(heightmap, blockVolume, new int[4 * 96 * 4], new float[0][0], true);
    }
}
