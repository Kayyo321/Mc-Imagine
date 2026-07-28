package com.mcimagine.model;

/**
 * One chunk's worth of decoded model output.
 *
 * @param blockVolumeFromFallback {@code true} when {@code blockVolume} was synthesized Java-side by
 *        {@link FallbackTerrain#buildBlockVolumeFromHeightmap} (or by
 *        {@link FallbackTerrain#generateFallbackChunkOutput}) rather than authored by the model.
 *        A fallback volume only ever fills up to the surface, so the consumer has to add water
 *        below sea level itself; a spec-conformant model bakes water into {@code block_volume}
 *        in-graph (docs/model-spec.md) and its air must be taken literally.
 */
public record ChunkOutput(
    int[] heightmap,
    int[] blockVolume,
    int[] biomeGrid,
    float[][] structureMarkers,
    boolean blockVolumeFromFallback
) {
    /** Convenience overload for model-authored volumes, which are the common case. */
    public ChunkOutput(int[] heightmap, int[] blockVolume, int[] biomeGrid, float[][] structureMarkers) {
        this(heightmap, blockVolume, biomeGrid, structureMarkers, false);
    }
}
