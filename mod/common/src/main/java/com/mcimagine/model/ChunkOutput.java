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
 *
 *        <p>It stays a boolean rather than growing into a provenance enum, and both of its consumers
 *        want exactly this one bit. {@code ImagineChunkGenerator#resolveBlockState} needs "does this
 *        volume stop at the surface", to decide the water fill.
 *        {@code ImagineChunkGenerator#validateBlockVolume} needs "did a model author this array", to
 *        decide whether a malformed volume is a model failure worth reporting to
 *        {@link VolumeFallbackLog} - and for that question the bit is sufficient, because it is the
 *        <em>reason</em> a volume was rejected (null, wrong length, empty output) that this record
 *        cannot carry, not the fact of it. Each reason is known at its own failure site, together with
 *        the chunk coordinates, and is reported there rather than travelled here.
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
