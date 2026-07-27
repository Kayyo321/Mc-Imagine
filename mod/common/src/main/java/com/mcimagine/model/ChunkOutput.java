package com.mcimagine.model;

public record ChunkOutput(
    int[] heightmap,
    int[] blockVolume,
    int[] biomeGrid,
    float[][] structureMarkers
) {}