package com.mcimagine.model;

public record ChunkInput(
    int chunkX,
    int chunkZ,
    long seed,
    int[] promptTokens,
    byte[] neighborContext
) {}