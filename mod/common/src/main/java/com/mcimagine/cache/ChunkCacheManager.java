package com.mcimagine.cache;

/**
 * Manages chunk caching with configurable LRU size.
 */
public class ChunkCacheManager {
    private int cacheSize;

    public ChunkCacheManager(int cacheSize) {
        this.cacheSize = cacheSize;
    }
}