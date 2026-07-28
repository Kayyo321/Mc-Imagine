package com.mcimagine.generation;

import com.mcimagine.model.ChunkOutput;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.LongFunction;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Bounded LRU cache for generated chunk outputs, replacing the previous unbounded
 * {@code ConcurrentHashMap<Long, ChunkOutput>} (PROJECT.md §"Performance & Caching" flags that as an
 * unbounded memory leak over a long play session). Backed by an access-ordered {@link LinkedHashMap} with
 * {@code removeEldestEntry}, sized generously by default - a few thousand entries comfortably covers
 * render-distance-scale exploration, per docs/model-spec.md's guidance that this tier needs nothing fancier.
 */
public class ChunkCache {

    private static final int DEFAULT_MAX_ENTRIES = 4096;

    private final int maxEntries;
    private final LinkedHashMap<Long, ChunkOutput> map;
    private final ReentrantLock lock = new ReentrantLock();

    public ChunkCache() {
        this(DEFAULT_MAX_ENTRIES);
    }

    public ChunkCache(int maxEntries) {
        this.maxEntries = Math.max(1, maxEntries);
        this.map = new LinkedHashMap<>(16, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<Long, ChunkOutput> eldest) {
                return size() > ChunkCache.this.maxEntries;
            }
        };
    }

    public ChunkOutput get(long key) {
        lock.lock();
        try {
            return map.get(key);
        } finally {
            lock.unlock();
        }
    }

    /**
     * Returns the cached value for {@code key}, computing (and caching) it via {@code supplier} if absent.
     * The (potentially slow, inference-driving) computation runs outside the lock so one chunk's generation
     * never blocks lookups for every other chunk.
     */
    public ChunkOutput computeIfAbsent(long key, LongFunction<ChunkOutput> supplier) {
        lock.lock();
        try {
            ChunkOutput existing = map.get(key);
            if (existing != null) {
                return existing;
            }
        } finally {
            lock.unlock();
        }

        ChunkOutput computed = supplier.apply(key);

        lock.lock();
        try {
            ChunkOutput racedWinner = map.get(key);
            if (racedWinner != null) {
                return racedWinner;
            }
            map.put(key, computed);
            return computed;
        } finally {
            lock.unlock();
        }
    }

    public void put(long key, ChunkOutput value) {
        lock.lock();
        try {
            map.put(key, value);
        } finally {
            lock.unlock();
        }
    }

    public void clear() {
        lock.lock();
        try {
            map.clear();
        } finally {
            lock.unlock();
        }
    }

    public int size() {
        lock.lock();
        try {
            return map.size();
        } finally {
            lock.unlock();
        }
    }
}
