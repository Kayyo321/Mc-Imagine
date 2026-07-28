package com.mcimagine.model;

import com.mcimagine.McImagine;
import net.minecraft.core.Holder;
import net.minecraft.core.Registry;
import net.minecraft.core.registries.Registries;
import net.minecraft.resources.ResourceKey;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.world.level.biome.Biome;
import net.minecraft.world.level.biome.Biomes;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Resolves {@code biome_grid} integer ids (docs/model-spec.md) through a model's
 * {@code capabilities.biome_palette} into {@link Holder}s of {@link Biome}, the exact mirror of
 * {@link BlockPalette}. Unknown/out-of-range ids fall back to {@code minecraft:plains}, warned once.
 */
public class BiomePalette {

    private final List<Holder<Biome>> resolved;
    private final Holder<Biome> fallback;
    private final AtomicBoolean warnedOutOfRange = new AtomicBoolean(false);

    private BiomePalette(List<Holder<Biome>> resolved, Holder<Biome> fallback) {
        this.resolved = resolved != null ? resolved : List.of();
        this.fallback = fallback;
    }

    /**
     * Resolves namespaced biome-id strings (in {@code capabilities.biome_palette} order) against the given
     * registry. A manifest omitting {@code biome_palette} yields an empty palette; callers should treat that
     * as "biome assignment stays vanilla-default" per docs/model-spec.md's Versioning note.
     */
    public static BiomePalette resolveFrom(List<String> paletteIds, Registry<Biome> biomeRegistry) {
        Holder<Biome> plains = biomeRegistry.getHolderOrThrow(Biomes.PLAINS);
        List<Holder<Biome>> resolvedList = new ArrayList<>();
        AtomicBoolean warned = new AtomicBoolean(false);
        if (paletteIds != null) {
            for (String id : paletteIds) {
                resolvedList.add(resolveOne(id, biomeRegistry, plains, warned));
            }
        }
        return new BiomePalette(resolvedList, plains);
    }

    /** Wraps an already-resolved list (e.g. decoded straight off a {@code RegistryFixedCodec} in a CODEC). */
    public static BiomePalette of(List<Holder<Biome>> alreadyResolved, Holder<Biome> fallback) {
        return new BiomePalette(alreadyResolved, fallback);
    }

    private static Holder<Biome> resolveOne(String id, Registry<Biome> registry, Holder<Biome> plains, AtomicBoolean warned) {
        if (id == null || id.isBlank()) {
            return plains;
        }
        try {
            ResourceLocation loc = new ResourceLocation(id);
            ResourceKey<Biome> key = ResourceKey.create(Registries.BIOME, loc);
            return registry.getHolder(key).map(h -> (Holder<Biome>) h).orElseGet(() -> {
                warnUnknown(id, warned);
                return plains;
            });
        } catch (Exception e) {
            warnUnknown(id, warned);
            return plains;
        }
    }

    private static void warnUnknown(String id, AtomicBoolean warned) {
        if (warned.compareAndSet(false, true)) {
            McImagine.LOGGER.warn("BiomePalette: unknown biome id '{}' in capabilities.biome_palette; " +
                    "falling back to minecraft:plains for this and any further unknown ids", id);
        }
    }

    /** Resolves a raw {@code biome_grid} integer id to a biome {@link Holder}. */
    public Holder<Biome> get(int id) {
        if (resolved.isEmpty()) {
            return fallback;
        }
        if (id < 0 || id >= resolved.size()) {
            if (warnedOutOfRange.compareAndSet(false, true)) {
                McImagine.LOGGER.warn("BiomePalette: biome_grid id {} is out of range for the declared " +
                        "palette (size {}); falling back to minecraft:plains for this and any further out-of-range ids", id, resolved.size());
            }
            return fallback;
        }
        return resolved.get(id);
    }

    public boolean isEmpty() {
        return resolved.isEmpty();
    }

    public List<Holder<Biome>> palette() {
        return resolved;
    }

    public int size() {
        return resolved.size();
    }
}
