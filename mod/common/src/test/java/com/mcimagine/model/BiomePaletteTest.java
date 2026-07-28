package com.mcimagine.model;

import com.mojang.serialization.Lifecycle;
import net.minecraft.SharedConstants;
import net.minecraft.core.Holder;
import net.minecraft.core.HolderLookup;
import net.minecraft.core.MappedRegistry;
import net.minecraft.core.Registry;
import net.minecraft.core.registries.Registries;
import net.minecraft.data.registries.VanillaRegistries;
import net.minecraft.resources.ResourceKey;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.server.Bootstrap;
import net.minecraft.world.level.biome.Biome;
import net.minecraft.world.level.biome.Biomes;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Covers {@link BiomePalette} against the Phase 2 biome palette (docs/phase2-plan.md §2 Phase 1.4), which
 * grew from 8 to 12 entries. Day 1 had no badlands-family biome, so mesa terrain was forced to label itself
 * {@code savanna} and vanilla savanna lake decoration scattered water pools over otherwise-correct red mesa.
 *
 * <p>The four additions are {@code badlands}, {@code eroded_badlands}, {@code stony_peaks} and
 * {@code beach}. {@link #everyPaletteIdIsARealBiomeInThisMinecraftVersion()} is the load-bearing check:
 * it resolves each id against the real 1.20.1 biome registry, so a name the Python side got wrong (rather
 * than silently degrading to plains in-game) fails the build here.
 */
public class BiomePaletteTest {

    /** Must stay byte-identical to {@code model/src/mc_imagine_model/spec_constants.py}'s BIOME_PALETTE. */
    private static final List<String> PHASE2_BIOME_PALETTE = List.of(
            "minecraft:plains",           // 0
            "minecraft:desert",           // 1
            "minecraft:forest",           // 2
            "minecraft:snowy_plains",     // 3
            "minecraft:swamp",            // 4
            "minecraft:savanna",          // 5
            "minecraft:taiga",            // 6
            "minecraft:ocean",            // 7
            "minecraft:badlands",         // 8  - Phase 2
            "minecraft:eroded_badlands",  // 9  - Phase 2
            "minecraft:stony_peaks",      // 10 - Phase 2
            "minecraft:beach"             // 11 - Phase 2
    );

    private static HolderLookup.RegistryLookup<Biome> vanillaBiomes;

    @BeforeAll
    public static void bootstrapMinecraft() {
        SharedConstants.tryDetectVersion();
        Bootstrap.bootStrap();
        vanillaBiomes = VanillaRegistries.createLookup().lookupOrThrow(Registries.BIOME);
    }

    private static ResourceKey<Biome> key(String id) {
        return ResourceKey.create(Registries.BIOME, new ResourceLocation(id));
    }

    /**
     * Builds a real (frozen) {@code Registry<Biome>} holding exactly the requested ids, taking the biome
     * values straight out of vanilla's own datagen lookup. Ids that do not exist in this Minecraft version
     * are simply left out, which is what lets the unknown-id path be exercised.
     */
    private static Registry<Biome> registryOf(List<String> ids) {
        MappedRegistry<Biome> registry = new MappedRegistry<>(Registries.BIOME, Lifecycle.stable());
        List<String> withFallback = new ArrayList<>(ids);
        // BiomePalette.resolveFrom looks up minecraft:plains as its fallback, so it must always be present.
        if (!withFallback.contains("minecraft:plains")) {
            withFallback.add("minecraft:plains");
        }
        for (String id : withFallback) {
            ResourceKey<Biome> key = key(id);
            if (registry.getHolder(key).isPresent()) {
                continue;
            }
            vanillaBiomes.get(key).ifPresent(ref -> registry.register(key, ref.value(), Lifecycle.stable()));
        }
        return registry.freeze();
    }

    private static ResourceKey<Biome> keyOf(Holder<Biome> holder) {
        return holder.unwrapKey().orElseThrow(() -> new AssertionError("resolved holder has no registry key"));
    }

    @Test
    public void everyPaletteIdIsARealBiomeInThisMinecraftVersion() {
        for (String id : PHASE2_BIOME_PALETTE) {
            assertTrue(vanillaBiomes.get(key(id)).isPresent(),
                    "'" + id + "' is not a registered biome in this Minecraft version - the Python "
                            + "capabilities.biome_palette must be corrected, not silently substituted here");
        }
    }

    @Test
    public void resolvesAllTwelveEntriesIncludingThePhase2Additions() {
        BiomePalette palette = BiomePalette.resolveFrom(PHASE2_BIOME_PALETTE, registryOf(PHASE2_BIOME_PALETTE));

        assertEquals(12, palette.size(), "the palette length must follow the manifest, not a hardcoded 8");
        for (int id = 0; id < PHASE2_BIOME_PALETTE.size(); id++) {
            assertEquals(key(PHASE2_BIOME_PALETTE.get(id)), keyOf(palette.get(id)),
                    "biome_grid id " + id + " must resolve to " + PHASE2_BIOME_PALETTE.get(id));
        }

        // The four Phase 2 additions specifically, by name - the mesa->savanna mislabel is the bug being fixed.
        assertEquals(Biomes.BADLANDS, keyOf(palette.get(8)));
        assertEquals(Biomes.ERODED_BADLANDS, keyOf(palette.get(9)));
        assertEquals(Biomes.STONY_PEAKS, keyOf(palette.get(10)));
        assertEquals(Biomes.BEACH, keyOf(palette.get(11)));
        assertNotEquals(Biomes.SAVANNA, keyOf(palette.get(8)), "badlands must no longer collapse to savanna");
    }

    @Test
    public void idsBeyondThePaletteFallBackToPlainsWithoutThrowingOrSpamming() {
        BiomePalette palette = BiomePalette.resolveFrom(PHASE2_BIOME_PALETTE, registryOf(PHASE2_BIOME_PALETTE));

        // 12 is the first id past a 12-entry palette; a model trained against a larger palette than the
        // manifest declares must degrade, not crash, mid-chunk.
        assertEquals(Biomes.PLAINS, keyOf(palette.get(12)));
        assertEquals(Biomes.PLAINS, keyOf(palette.get(9999)));
        assertEquals(Biomes.PLAINS, keyOf(palette.get(-1)));

        // Called once per biome cell in-game, so the warning must be one-shot rather than per-call spam;
        // this at minimum pins that repeated out-of-range lookups stay cheap and never throw.
        for (int i = 0; i < 5000; i++) {
            assertNotNull(palette.get(12 + i));
        }
    }

    @Test
    public void anUnknownIdFallsBackToPlainsWithoutShiftingLaterEntries() {
        List<String> withBogus = new ArrayList<>(PHASE2_BIOME_PALETTE);
        withBogus.add(8, "minecraft:definitely_not_a_biome");

        BiomePalette palette = BiomePalette.resolveFrom(withBogus, registryOf(withBogus));

        assertEquals(13, palette.size(), "an unresolvable id must still occupy its slot");
        assertEquals(Biomes.PLAINS, keyOf(palette.get(8)), "unknown ids fall back to plains");
        assertEquals(Biomes.BADLANDS, keyOf(palette.get(9)), "later entries must not shift");
        assertEquals(Biomes.BEACH, keyOf(palette.get(12)));
    }

    @Test
    public void anAbsentPaletteIsEmptySoBiomeAssignmentStaysVanillaDefault() {
        BiomePalette palette = BiomePalette.resolveFrom(List.of(), registryOf(List.of()));
        assertTrue(palette.isEmpty());
        assertEquals(0, palette.size());
        assertEquals(Biomes.PLAINS, keyOf(palette.get(0)));
    }
}
