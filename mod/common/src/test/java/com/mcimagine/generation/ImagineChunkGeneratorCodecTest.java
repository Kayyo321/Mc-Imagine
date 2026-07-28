package com.mcimagine.generation;

import com.google.gson.JsonElement;
import com.mojang.datafixers.util.Pair;
import com.mojang.serialization.DataResult;
import com.mojang.serialization.JsonOps;
import net.minecraft.SharedConstants;
import net.minecraft.core.Holder;
import net.minecraft.core.HolderLookup;
import net.minecraft.core.registries.Registries;
import net.minecraft.data.registries.VanillaRegistries;
import net.minecraft.resources.RegistryOps;
import net.minecraft.server.Bootstrap;
import net.minecraft.world.level.biome.Biomes;
import net.minecraft.world.level.biome.BiomeSource;
import net.minecraft.world.level.biome.FixedBiomeSource;
import net.minecraft.world.level.levelgen.NoiseGeneratorSettings;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Phase 2 gate (docs/poc-plan.md): "the prompt+model_id round-trip through the CODEC" - verified here by
 * serializing/deserializing {@link ImagineChunkGenerator#CODEC} directly rather than relying on a live
 * world quit-and-reload, which isn't exercisable headless.
 */
public class ImagineChunkGeneratorCodecTest {

    private static RegistryOps<JsonElement> registryOps;
    private static Holder<NoiseGeneratorSettings> overworldSettings;
    private static BiomeSource plainsBiomeSource;

    @BeforeAll
    public static void setup() {
        SharedConstants.tryDetectVersion();
        Bootstrap.bootStrap();

        HolderLookup.Provider registries = VanillaRegistries.createLookup();
        registryOps = RegistryOps.create(JsonOps.INSTANCE, registries);
        overworldSettings = registries.lookupOrThrow(Registries.NOISE_SETTINGS).getOrThrow(NoiseGeneratorSettings.OVERWORLD);
        plainsBiomeSource = new FixedBiomeSource(registries.lookupOrThrow(Registries.BIOME).getOrThrow(Biomes.PLAINS));
    }

    @Test
    public void promptModelIdAndSeedRoundTripThroughCodec() {
        String prompt = "vast deep valleys, beautiful mountains overlooking water-filled craters";
        String modelId = "imaginator-low_intensity-no_structures";
        long seed = 123456789L;

        ImagineChunkGenerator original = new ImagineChunkGenerator(plainsBiomeSource, overworldSettings, prompt, modelId, seed);

        DataResult<JsonElement> encodeResult = ImagineChunkGenerator.CODEC.encodeStart(registryOps, original);
        assertTrue(encodeResult.result().isPresent(), () -> "CODEC encode failed: " + encodeResult.error());
        JsonElement encoded = encodeResult.result().get();

        DataResult<Pair<ImagineChunkGenerator, JsonElement>> decodeResult = ImagineChunkGenerator.CODEC.decode(registryOps, encoded);
        assertTrue(decodeResult.result().isPresent(), () -> "CODEC decode failed: " + decodeResult.error());
        ImagineChunkGenerator decoded = decodeResult.result().get().getFirst();

        assertEquals(prompt, decoded.getPrompt(), "prompt must survive a CODEC round-trip (level.dat persistence)");
        assertEquals(modelId, decoded.getModelId(), "model_id must survive a CODEC round-trip");
        assertEquals(seed, decoded.getSeed(), "seed must survive a CODEC round-trip");
    }

    @Test
    public void emptyPromptAndModelIdDefaultCleanly() {
        ImagineChunkGenerator original = new ImagineChunkGenerator(plainsBiomeSource, overworldSettings);

        DataResult<JsonElement> encodeResult = ImagineChunkGenerator.CODEC.encodeStart(registryOps, original);
        assertTrue(encodeResult.result().isPresent(), () -> "CODEC encode failed: " + encodeResult.error());

        DataResult<Pair<ImagineChunkGenerator, JsonElement>> decodeResult =
                ImagineChunkGenerator.CODEC.decode(registryOps, encodeResult.result().get());
        assertTrue(decodeResult.result().isPresent(), () -> "CODEC decode failed: " + decodeResult.error());
        ImagineChunkGenerator decoded = decodeResult.result().get().getFirst();

        assertEquals("", decoded.getPrompt());
        assertEquals("", decoded.getModelId());
        assertEquals(0L, decoded.getSeed());
    }
}
