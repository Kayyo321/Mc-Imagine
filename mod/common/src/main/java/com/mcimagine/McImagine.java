package com.mcimagine;

import com.mcimagine.generation.ImagineBiomeSource;
import com.mcimagine.generation.ImagineChunkGenerator;
import net.minecraft.core.Registry;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.resources.ResourceLocation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class McImagine {
    public static final String MOD_ID = "mcimagine";
    public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

    public static void init() {
        LOGGER.info("Initializing Mc-Imagine: AI-powered world generation");

        // Register the custom chunk generator codec
        Registry.register(BuiltInRegistries.CHUNK_GENERATOR, new ResourceLocation(MOD_ID, "imagine_generator"), ImagineChunkGenerator.CODEC);
        // Register the custom biome source codec (docs/poc-plan.md Phase 3 - biome_grid consumption)
        Registry.register(BuiltInRegistries.BIOME_SOURCE, new ResourceLocation(MOD_ID, "imagine_biome_source"), ImagineBiomeSource.CODEC);
    }
}