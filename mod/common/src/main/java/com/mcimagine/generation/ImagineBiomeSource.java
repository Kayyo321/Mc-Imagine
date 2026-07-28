package com.mcimagine.generation;

import com.mcimagine.model.BiomePalette;
import com.mcimagine.model.ChunkOutput;
import com.mojang.serialization.Codec;
import com.mojang.serialization.codecs.RecordCodecBuilder;
import net.minecraft.core.Holder;
import net.minecraft.core.registries.Registries;
import net.minecraft.resources.RegistryFixedCodec;
import net.minecraft.world.level.biome.Biome;
import net.minecraft.world.level.biome.BiomeSource;
import net.minecraft.world.level.biome.Climate;

import java.util.List;
import java.util.stream.Stream;

/**
 * A {@link BiomeSource} that reads {@code biome_grid} out of the same per-chunk inference cache
 * {@link ImagineChunkGenerator} populates, mapping ids through a model's {@code capabilities.biome_palette}
 * (docs/model-spec.md) via {@link BiomePalette}. Falls back to a single fallback biome everywhere if the
 * palette is empty (manifest omits {@code biome_palette} entirely - docs/model-spec.md's Versioning note
 * says biome assignment then "stays vanilla-default") or if this source hasn't been {@link #attachGenerator
 * attached} to a generator yet.
 *
 * <p>Codec-serializable so it round-trips through {@code level.dat}. Note the construction order:
 * {@link BiomeSource} instances are built <em>before</em> the {@link net.minecraft.world.level.chunk.ChunkGenerator}
 * that owns them (a {@code BiomeSource} is a field inside the generator's own record codec), so the link
 * back to the generator sharing the inference cache is completed post-construction via
 * {@link #attachGenerator}, called from {@code ImagineChunkGenerator}'s constructor.
 */
public class ImagineBiomeSource extends BiomeSource {

    public static final Codec<ImagineBiomeSource> CODEC = RecordCodecBuilder.create(instance ->
            instance.group(
                    RegistryFixedCodec.create(Registries.BIOME).listOf()
                            .fieldOf("biome_palette").forGetter(ImagineBiomeSource::getBiomePaletteHolders),
                    RegistryFixedCodec.create(Registries.BIOME)
                            .fieldOf("fallback_biome").forGetter(ImagineBiomeSource::getFallback)
            ).apply(instance, instance.stable(ImagineBiomeSource::new))
    );

    private final BiomePalette palette;
    private final Holder<Biome> fallback;
    private ImagineChunkGenerator owningGenerator;

    public ImagineBiomeSource(List<Holder<Biome>> biomePaletteHolders, Holder<Biome> fallback) {
        this.palette = BiomePalette.of(biomePaletteHolders, fallback);
        this.fallback = fallback;
    }

    private List<Holder<Biome>> getBiomePaletteHolders() {
        return palette.palette();
    }

    private Holder<Biome> getFallback() {
        return fallback;
    }

    /** Wires this biome source to the generator sharing its per-chunk inference cache. */
    public void attachGenerator(ImagineChunkGenerator generator) {
        this.owningGenerator = generator;
    }

    @Override
    protected Codec<? extends BiomeSource> codec() {
        return CODEC;
    }

    @Override
    protected Stream<Holder<Biome>> collectPossibleBiomes() {
        return palette.isEmpty() ? Stream.of(fallback) : palette.palette().stream();
    }

    @Override
    public Holder<Biome> getNoiseBiome(int quartX, int quartY, int quartZ, Climate.Sampler sampler) {
        if (owningGenerator == null || palette.isEmpty()) {
            return fallback;
        }

        // Quart coordinates are 1/4-block resolution; 4 quarts per 16-block chunk.
        int chunkX = Math.floorDiv(quartX, 4);
        int chunkZ = Math.floorDiv(quartZ, 4);
        int localQx = Math.floorMod(quartX, 4);
        int localQz = Math.floorMod(quartZ, 4);

        ChunkOutput output = owningGenerator.generateOrGetChunkOutput(chunkX, chunkZ);
        int[] biomeGrid = output.biomeGrid();
        if (biomeGrid == null || biomeGrid.length == 0) {
            return fallback;
        }

        // biome_grid: int32[4, 96, 4] at quarter resolution, [x][y][z], y from quarter-minY (-64/4 = -16).
        int quarterMinY = com.mcimagine.model.FallbackTerrain.MIN_Y / 4;
        int yLocal = quartY - quarterMinY;
        if (yLocal < 0) yLocal = 0;
        if (yLocal >= 96) yLocal = 95;

        int index = (localQx * 96 + yLocal) * 4 + localQz;
        if (index < 0 || index >= biomeGrid.length) {
            return fallback;
        }
        return palette.get(biomeGrid[index]);
    }
}
