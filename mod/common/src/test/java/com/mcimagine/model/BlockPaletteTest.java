package com.mcimagine.model;

import net.minecraft.SharedConstants;
import net.minecraft.server.Bootstrap;
import net.minecraft.world.level.block.Blocks;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Covers {@link BlockPalette}'s id-to-{@code BlockState} mapping, in particular that the palette length is
 * driven entirely by the manifest (nothing hardcodes 13) and that the two manifest conventions for
 * "id 0 is air" both land on the same numbering.
 */
public class BlockPaletteTest {

    /**
     * Must stay byte-identical to {@code model/src/mc_imagine_model/spec_constants.py}'s BLOCK_PALETTE,
     * which is what the trained exporter writes into {@code capabilities.block_palette} - note it declares
     * {@code minecraft:air} explicitly at index 0.
     */
    private static final List<String> TRAINED_BLOCK_PALETTE = List.of(
            "minecraft:air",          // 0
            "minecraft:bedrock",      // 1
            "minecraft:stone",        // 2
            "minecraft:deepslate",    // 3
            "minecraft:dirt",         // 4
            "minecraft:grass_block",  // 5
            "minecraft:sand",         // 6
            "minecraft:water",        // 7
            "minecraft:red_sand",     // 8
            "minecraft:mud",          // 9
            "minecraft:gravel",       // 10
            "minecraft:snow_block",   // 11
            "minecraft:podzol"        // 12
    );

    @BeforeAll
    public static void bootstrapMinecraft() {
        SharedConstants.tryDetectVersion();
        Bootstrap.bootStrap();
    }

    /**
     * The trained exporter writes air explicitly at index 0, so the list is already 0-indexed. Prepending
     * another air would shift every id by one - mapping the model's bedrock to air, its stone to bedrock and
     * so on, which is exactly the "bedrock floor mid-world" the Day-1 PoC showed.
     */
    @Test
    public void explicitAirAtIndexZeroIsNotDoubleCounted() {
        BlockPalette palette = new BlockPalette(TRAINED_BLOCK_PALETTE);

        assertEquals(13, palette.size(), "palette size must follow the manifest exactly, air included");
        assertEquals(Blocks.AIR.defaultBlockState(), palette.get(0));
        assertEquals(Blocks.BEDROCK.defaultBlockState(), palette.get(1));
        assertEquals(Blocks.STONE.defaultBlockState(), palette.get(2));
        assertEquals(Blocks.DEEPSLATE.defaultBlockState(), palette.get(3));
        assertEquals(Blocks.WATER.defaultBlockState(), palette.get(7));
        assertEquals(Blocks.RED_SAND.defaultBlockState(), palette.get(8), "red_sand is the mesa/badlands surface");
        assertEquals(Blocks.PODZOL.defaultBlockState(), palette.get(12));
    }

    /**
     * docs/model-spec.md's worked example (and the legacy dummy-test-v1 fixture) omit air and enumerate ids
     * 1..N only, so air still has to be synthesized for those.
     */
    @Test
    public void implicitAirPaletteStillStartsAtIdOne() {
        BlockPalette palette = new BlockPalette(List.of("minecraft:stone", "minecraft:dirt", "minecraft:grass_block"));

        assertEquals(4, palette.size());
        assertEquals(Blocks.AIR.defaultBlockState(), palette.get(0));
        assertEquals(Blocks.STONE.defaultBlockState(), palette.get(1));
        assertEquals(Blocks.DIRT.defaultBlockState(), palette.get(2));
        assertEquals(Blocks.GRASS_BLOCK.defaultBlockState(), palette.get(3));
    }

    @Test
    public void idsBeyondThePaletteFallBackToStoneWithoutThrowingOrSpamming() {
        BlockPalette palette = new BlockPalette(TRAINED_BLOCK_PALETTE);

        assertEquals(Blocks.STONE.defaultBlockState(), palette.get(13), "first id past a 13-entry palette");
        assertEquals(Blocks.STONE.defaultBlockState(), palette.get(9999));
        assertEquals(Blocks.STONE.defaultBlockState(), palette.get(-1));

        // block_volume is 98,304 cells per chunk, so an out-of-range id must warn once and stay cheap.
        for (int i = 0; i < 5000; i++) {
            assertNotNull(palette.get(13 + i));
        }
    }

    @Test
    public void anUnknownBlockIdFallsBackToStoneWithoutShiftingLaterEntries() {
        List<String> withBogus = new ArrayList<>(TRAINED_BLOCK_PALETTE);
        withBogus.add(8, "minecraft:definitely_not_a_block");

        BlockPalette palette = new BlockPalette(withBogus);

        assertEquals(14, palette.size(), "an unresolvable id must still occupy its slot");
        assertEquals(Blocks.STONE.defaultBlockState(), palette.get(8));
        assertEquals(Blocks.RED_SAND.defaultBlockState(), palette.get(9), "later entries must not shift");
    }

    @Test
    public void anEmptyPaletteStillResolvesAir() {
        BlockPalette palette = new BlockPalette(List.of());
        assertEquals(1, palette.size());
        assertEquals(Blocks.AIR.defaultBlockState(), palette.get(0));
        assertEquals(Blocks.STONE.defaultBlockState(), palette.get(1));
    }
}
