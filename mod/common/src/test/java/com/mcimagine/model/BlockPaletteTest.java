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

    /**
     * <b>Not a test of docs/phase4-plan.md §1.5's dressing rules.</b> The topmost-solid-run-only
     * surface-dressing rule (and the water rule below) is implemented entirely in Python and baked
     * into the ONNX graph at export time ({@code model/src/mc_imagine_model/export/export_onnx.py});
     * {@link BlockPalette} is a pure id-to-{@link net.minecraft.world.level.block.state.BlockState}
     * lookup table with no dressing or water logic of its own (see its class javadoc), so nothing on
     * the Java side can exercise that rule. A companion Python-side test covers the actual rule logic.
     *
     * <p>What this test legitimately covers: this hand-built {@code blockVolume} has the dressing rule
     * pre-applied to a multi-solid-run column (bedrock/stone/dirt/grass/water ids placed exactly where
     * §1.5's rule would place them) purely as a source of realistic ids, and then asserts that
     * {@link BlockPalette#get} maps each of those ids - including the "stone, not grass" id on the
     * *lower*, non-topmost run - to the {@link net.minecraft.world.level.block.state.BlockState} the
     * manifest says it should. That is: "{@code BlockPalette} correctly maps the block ids the overhang
     * dressing pipeline can emit for a multi-run column," not "the dressing pipeline dresses correctly."
     */
    @Test
    public void blockPaletteResolvesIdsThatTheOverhangDressingPipelineCanEmitForAMultiRunColumn() {
        BlockPalette palette = new BlockPalette(TRAINED_BLOCK_PALETTE);

        int minY = -64;
        int heightSpan = 384;
        int[] blockVolume = new int[16 * heightSpan * 16];

        int localX = 5;
        int localZ = 5;

        // Populate column with multiple solid runs & open-to-sky water, as export_onnx.py's dressing/water
        // rules would produce for a column with an overhang - see that module for the actual rule logic.
        // y = -64: bedrock
        int idxBedrock = (localX * heightSpan + (-64 - minY)) * 16 + localZ;
        blockVolume[idxBedrock] = 1; // bedrock

        // y = -63..20: lower solid run (stone)
        for (int y = -63; y <= 20; y++) {
            int idx = (localX * heightSpan + (y - minY)) * 16 + localZ;
            blockVolume[idx] = 2; // stone
        }

        // y = 21..35: cave cavity under overhang (air, stays air even though y <= 62 sea level)
        for (int y = 21; y <= 35; y++) {
            int idx = (localX * heightSpan + (y - minY)) * 16 + localZ;
            blockVolume[idx] = 0; // air
        }

        // y = 36..55: upper solid run rock (stone)
        for (int y = 36; y <= 55; y++) {
            int idx = (localX * heightSpan + (y - minY)) * 16 + localZ;
            blockVolume[idx] = 2; // stone
        }

        // y = 56..59: subsurface dirt (topmost_y - 4 <= y <= topmost_y - 1)
        for (int y = 56; y <= 59; y++) {
            int idx = (localX * heightSpan + (y - minY)) * 16 + localZ;
            blockVolume[idx] = 4; // dirt
        }

        // y = 60: surface grass block (topmost_y = 60)
        int idxSurface = (localX * heightSpan + (60 - minY)) * 16 + localZ;
        blockVolume[idxSurface] = 5; // grass_block

        // y = 61..62: open-to-sky water below sea level 62
        for (int y = 61; y <= 62; y++) {
            int idx = (localX * heightSpan + (y - minY)) * 16 + localZ;
            blockVolume[idx] = 7; // water
        }

        // Verify BlockPalette resolves each id to the BlockState the manifest declares for it - this is a
        // palette-lookup assertion, not a re-derivation or verification of the dressing rule itself.
        assertEquals(Blocks.GRASS_BLOCK.defaultBlockState(), palette.get(blockVolume[idxSurface]), "id 5 (topmost solid cell's id) must resolve to grass block");
        assertEquals(Blocks.DIRT.defaultBlockState(), palette.get(blockVolume[(localX * heightSpan + (58 - minY)) * 16 + localZ]), "id 4 (subsurface cell's id) must resolve to dirt");
        assertEquals(Blocks.STONE.defaultBlockState(), palette.get(blockVolume[(localX * heightSpan + (45 - minY)) * 16 + localZ]), "id 2 (upper solid run core's id) must resolve to stone");
        assertEquals(Blocks.STONE.defaultBlockState(), palette.get(blockVolume[(localX * heightSpan + (20 - minY)) * 16 + localZ]), "id 2 (lower, non-topmost solid run's id) must also resolve to stone, not grass - pinning that the palette itself does not special-case 'topmost'; that is the pipeline's job, not this lookup table's");
        assertEquals(Blocks.AIR.defaultBlockState(), palette.get(blockVolume[(localX * heightSpan + (25 - minY)) * 16 + localZ]), "id 0 (cave-cavity cell's id) must resolve to air");
        assertEquals(Blocks.WATER.defaultBlockState(), palette.get(blockVolume[(localX * heightSpan + (61 - minY)) * 16 + localZ]), "id 7 (open-to-sky water cell's id) must resolve to water");
    }

    /**
     * <b>Not a test of docs/phase4-plan.md §1.5's water rule</b> (see the javadoc on
     * {@link #blockPaletteResolvesIdsThatTheOverhangDressingPipelineCanEmitForAMultiRunColumn} for why
     * that rule cannot be exercised from the Java side at all). This pins the same two ids from that
     * test - air (0) and water (7) - directly, as a minimal, fast, non-volume-shaped sanity check that
     * {@link BlockPalette} maps the two ids the water rule swaps between correctly. It is a strict
     * subset of the coverage above and exists only for that fast/minimal property; it does not exercise
     * any water/openness logic, because {@link BlockPalette} has none.
     */
    @Test
    public void blockPaletteResolvesTheAirAndWaterIdsTheWaterRuleSwapsBetween() {
        BlockPalette palette = new BlockPalette(TRAINED_BLOCK_PALETTE);

        // The id an air cell under an overhang would carry in a dressed volume (docs/model-spec.md).
        int airId = 0;
        // The id an open-to-sky, below-sea-level cell would carry in a dressed volume.
        int waterId = 7;

        assertEquals(Blocks.AIR.defaultBlockState(), palette.get(airId), "id 0 must resolve to air");
        assertEquals(Blocks.WATER.defaultBlockState(), palette.get(waterId), "id 7 must resolve to water");
    }
}
