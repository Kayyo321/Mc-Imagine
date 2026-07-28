package com.mcimagine.model;

import com.mcimagine.McImagine;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Resolves {@code block_volume} integer ids (docs/model-spec.md) through a model's
 * {@code capabilities.block_palette} into actual {@link BlockState}s via {@link BuiltInRegistries#BLOCK}.
 *
 * Index 0 is always air. Unknown/out-of-range ids fall back to stone, with only a single warning logged
 * per palette instance (not per-call spam).
 */
public class BlockPalette {

    private final BlockState[] resolved;
    private final AtomicBoolean warnedOutOfRange = new AtomicBoolean(false);
    private final AtomicBoolean warnedUnknownId = new AtomicBoolean(false);

    public BlockPalette(List<String> paletteIds) {
        List<BlockState> states = new ArrayList<>();
        // Raw block_volume id 0 is always air (docs/model-spec.md) - this is implicit and NOT itself an
        // entry in capabilities.block_palette. The manifest's list only enumerates ids 1..N, e.g. the
        // reference model's ["minecraft:stone", "minecraft:dirt", "minecraft:grass_block"] means
        // id 1 = stone, id 2 = dirt, id 3 = grass_block (matching the historical hardcoded switch this
        // class replaces), not id 0 = stone.
        states.add(Blocks.AIR.defaultBlockState());
        if (paletteIds != null) {
            for (String id : paletteIds) {
                states.add(resolveBlock(id));
            }
        }
        this.resolved = states.toArray(new BlockState[0]);
    }

    private BlockState resolveBlock(String namespacedId) {
        if (namespacedId == null || namespacedId.isBlank()) {
            return Blocks.STONE.defaultBlockState();
        }
        try {
            ResourceLocation loc = new ResourceLocation(namespacedId);
            Block block = BuiltInRegistries.BLOCK.get(loc);
            if (block == null || (block == Blocks.AIR && !"minecraft:air".equals(namespacedId))) {
                warnUnknown(namespacedId);
                return Blocks.STONE.defaultBlockState();
            }
            return block.defaultBlockState();
        } catch (Exception e) {
            warnUnknown(namespacedId);
            return Blocks.STONE.defaultBlockState();
        }
    }

    private void warnUnknown(String id) {
        if (warnedUnknownId.compareAndSet(false, true)) {
            McImagine.LOGGER.warn("BlockPalette: unknown block id '{}' in capabilities.block_palette; " +
                    "falling back to minecraft:stone for this and any further unknown ids", id);
        }
    }

    /** Resolves a raw {@code block_volume} integer id to a {@link BlockState}. Index 0 is always air. */
    public BlockState get(int id) {
        if (id == 0) {
            return Blocks.AIR.defaultBlockState();
        }
        if (id < 0 || id >= resolved.length) {
            if (warnedOutOfRange.compareAndSet(false, true)) {
                McImagine.LOGGER.warn("BlockPalette: block_volume id {} is out of range for the declared " +
                        "palette (size {}); falling back to minecraft:stone for this and any further out-of-range ids", id, resolved.length);
            }
            return Blocks.STONE.defaultBlockState();
        }
        return resolved[id];
    }

    public int size() {
        return resolved.length;
    }
}
