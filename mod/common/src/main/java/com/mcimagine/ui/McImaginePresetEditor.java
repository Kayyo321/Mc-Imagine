package com.mcimagine.ui;

import com.mcimagine.McImagine;
import net.minecraft.client.gui.screens.worldselection.PresetEditor;
import net.minecraft.resources.ResourceKey;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.core.registries.Registries;
import net.minecraft.world.level.levelgen.presets.WorldPreset;

import java.util.Optional;

/**
 * Registers a {@link PresetEditor} for the Mc-Imagine world preset.
 * <p>
 * This inserts an entry into the vanilla {@code PresetEditor.EDITORS} map so that the
 * "Customize" button on the Create World screen becomes active when the Mc-Imagine
 * world type is selected. Clicking it opens {@link McImagineCustomizeScreen}.
 */
public class McImaginePresetEditor {

    /** The ResourceKey for our world preset (matches data/mcimagine/worldgen/world_preset/imagine.json) */
    private static final ResourceKey<WorldPreset> IMAGINE_PRESET =
            ResourceKey.create(Registries.WORLD_PRESET, new ResourceLocation(McImagine.MOD_ID, "imagine"));

    /**
     * Registers the preset editor into the vanilla EDITORS map.
     * Must be called from a client-side initializer.
     */
    public static void register() {
        try {
            java.util.Map<Optional<ResourceKey<WorldPreset>>, PresetEditor> mutableEditors = new java.util.HashMap<>(PresetEditor.EDITORS);
            mutableEditors.put(
                    Optional.of(IMAGINE_PRESET),
                    (createWorldScreen, worldCreationContext) -> new McImagineCustomizeScreen(createWorldScreen)
            );

            java.lang.reflect.Field editorsField = PresetEditor.class.getDeclaredField("EDITORS"); // Mojang mapping name is EDITORS (or we can use mapping resolver if needed, but in dev it's EDITORS)
            java.lang.reflect.Field unsafeField = sun.misc.Unsafe.class.getDeclaredField("theUnsafe");
            unsafeField.setAccessible(true);
            sun.misc.Unsafe unsafe = (sun.misc.Unsafe) unsafeField.get(null);
            
            Object base = unsafe.staticFieldBase(editorsField);
            long offset = unsafe.staticFieldOffset(editorsField);
            unsafe.putObject(base, offset, mutableEditors);

            McImagine.LOGGER.info("Registered Mc-Imagine preset editor for 'Customize' button via Unsafe");
        } catch (Exception e) {
            McImagine.LOGGER.error("Failed to register Mc-Imagine preset editor", e);
        }
    }
}
