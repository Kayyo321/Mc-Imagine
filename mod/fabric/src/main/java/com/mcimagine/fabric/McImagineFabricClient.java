package com.mcimagine.fabric;

import com.mcimagine.ui.McImaginePresetEditor;
import net.fabricmc.api.ClientModInitializer;

/**
 * Fabric client-side entrypoint for Mc-Imagine.
 * Registers client-only features like the preset editor (screen registration).
 */
public class McImagineFabricClient implements ClientModInitializer {
    @Override
    public void onInitializeClient() {
        McImaginePresetEditor.register();
    }
}
