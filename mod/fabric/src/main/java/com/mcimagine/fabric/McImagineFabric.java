package com.mcimagine.fabric;

import com.mcimagine.McImagine;
import com.mcimagine.model.ModelRegistry;
import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;

public class McImagineFabric implements ModInitializer {
    @Override
    public void onInitialize() {
        McImagine.init();
        // Close every ModelRegistry-held ONNX session (but never the shared OrtEnvironment - see
        // ModelSession#close) when the server shuts down, so a world reload/JVM exit doesn't leak sessions.
        ServerLifecycleEvents.SERVER_STOPPING.register(server -> ModelRegistry.closeAll());
    }
}