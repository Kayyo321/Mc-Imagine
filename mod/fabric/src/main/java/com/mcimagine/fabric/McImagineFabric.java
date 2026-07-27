package com.mcimagine.fabric;

import com.mcimagine.McImagine;
import net.fabricmc.api.ModInitializer;

public class McImagineFabric implements ModInitializer {
    @Override
    public void onInitialize() {
        McImagine.init();
    }
}\n