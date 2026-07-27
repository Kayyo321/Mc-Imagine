package com.mcimagine.api;

import java.util.List;
import java.util.Optional;

/**
 * Entry point for other mods to interact with Mc-Imagine.
 */
public interface McImagineApi {
    Optional<ModelInfo> getActiveModel();
    List<ModelInfo> getAvailableModels();
    String getActivePrompt();
}\n