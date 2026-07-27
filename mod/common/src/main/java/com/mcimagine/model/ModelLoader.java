package com.mcimagine.model;

import com.mcimagine.api.ModelInfo;
import java.nio.file.Path;
import java.util.List;
import java.util.Collections;

/**
 * Scans the models directory, validates .mcim files, and manages ONNX Runtime sessions.
 */
public class ModelLoader {
    private final Path modelsDirectory;
    private boolean isLoaded = false;

    public ModelLoader(Path modelsDirectory) {
        this.modelsDirectory = modelsDirectory;
    }

    public List<ModelInfo> discoverModels() {
        return Collections.emptyList(); // TODO: Implement discovery
    }

    public void loadModel(ModelInfo model) {
        this.isLoaded = true;
    }

    public void unloadModel() {
        this.isLoaded = false;
    }

    public boolean isModelLoaded() {
        return isLoaded;
    }
}\n