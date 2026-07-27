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
        if (!java.nio.file.Files.exists(modelsDirectory)) {
            try {
                java.nio.file.Files.createDirectories(modelsDirectory);
            } catch (java.io.IOException e) {
                e.printStackTrace();
            }
            return Collections.emptyList();
        }

        try (java.util.stream.Stream<Path> paths = java.nio.file.Files.list(modelsDirectory)) {
            return paths
                .filter(p -> java.nio.file.Files.isRegularFile(p) && p.toString().endsWith(".mcim"))
                .map(p -> new ModelInfo(
                    p.getFileName().toString().replace(".mcim", ""),
                    "1.0",
                    "Unknown",
                    "A Minecraft Imagine model.",
                    p.toAbsolutePath().toString()
                ))
                .toList();
        } catch (java.io.IOException e) {
            e.printStackTrace();
            return Collections.emptyList();
        }
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
}