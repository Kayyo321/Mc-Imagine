package com.mcimagine.model;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.mcimagine.api.ModelInfo;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/**
 * Scans the models directory, validates .mcim files, parses manifests, and manages ONNX Runtime model extraction.
 */
public class ModelLoader {
    private final Path modelsDirectory;
    private boolean isLoaded = false;

    public ModelLoader(Path modelsDirectory) {
        this.modelsDirectory = modelsDirectory;
    }

    public List<ModelInfo> discoverModels() {
        if (!Files.exists(modelsDirectory)) {
            try {
                Files.createDirectories(modelsDirectory);
            } catch (IOException e) {
                e.printStackTrace();
            }
            return Collections.emptyList();
        }

        try (Stream<Path> paths = Files.list(modelsDirectory)) {
            return paths
                    .filter(p -> Files.isRegularFile(p) && p.toString().endsWith(".mcim"))
                    .map(this::parseModelInfo)
                    .filter(Objects::nonNull)
                    .toList();
        } catch (IOException e) {
            e.printStackTrace();
            return Collections.emptyList();
        }
    }

    public ModelInfo parseModelInfo(Path mcimPath) {
        String fileName = mcimPath.getFileName().toString();
        String defaultName = fileName.endsWith(".mcim") ? fileName.substring(0, fileName.length() - 5) : fileName;
        String absPath = mcimPath.toAbsolutePath().toString();

        if (!Files.exists(mcimPath)) {
            return null;
        }

        try {
            if (Files.size(mcimPath) == 0) {
                return new ModelInfo(defaultName, defaultName, "1.0.0", "Unknown", "A Minecraft Imagine model.", absPath, "model.onnx");
            }
        } catch (IOException ignored) {}

        try (ZipFile zipFile = new ZipFile(mcimPath.toFile())) {
            ZipEntry manifestEntry = zipFile.getEntry("manifest.json");
            if (manifestEntry == null) {
                manifestEntry = zipFile.stream()
                        .filter(e -> !e.isDirectory() && e.getName().endsWith("manifest.json"))
                        .findFirst()
                        .orElse(null);
            }

            if (manifestEntry != null) {
                try (InputStream is = zipFile.getInputStream(manifestEntry);
                     InputStreamReader reader = new InputStreamReader(is, StandardCharsets.UTF_8)) {
                    JsonObject json = JsonParser.parseReader(reader).getAsJsonObject();

                    String id = getString(json, "id", defaultName);
                    String name = getString(json, "name", defaultName);
                    String version = getString(json, "version", "1.0.0");
                    String author = getString(json, "author", "Unknown");
                    String description = getString(json, "description", "A Minecraft Imagine model.");
                    String onnxFilename = getString(json, "onnx_filename", "model.onnx");

                    if (json.has("model") && json.get("model").isJsonObject()) {
                        JsonObject modelObj = json.getAsJsonObject("model");
                        if (id.equals(defaultName)) id = getString(modelObj, "id", id);
                        if (name.equals(defaultName)) name = getString(modelObj, "name", name);
                        if ("1.0.0".equals(version)) version = getString(modelObj, "version", version);
                        if ("Unknown".equals(author)) author = getString(modelObj, "author", author);
                        if ("A Minecraft Imagine model.".equals(description)) description = getString(modelObj, "description", description);
                        if ("model.onnx".equals(onnxFilename)) onnxFilename = getString(modelObj, "onnx_filename", onnxFilename);
                    }

                    return new ModelInfo(id, name, version, author, description, absPath, onnxFilename);
                }
            }
        } catch (Exception e) {
            // Log or ignore zip read error for empty/corrupt files and fall back
        }

        return new ModelInfo(defaultName, defaultName, "1.0.0", "Unknown", "A Minecraft Imagine model.", absPath, "model.onnx");
    }

    public byte[] extractModelBytes(Path mcimPath, String onnxFilename) throws IOException {
        try (ZipFile zipFile = new ZipFile(mcimPath.toFile())) {
            ZipEntry entry = zipFile.getEntry(onnxFilename);
            if (entry == null) {
                entry = zipFile.stream()
                        .filter(e -> !e.isDirectory() && (e.getName().equalsIgnoreCase(onnxFilename)
                                || e.getName().endsWith("/" + onnxFilename)
                                || e.getName().endsWith(".onnx")))
                        .findFirst()
                        .orElseThrow(() -> new IOException("ONNX model file '" + onnxFilename + "' not found in " + mcimPath));
            }
            try (InputStream is = zipFile.getInputStream(entry)) {
                return is.readAllBytes();
            }
        }
    }

    public byte[] extractModelBytes(ModelInfo modelInfo) throws IOException {
        return extractModelBytes(Path.of(modelInfo.path()), modelInfo.onnxFilename());
    }

    public byte[] extractModelBytes(Path mcimPath) throws IOException {
        ModelInfo info = parseModelInfo(mcimPath);
        String filename = info != null ? info.onnxFilename() : "model.onnx";
        return extractModelBytes(mcimPath, filename);
    }

    public InputStream extractModelInputStream(Path mcimPath, String onnxFilename) throws IOException {
        return new java.io.ByteArrayInputStream(extractModelBytes(mcimPath, onnxFilename));
    }

    private String getString(JsonObject json, String memberName, String fallback) {
        if (json.has(memberName) && !json.get(memberName).isJsonNull()) {
            return json.get(memberName).getAsString();
        }
        return fallback;
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