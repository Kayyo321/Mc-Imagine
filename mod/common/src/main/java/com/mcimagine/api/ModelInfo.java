package com.mcimagine.api;

public record ModelInfo(
    String id,
    String name,
    String version,
    String author,
    String description,
    String path,
    String onnxFilename
) {
    public ModelInfo(String name, String version, String author, String description, String path) {
        this(name, name, version, author, description, path, "model.onnx");
    }
}