package com.mcimagine.api;

public record ModelInfo(
    String name,
    String version,
    String author,
    String description,
    String path
) {}