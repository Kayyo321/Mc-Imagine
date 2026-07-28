package com.mcimagine.api;

import java.util.List;

public record ModelInfo(
    String id,
    String name,
    String version,
    String author,
    String description,
    String path,
    String onnxFilename,
    String formatVersion,
    ModelCapabilities capabilities,
    boolean available
) {
    public ModelInfo(String name, String version, String author, String description, String path) {
        this(name, name, version, author, description, path, "model.onnx", "0.1.0", ModelCapabilities.legacyDefaults(), true);
    }

    public ModelInfo(String id, String name, String version, String author, String description, String path, String onnxFilename) {
        this(id, name, version, author, description, path, onnxFilename, "0.1.0", ModelCapabilities.legacyDefaults(), true);
    }

    /** Returns a copy of this ModelInfo with the given availability flag. */
    public ModelInfo withAvailable(boolean isAvailable) {
        return new ModelInfo(id, name, version, author, description, path, onnxFilename, formatVersion, capabilities, isAvailable);
    }

    /**
     * Mirrors the {@code capabilities} block of a {@code manifest.json} (docs/model-spec.md), field-for-field.
     * Every field defaults sensibly for legacy (pre-0.5.0, or even pre-capabilities-block) manifests so that
     * older `.mcim` archives remain loadable without throwing.
     */
    public record ModelCapabilities(
        String intensity,
        String structureSupport,
        List<String> promptTags,
        List<String> blockPalette,
        List<String> biomePalette,
        int maxPromptTokens,
        boolean requiresMacroField,
        int detailPasses
    ) {
        public static ModelCapabilities legacyDefaults() {
            return new ModelCapabilities(
                "low",
                "none",
                List.of(),
                List.of(),
                List.of(),
                128,
                false,
                0
            );
        }
    }
}
