package com.mcimagine.model;

import com.mcimagine.McImagine;
import com.mcimagine.api.ModelInfo;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Process-wide registry of loaded {@link ModelSession}s, keyed by model id. A model is loaded once here and
 * shared by every {@code ImagineChunkGenerator}/{@code ImagineBiomeSource} instance that references it,
 * rather than each generator instance re-parsing and re-loading its own ONNX session (PROJECT.md
 * §"Interface Contracts" / docs/model-spec.md - one session per model per JVM, not one per generator).
 */
public final class ModelRegistry {

    private static final Map<String, ModelSession> SESSIONS = new ConcurrentHashMap<>();
    private static final Map<String, ModelInfo> INFOS = new ConcurrentHashMap<>();

    private ModelRegistry() {}

    /**
     * Returns the shared session for {@code modelId}, loading it from {@code modelsDirectory} the first
     * time it's requested. Returns {@link Optional#empty()} if no such model can be discovered/loaded.
     */
    public static synchronized Optional<ModelSession> getOrLoad(String modelId, Path modelsDirectory) {
        if (modelId == null || modelId.isBlank()) {
            return Optional.empty();
        }
        ModelSession existing = SESSIONS.get(modelId);
        if (existing != null) {
            return Optional.of(existing);
        }
        try {
            ModelLoader loader = new ModelLoader(modelsDirectory);
            List<ModelInfo> models = loader.discoverModels();
            ModelInfo match = models.stream()
                    .filter(m -> modelId.equals(m.id()))
                    .findFirst()
                    .orElse(null);
            if (match == null || !match.available()) {
                return Optional.empty();
            }
            byte[] bytes = loader.extractModelBytes(match);
            ModelSession session = new ModelSession(bytes);
            SESSIONS.put(modelId, session);
            INFOS.put(modelId, match);
            return Optional.of(session);
        } catch (Exception e) {
            McImagine.LOGGER.warn("ModelRegistry: failed to load model '{}': {}", modelId, e.getMessage());
            return Optional.empty();
        }
    }

    /** Registers an already-constructed session/info pair (e.g. one built directly from a resolved path). */
    public static void register(ModelInfo info, ModelSession session) {
        if (info == null || session == null) return;
        SESSIONS.put(info.id(), session);
        INFOS.put(info.id(), info);
    }

    public static Optional<ModelInfo> getInfo(String modelId) {
        return Optional.ofNullable(INFOS.get(modelId));
    }

    public static Optional<ModelSession> get(String modelId) {
        return Optional.ofNullable(SESSIONS.get(modelId));
    }

    /** Closes every loaded session. Intended to be called once, on server stop. */
    public static synchronized void closeAll() {
        for (Map.Entry<String, ModelSession> entry : SESSIONS.entrySet()) {
            try {
                entry.getValue().close();
            } catch (Exception e) {
                McImagine.LOGGER.warn("ModelRegistry: failed to close session for model '{}': {}", entry.getKey(), e.getMessage());
            }
        }
        SESSIONS.clear();
        INFOS.clear();
    }
}
