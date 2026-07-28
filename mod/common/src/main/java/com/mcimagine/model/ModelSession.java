package com.mcimagine.model;

import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import com.mcimagine.McImagine;
import com.mcimagine.api.ModelInfo;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Bridge between the mod and ONNX Runtime. Wraps an ONNX Runtime inference session for a single
 * per-chunk graph ({@code model.onnx}).
 */
public class ModelSession implements AutoCloseable {

    private static final int DEFAULT_MAX_TOKENS = 128;

    /** {@link OrtEnvironment} is a process-wide singleton (bug #3 fix) — never closed by an individual session. */
    private OrtEnvironment env;
    private OrtSession session;
    private boolean isClosed = false;

    public ModelSession() {
        // Default constructor - no model loaded, generateChunk falls back to FallbackTerrain.
    }

    public ModelSession(byte[] modelBytes) throws OrtException {
        if (modelBytes != null && modelBytes.length > 0) {
            this.env = OrtEnvironment.getEnvironment();
            this.session = env.createSession(modelBytes, new OrtSession.SessionOptions());
        }
    }

    public ModelSession(Path mcimPath) throws Exception {
        ModelLoader loader = new ModelLoader(mcimPath.getParent());
        byte[] modelBytes = loader.extractModelBytes(mcimPath);
        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(modelBytes, new OrtSession.SessionOptions());
    }

    public ModelSession(ModelInfo modelInfo) throws Exception {
        this(Path.of(modelInfo.path()));
    }

    public boolean isLoaded() {
        return session != null && env != null;
    }

    public Set<String> getInputNames() throws OrtException {
        return session != null ? session.getInputNames() : Set.of();
    }

    public Set<String> getOutputNames() throws OrtException {
        return session != null ? session.getOutputNames() : Set.of();
    }

    public ChunkOutput generateChunk(int chunkX, int chunkZ, long seed, int[] promptTokens) throws Exception {
        if (session == null || env == null) {
            return FallbackTerrain.generateFallbackChunkOutput(chunkX, chunkZ);
        }

        int maxTokens = DEFAULT_MAX_TOKENS;
        int[] tokens = new int[maxTokens];
        if (promptTokens != null && promptTokens.length > 0) {
            System.arraycopy(promptTokens, 0, tokens, 0, Math.min(promptTokens.length, maxTokens));
        }

        int[][] tokens2D = new int[][]{ tokens };
        int[] cx = new int[]{ chunkX };
        int[] cz = new int[]{ chunkZ };
        long[] s = new long[]{ seed };

        Map<String, OnnxTensor> inputs = new HashMap<>();
        Set<String> sessionInputNames = session.getInputNames();

        try {
            if (sessionInputNames.contains("prompt_tokens")) {
                inputs.put("prompt_tokens", OnnxTensor.createTensor(env, tokens2D));
            }
            if (sessionInputNames.contains("chunk_x")) {
                inputs.put("chunk_x", OnnxTensor.createTensor(env, cx));
            }
            if (sessionInputNames.contains("chunk_z")) {
                inputs.put("chunk_z", OnnxTensor.createTensor(env, cz));
            }
            if (sessionInputNames.contains("seed")) {
                inputs.put("seed", OnnxTensor.createTensor(env, s));
            }

            try (OrtSession.Result results = session.run(inputs)) {
                int[] heightmap = new int[256];
                int[] blockVolume = new int[0];
                int[] biomeGrid = new int[0];
                float[][] structureMarkers = new float[0][0];

                // Bug #1 fix: dispatch strictly by exact tensor name (docs/model-spec.md's pinned output
                // names), never "the first output of any name" — with a 3-output model that previously
                // misread block_volume as the heightmap.
                for (Map.Entry<String, OnnxValue> entry : results) {
                    String name = entry.getKey();
                    OnnxValue value = entry.getValue();
                    if (!(value instanceof OnnxTensor tensor)) {
                        continue;
                    }
                    switch (name) {
                        case "heightmap" -> heightmap = padOrTruncate(extractTensorAsInts(tensor, name), 256, heightmap);
                        case "block_volume" -> blockVolume = extractTensorAsInts(tensor, name);
                        case "biome_grid" -> biomeGrid = extractTensorAsInts(tensor, name);
                        default -> {
                            // Unknown/unsupported output for this tier (e.g. structure_* tensors on a
                            // "none" structure_support model) - ignore rather than misinterpret it.
                        }
                    }
                }

                // A legacy/heightmap-only graph gets a Java-synthesized volume, which fills only up to
                // the surface - the consumer has to flood it below sea level itself, so flag it as
                // fallback-derived (see ChunkOutput#blockVolumeFromFallback).
                boolean volumeFromFallback = false;
                if (blockVolume.length == 0 && heightmap.length > 0) {
                    blockVolume = FallbackTerrain.buildBlockVolumeFromHeightmap(heightmap);
                    volumeFromFallback = true;
                }
                if (biomeGrid.length == 0) {
                    biomeGrid = new int[4 * 96 * 4];
                }

                return new ChunkOutput(heightmap, blockVolume, biomeGrid, structureMarkers, volumeFromFallback);
            }
        } finally {
            for (OnnxTensor tensor : inputs.values()) {
                if (tensor != null) {
                    try { tensor.close(); } catch (Exception ignored) {}
                }
            }
        }
    }

    /**
     * Runs a one-time trial inference with dummy zero-filled inputs, matching the shapes/names this
     * session's graph declares. Used by {@link ModelLoader} at discovery time so a broken graph is marked
     * unavailable up front rather than failing on the first real chunk request (PROJECT.md §"Error Handling
     * & Fallback Chain" / docs/model-spec.md's "Load time" guidance).
     */
    public boolean trialInference(int maxPromptTokens) {
        if (session == null || env == null) {
            return true; // no graph loaded at all is handled by the FallbackTerrain path, not a failure.
        }
        try {
            ChunkOutput output = generateChunk(0, 0, 0L, new int[Math.max(1, maxPromptTokens)]);
            return output != null;
        } catch (Exception e) {
            McImagine.LOGGER.warn("ModelSession: trial inference failed, marking model unavailable: {}", e.getMessage());
            return false;
        }
    }

    public ChunkOutput inferChunk(ChunkInput input) {
        try {
            int[] tokens = input.promptTokens() != null ? input.promptTokens() : new int[0];
            return generateChunk(input.chunkX(), input.chunkZ(), input.seed(), tokens);
        } catch (Exception e) {
            McImagine.LOGGER.warn("ModelSession: inference failed for chunk ({}, {}), using fallback terrain: {}",
                    input.chunkX(), input.chunkZ(), e.getMessage());
            return FallbackTerrain.generateFallbackChunkOutput(input.chunkX(), input.chunkZ());
        }
    }

    private static int[] padOrTruncate(int[] arr, int expectedLength, int[] fallback) {
        if (arr.length == expectedLength) {
            return arr;
        }
        if (arr.length == 0) {
            return fallback;
        }
        int[] out = new int[expectedLength];
        System.arraycopy(arr, 0, out, 0, Math.min(arr.length, expectedLength));
        return out;
    }

    /**
     * Reads a tensor's raw contents into a flat {@code int[]} via its typed NIO buffer (bug #2 fix) rather
     * than materializing a nested Java array and pattern-matching on its shape — this is what previously
     * silently returned {@code new int[0]} for a nested {@code int[16][384][16]} block_volume output.
     */
    private int[] extractTensorAsInts(OnnxTensor tensor, String tensorName) {
        try {
            TensorInfo info = tensor.getInfo();
            int numElements = (int) info.getNumElements();
            int[] out = new int[numElements];
            OnnxJavaType type = info.type;
            if (type == OnnxJavaType.INT32) {
                tensor.getIntBuffer().get(out);
            } else if (type == OnnxJavaType.INT64) {
                java.nio.LongBuffer buf = tensor.getLongBuffer();
                for (int i = 0; i < numElements; i++) out[i] = (int) buf.get(i);
            } else if (type == OnnxJavaType.FLOAT) {
                java.nio.FloatBuffer buf = tensor.getFloatBuffer();
                for (int i = 0; i < numElements; i++) out[i] = Math.round(buf.get(i));
            } else if (type == OnnxJavaType.DOUBLE) {
                java.nio.DoubleBuffer buf = tensor.getDoubleBuffer();
                for (int i = 0; i < numElements; i++) out[i] = (int) Math.round(buf.get(i));
            } else {
                McImagine.LOGGER.warn("ModelSession: output '{}' has unsupported dtype {}; ignoring", tensorName, type);
                return new int[0];
            }
            return out;
        } catch (Exception e) {
            McImagine.LOGGER.warn("ModelSession: failed to extract output '{}': {}", tensorName, e.getMessage());
            return new int[0];
        }
    }

    @Override
    public void close() throws Exception {
        if (!isClosed) {
            isClosed = true;
            if (session != null) {
                try { session.close(); } catch (Exception ignored) {}
            }
            // Bug #3 fix: OrtEnvironment.getEnvironment() is a process-wide singleton shared by every
            // ModelSession in the JVM - only the session owned by this instance is ever closed here.
        }
    }
}
