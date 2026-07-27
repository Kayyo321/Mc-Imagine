package com.mcimagine.model;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.mcimagine.api.ModelInfo;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Bridge between the mod and ONNX Runtime. Wraps an ONNX Runtime inference session.
 */
public class ModelSession implements AutoCloseable {

    private OrtEnvironment env;
    private OrtSession session;
    private boolean isClosed = false;

    public ModelSession() {
        // Default constructor
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

    public ChunkOutput generateChunk(int chunkX, int chunkZ, long seed, int[] promptTokens) throws Exception {
        if (session == null || env == null) {
            return generateFallbackChunkOutput(chunkX, chunkZ);
        }

        int[] tokens = new int[128];
        if (promptTokens != null && promptTokens.length > 0) {
            System.arraycopy(promptTokens, 0, tokens, 0, Math.min(promptTokens.length, 128));
        }

        int[][] tokens2D = new int[][]{ tokens };
        int[] cx = new int[]{ chunkX };
        int[] cz = new int[]{ chunkZ };
        long[] s = new long[]{ seed };

        Map<String, OnnxTensor> inputs = new HashMap<>();
        Set<String> sessionInputNames = session.getInputNames();

        OnnxTensor promptTensor = null;
        OnnxTensor chunkXTensor = null;
        OnnxTensor chunkZTensor = null;
        OnnxTensor seedTensor = null;

        try {
            if (sessionInputNames.contains("prompt_tokens")) {
                promptTensor = OnnxTensor.createTensor(env, tokens2D);
                inputs.put("prompt_tokens", promptTensor);
            }
            if (sessionInputNames.contains("chunk_x")) {
                chunkXTensor = OnnxTensor.createTensor(env, cx);
                inputs.put("chunk_x", chunkXTensor);
            }
            if (sessionInputNames.contains("chunk_z")) {
                chunkZTensor = OnnxTensor.createTensor(env, cz);
                inputs.put("chunk_z", chunkZTensor);
            }
            if (sessionInputNames.contains("seed")) {
                seedTensor = OnnxTensor.createTensor(env, s);
                inputs.put("seed", seedTensor);
            }

            try (OrtSession.Result results = session.run(inputs)) {
                int[] heightmap = new int[256];
                int[] blockVolume = new int[0];
                int[] biomeGrid = new int[256];
                float[][] structureMarkers = new float[0][0];

                boolean foundHeightmap = false;

                for (Map.Entry<String, OnnxValue> entry : results) {
                    String name = entry.getKey();
                    OnnxValue value = entry.getValue();

                    if (value instanceof OnnxTensor outputTensor) {
                        Object rawVal = outputTensor.getValue();
                        if ("heightmap".equalsIgnoreCase(name) || !foundHeightmap) {
                            heightmap = extractHeightmap(rawVal);
                            foundHeightmap = true;
                        } else if ("block_volume".equalsIgnoreCase(name) || "blocks".equalsIgnoreCase(name)) {
                            blockVolume = extractIntArray(rawVal);
                        } else if ("biomes".equalsIgnoreCase(name) || "biome_grid".equalsIgnoreCase(name)) {
                            biomeGrid = extractIntArray(rawVal);
                        }
                    }
                }

                if (blockVolume.length == 0 && heightmap.length > 0) {
                    blockVolume = buildBlockVolumeFromHeightmap(heightmap);
                }

                return new ChunkOutput(heightmap, blockVolume, biomeGrid, structureMarkers);
            }
        } finally {
            for (OnnxTensor tensor : inputs.values()) {
                if (tensor != null) {
                    try { tensor.close(); } catch (Exception ignored) {}
                }
            }
        }
    }

    public ChunkOutput inferChunk(ChunkInput input) {
        try {
            int[] tokens = input.promptTokens() != null ? input.promptTokens() : new int[0];
            return generateChunk(input.chunkX(), input.chunkZ(), input.seed(), tokens);
        } catch (Exception e) {
            e.printStackTrace();
            return generateFallbackChunkOutput(input.chunkX(), input.chunkZ());
        }
    }

    private int[] extractHeightmap(Object rawVal) {
        int[] hmap = new int[256];
        if (rawVal instanceof float[][] matrix2d) {
            for (int r = 0; r < matrix2d.length && r < 16; r++) {
                for (int c = 0; c < matrix2d[r].length && c < 16; c++) {
                    hmap[r * 16 + c] = Math.round(matrix2d[r][c]);
                }
            }
        } else if (rawVal instanceof float[][][] tensor3d) {
            float[][] matrix2d = tensor3d[0];
            for (int r = 0; r < matrix2d.length && r < 16; r++) {
                for (int c = 0; c < matrix2d[r].length && c < 16; c++) {
                    hmap[r * 16 + c] = Math.round(matrix2d[r][c]);
                }
            }
        } else if (rawVal instanceof float[] flat) {
            for (int i = 0; i < flat.length && i < 256; i++) {
                hmap[i] = Math.round(flat[i]);
            }
        } else if (rawVal instanceof double[][] matrix2d) {
            for (int r = 0; r < matrix2d.length && r < 16; r++) {
                for (int c = 0; c < matrix2d[r].length && c < 16; c++) {
                    hmap[r * 16 + c] = (int) Math.round(matrix2d[r][c]);
                }
            }
        } else if (rawVal instanceof double[] flat) {
            for (int i = 0; i < flat.length && i < 256; i++) {
                hmap[i] = (int) Math.round(flat[i]);
            }
        } else if (rawVal instanceof int[][] matrix2d) {
            for (int r = 0; r < matrix2d.length && r < 16; r++) {
                for (int c = 0; c < matrix2d[r].length && c < 16; c++) {
                    hmap[r * 16 + c] = matrix2d[r][c];
                }
            }
        } else if (rawVal instanceof int[] flat) {
            for (int i = 0; i < flat.length && i < 256; i++) {
                hmap[i] = flat[i];
            }
        }
        return hmap;
    }

    private int[] extractIntArray(Object rawVal) {
        if (rawVal instanceof int[] arr) {
            return arr;
        } else if (rawVal instanceof float[] arr) {
            int[] result = new int[arr.length];
            for (int i = 0; i < arr.length; i++) {
                result[i] = Math.round(arr[i]);
            }
            return result;
        }
        return new int[0];
    }

    private int[] buildBlockVolumeFromHeightmap(int[] heightmap) {
        int minY = -64;
        int maxY = 320;
        int heightSpan = maxY - minY; // 384
        int[] blockVolume = new int[16 * heightSpan * 16];
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                int h = heightmap[z * 16 + x];
                for (int y = minY; y <= h && y < maxY; y++) {
                    int yIdx = y - minY;
                    int index = (yIdx * 16 + z) * 16 + x;
                    if (index >= 0 && index < blockVolume.length) {
                        if (y < h - 1) blockVolume[index] = 1; // Stone
                        else if (y == h - 1) blockVolume[index] = 2; // Dirt
                        else blockVolume[index] = 3; // Grass Block
                    }
                }
            }
        }
        return blockVolume;
    }

    private ChunkOutput generateFallbackChunkOutput(int chunkX, int chunkZ) {
        int[] heightmap = new int[256];
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                int globalX = chunkX * 16 + x;
                int globalZ = chunkZ * 16 + z;
                int h = 60 + (int) (Math.sin(globalX / 10.0) * 5.0 + Math.cos(globalZ / 10.0) * 5.0);
                heightmap[z * 16 + x] = h;
            }
        }
        int[] blockVolume = buildBlockVolumeFromHeightmap(heightmap);
        return new ChunkOutput(heightmap, blockVolume, new int[256], new float[0][0]);
    }

    @Override
    public void close() throws Exception {
        if (!isClosed) {
            isClosed = true;
            if (session != null) {
                try { session.close(); } catch (Exception ignored) {}
            }
            if (env != null) {
                try { env.close(); } catch (Exception ignored) {}
            }
        }
    }
}