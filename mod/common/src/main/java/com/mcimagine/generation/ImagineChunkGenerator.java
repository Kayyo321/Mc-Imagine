package com.mcimagine.generation;

import com.mcimagine.api.ModelInfo;
import com.mcimagine.model.ChunkOutput;
import com.mcimagine.model.ModelLoader;
import com.mcimagine.model.ModelSession;
import com.mojang.serialization.Codec;
import com.mojang.serialization.codecs.RecordCodecBuilder;
import net.minecraft.core.BlockPos;
import net.minecraft.server.level.WorldGenRegion;
import net.minecraft.world.level.ChunkPos;
import net.minecraft.world.level.LevelHeightAccessor;
import net.minecraft.world.level.NoiseColumn;
import net.minecraft.world.level.StructureManager;
import net.minecraft.world.level.biome.BiomeManager;
import net.minecraft.world.level.biome.BiomeSource;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.chunk.ChunkAccess;
import net.minecraft.world.level.chunk.ChunkGenerator;
import net.minecraft.world.level.levelgen.GenerationStep;
import net.minecraft.world.level.levelgen.Heightmap;
import net.minecraft.world.level.levelgen.RandomState;
import net.minecraft.world.level.levelgen.blending.Blender;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executor;

/**
 * Replaces vanilla chunk generation with AI-powered generation driven by ONNX model tensors.
 */
public class ImagineChunkGenerator extends ChunkGenerator implements AutoCloseable {

    public static final Codec<ImagineChunkGenerator> CODEC = RecordCodecBuilder.create(instance ->
            instance.group(
                    BiomeSource.CODEC.fieldOf("biome_source").forGetter(ImagineChunkGenerator::getBiomeSource)
            ).apply(instance, instance.stable(ImagineChunkGenerator::new))
    );

    private static final Path[] DEFAULT_MODEL_DIRS = new Path[]{
            Paths.get("fabric/run/mcimagine/models"),
            Paths.get("../fabric/run/mcimagine/models"),
            Paths.get("mod/fabric/run/mcimagine/models"),
            Paths.get("../../mod/fabric/run/mcimagine/models"),
            Paths.get("../../../mod/fabric/run/mcimagine/models"),
            Paths.get("models/mcimagine"),
            Paths.get("mcimagine/models"),
            Paths.get("../models/mcimagine"),
            Paths.get("../../models/mcimagine")
    };

    private ModelSession modelSession;
    private ModelLoader modelLoader;
    private ModelInfo modelInfo;
    private long seed = 0L;
    private int[] promptTokens = new int[128];
    private final Map<Long, ChunkOutput> chunkCache = new ConcurrentHashMap<>();

    public ImagineChunkGenerator(BiomeSource biomeSource) {
        this(biomeSource, (Path) null);
    }

    public ImagineChunkGenerator(BiomeSource biomeSource, Path customModelPath) {
        super(biomeSource);
        initModel(customModelPath);
    }

    public ImagineChunkGenerator(BiomeSource biomeSource, ModelSession modelSession) {
        super(biomeSource);
        this.modelSession = modelSession;
    }

    public ImagineChunkGenerator(BiomeSource biomeSource, ModelSession modelSession, long seed, int[] promptTokens) {
        super(biomeSource);
        this.modelSession = modelSession;
        this.seed = seed;
        if (promptTokens != null) {
            this.promptTokens = promptTokens;
        }
    }

    private void initModel(Path customModelPath) {
        if (customModelPath != null && Files.exists(customModelPath)) {
            if (Files.isDirectory(customModelPath)) {
                if (loadFromDirectory(customModelPath)) return;
            } else if (customModelPath.toString().endsWith(".mcim")) {
                if (loadFromMcimFile(customModelPath)) return;
            }
        }

        String sysPropPath = System.getProperty("mcimagine.model.path");
        if (sysPropPath != null) {
            Path path = Paths.get(sysPropPath);
            if (Files.exists(path)) {
                if (Files.isDirectory(path)) {
                    if (loadFromDirectory(path)) return;
                } else if (loadFromMcimFile(path)) return;
            }
        }

        for (Path dir : DEFAULT_MODEL_DIRS) {
            if (Files.exists(dir) && Files.isDirectory(dir)) {
                if (loadFromDirectory(dir)) {
                    return;
                }
            }
        }

        // Fallback session
        this.modelSession = new ModelSession();
    }

    private boolean loadFromDirectory(Path dir) {
        try {
            ModelLoader loader = new ModelLoader(dir);
            List<ModelInfo> models = loader.discoverModels();
            if (!models.isEmpty()) {
                ModelInfo selected = models.stream()
                        .filter(m -> "dummy-test-v1".equals(m.id()) || (m.path() != null && m.path().contains("dummy-test-v1")))
                        .findFirst()
                        .orElse(models.get(0));
                byte[] bytes = loader.extractModelBytes(selected);
                this.modelLoader = loader;
                this.modelInfo = selected;
                this.modelSession = new ModelSession(bytes);
                return true;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return false;
    }

    private boolean loadFromMcimFile(Path mcimFile) {
        try {
            Path parent = mcimFile.getParent() != null ? mcimFile.getParent() : Paths.get(".");
            ModelLoader loader = new ModelLoader(parent);
            ModelInfo info = loader.parseModelInfo(mcimFile);
            if (info != null) {
                byte[] bytes = loader.extractModelBytes(info);
                this.modelLoader = loader;
                this.modelInfo = info;
                this.modelSession = new ModelSession(bytes);
                return true;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return false;
    }

    public ChunkOutput generateOrGetChunkOutput(int chunkX, int chunkZ) {
        long key = ChunkPos.asLong(chunkX, chunkZ);
        return chunkCache.computeIfAbsent(key, k -> {
            try {
                if (modelSession != null) {
                    return modelSession.generateChunk(chunkX, chunkZ, seed, promptTokens);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
            return generateFallbackChunkOutput(chunkX, chunkZ);
        });
    }

    private ChunkOutput generateFallbackChunkOutput(int chunkX, int chunkZ) {
        int[] heightmap = new int[256];
        int defaultHeight = 60;
        for (int i = 0; i < 256; i++) {
            heightmap[i] = defaultHeight;
        }
        int minY = -64;
        int heightSpan = 384;
        int[] blockVolume = new int[16 * heightSpan * 16];
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                for (int y = minY; y <= defaultHeight; y++) {
                    int yIdx = y - minY;
                    int volIndex = (yIdx * 16 + z) * 16 + x;
                    if (volIndex >= 0 && volIndex < blockVolume.length) {
                        if (y < defaultHeight - 1) blockVolume[volIndex] = 1;
                        else if (y == defaultHeight - 1) blockVolume[volIndex] = 2;
                        else blockVolume[volIndex] = 3;
                    }
                }
            }
        }
        return new ChunkOutput(heightmap, blockVolume, new int[256], new float[0][0]);
    }

    @Override
    protected Codec<? extends ChunkGenerator> codec() {
        return CODEC;
    }

    @Override
    public void applyCarvers(WorldGenRegion region, long seed, RandomState randomState, BiomeManager biomeManager, StructureManager structureManager, ChunkAccess chunk, GenerationStep.Carving step) {
    }

    @Override
    public void buildSurface(WorldGenRegion region, StructureManager structureManager, RandomState randomState, ChunkAccess chunk) {
    }

    @Override
    public void spawnOriginalMobs(WorldGenRegion region) {
    }

    @Override
    public int getGenDepth() {
        return 384;
    }

    @Override
    public CompletableFuture<ChunkAccess> fillFromNoise(Executor executor, Blender blender, RandomState randomState, StructureManager structureManager, ChunkAccess chunk) {
        return CompletableFuture.supplyAsync(() -> {
            int chunkX = chunk.getPos().x;
            int chunkZ = chunk.getPos().z;

            ChunkOutput output = generateOrGetChunkOutput(chunkX, chunkZ);
            int[] heightmap = output.heightmap();
            int[] blockVolume = output.blockVolume();

            int minBuildHeight = chunk.getMinBuildHeight();
            int maxBuildHeight = chunk.getMaxBuildHeight();

            for (int x = 0; x < 16; x++) {
                for (int z = 0; z < 16; z++) {
                    int height = (heightmap != null && heightmap.length >= 256)
                            ? heightmap[z * 16 + x]
                            : 60;

                    for (int y = minBuildHeight; y <= height && y < maxBuildHeight; y++) {
                        BlockState state;
                        int yIdx = y - minBuildHeight;
                        int volIndex = (yIdx * 16 + z) * 16 + x;

                        if (blockVolume != null && volIndex >= 0 && volIndex < blockVolume.length && blockVolume[volIndex] != 0) {
                            state = getBlockStateFromId(blockVolume[volIndex]);
                        } else {
                            if (y < height - 1) {
                                state = Blocks.STONE.defaultBlockState();
                            } else if (y == height - 1) {
                                state = Blocks.DIRT.defaultBlockState();
                            } else {
                                state = Blocks.GRASS_BLOCK.defaultBlockState();
                            }
                        }
                        chunk.setBlockState(new BlockPos(x, y, z), state, false);
                    }
                }
            }
            return chunk;
        }, executor);
    }

    @Override
    public int getSeaLevel() {
        return 63;
    }

    @Override
    public int getMinY() {
        return -64;
    }

    @Override
    public int getBaseHeight(int x, int z, Heightmap.Types type, LevelHeightAccessor level, RandomState randomState) {
        int chunkX = Math.floorDiv(x, 16);
        int chunkZ = Math.floorDiv(z, 16);
        int localX = Math.floorMod(x, 16);
        int localZ = Math.floorMod(z, 16);

        ChunkOutput output = generateOrGetChunkOutput(chunkX, chunkZ);
        int[] heightmap = output.heightmap();
        if (heightmap != null && heightmap.length >= 256) {
            return heightmap[localZ * 16 + localX];
        }
        return 60;
    }

    @Override
    public NoiseColumn getBaseColumn(int x, int z, LevelHeightAccessor level, RandomState randomState) {
        int chunkX = Math.floorDiv(x, 16);
        int chunkZ = Math.floorDiv(z, 16);
        int localX = Math.floorMod(x, 16);
        int localZ = Math.floorMod(z, 16);

        ChunkOutput output = generateOrGetChunkOutput(chunkX, chunkZ);
        int height = (output.heightmap() != null && output.heightmap().length >= 256)
                ? output.heightmap()[localZ * 16 + localX]
                : 60;

        int minHeight = level.getMinBuildHeight();
        int maxHeight = level.getMaxBuildHeight();
        int columnSpan = Math.max(0, height - minHeight + 1);

        BlockState[] states = new BlockState[Math.min(columnSpan, maxHeight - minHeight)];
        int[] blockVolume = output.blockVolume();

        for (int i = 0; i < states.length; i++) {
            int y = minHeight + i;
            int volIndex = (i * 16 + localZ) * 16 + localX;
            if (blockVolume != null && volIndex >= 0 && volIndex < blockVolume.length && blockVolume[volIndex] != 0) {
                states[i] = getBlockStateFromId(blockVolume[volIndex]);
            } else {
                if (y < height - 1) states[i] = Blocks.STONE.defaultBlockState();
                else if (y == height - 1) states[i] = Blocks.DIRT.defaultBlockState();
                else if (y == height) states[i] = Blocks.GRASS_BLOCK.defaultBlockState();
                else states[i] = Blocks.AIR.defaultBlockState();
            }
        }
        return new NoiseColumn(minHeight, states);
    }

    @Override
    public void addDebugScreenInfo(List<String> list, RandomState randomState, BlockPos pos) {
        if (modelInfo != null) {
            list.add("Model: " + modelInfo.name() + " v" + modelInfo.version());
        } else {
            list.add("Model: Fallback Engine");
        }
    }

    private BlockState getBlockStateFromId(int id) {
        return switch (id) {
            case 1 -> Blocks.STONE.defaultBlockState();
            case 2 -> Blocks.DIRT.defaultBlockState();
            case 3 -> Blocks.GRASS_BLOCK.defaultBlockState();
            case 4 -> Blocks.OAK_LOG.defaultBlockState();
            case 5 -> Blocks.OAK_LEAVES.defaultBlockState();
            case 6 -> Blocks.WATER.defaultBlockState();
            case 7 -> Blocks.SAND.defaultBlockState();
            case 8 -> Blocks.BEDROCK.defaultBlockState();
            default -> Blocks.STONE.defaultBlockState();
        };
    }

    public ModelSession getModelSession() {
        return modelSession;
    }

    public ModelInfo getModelInfo() {
        return modelInfo;
    }

    public ModelLoader getModelLoader() {
        return modelLoader;
    }

    public boolean isModelLoaded() {
        return modelInfo != null && modelSession != null;
    }

    public void setPromptTokens(int[] promptTokens) {
        if (promptTokens != null) {
            this.promptTokens = promptTokens;
            clearCache();
        }
    }

    public void setSeed(long seed) {
        this.seed = seed;
        clearCache();
    }

    public void clearCache() {
        chunkCache.clear();
    }

    @Override
    public void close() throws Exception {
        if (modelSession != null) {
            modelSession.close();
        }
    }
}