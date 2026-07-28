package com.mcimagine.generation;

import com.mcimagine.McImagine;
import com.mcimagine.api.ModelInfo;
import com.mcimagine.model.BlockPalette;
import com.mcimagine.model.ChunkOutput;
import com.mcimagine.model.FallbackTerrain;
import com.mcimagine.model.ModelLoader;
import com.mcimagine.model.ModelRegistry;
import com.mcimagine.model.ModelSession;
import com.mcimagine.prompt.PromptTokenizer;
import com.mojang.serialization.Codec;
import com.mojang.serialization.codecs.RecordCodecBuilder;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Holder;
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
import net.minecraft.world.level.chunk.LevelChunkSection;
import net.minecraft.world.level.levelgen.GenerationStep;
import net.minecraft.world.level.levelgen.Heightmap;
import net.minecraft.world.level.levelgen.NoiseBasedChunkGenerator;
import net.minecraft.world.level.levelgen.NoiseGeneratorSettings;
import net.minecraft.world.level.levelgen.RandomState;
import net.minecraft.world.level.levelgen.blending.Blender;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;

/**
 * Replaces vanilla chunk generation with AI-powered generation driven by ONNX model tensors.
 *
 * <p><b>Deviation from docs/poc-plan.md §1a:</b> the plan calls for extending
 * {@link NoiseBasedChunkGenerator} directly so vanilla {@code applyCarvers}/{@code buildSurface}/
 * {@code spawnOriginalMobs} are inherited "for free." In the actual 1.20.1 Mojang-mapped sources verified
 * here, {@code NoiseBasedChunkGenerator} is declared {@code final} and cannot be subclassed at all (this
 * was only discoverable by attempting the build - it compiles as "cannot inherit from final
 * NoiseBasedChunkGenerator"). This class therefore extends {@link ChunkGenerator} directly (as the
 * pre-existing Phase 1 code already did) and achieves the identical practical effect via composition: a
 * private {@link #vanillaDelegate} {@code NoiseBasedChunkGenerator}, built from the same biome source and
 * settings, that {@link #applyCarvers}, {@link #buildSurface}, {@link #spawnOriginalMobs}, and the
 * gen-depth/sea-level/min-Y accessors all delegate to. {@code createStructures}/{@code applyBiomeDecoration}
 * still come for free from {@link ChunkGenerator} itself (they were never {@code NoiseBasedChunkGenerator}-
 * specific). Only {@link #fillFromNoise}, {@link #getBaseHeight}, {@link #getBaseColumn}, and
 * {@link #codec()} carry AI-authored logic, matching the plan's intent even though the literal inheritance
 * mechanism had to change.
 */
public class ImagineChunkGenerator extends ChunkGenerator implements AutoCloseable {

    public static final Codec<ImagineChunkGenerator> CODEC = RecordCodecBuilder.create(instance ->
            instance.group(
                    BiomeSource.CODEC.fieldOf("biome_source").forGetter(ImagineChunkGenerator::getBiomeSource),
                    NoiseGeneratorSettings.CODEC.fieldOf("settings").forGetter(ImagineChunkGenerator::generatorSettings),
                    Codec.STRING.optionalFieldOf("prompt", "").forGetter(ImagineChunkGenerator::getPrompt),
                    Codec.STRING.optionalFieldOf("model_id", "").forGetter(ImagineChunkGenerator::getModelId),
                    Codec.LONG.optionalFieldOf("mcimagine_seed", 0L).forGetter(ImagineChunkGenerator::getSeed)
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

    private final Holder<NoiseGeneratorSettings> settings;
    /** Composition stand-in for the (final, un-subclassable) NoiseBasedChunkGenerator - see class javadoc. */
    private final NoiseBasedChunkGenerator vanillaDelegate;

    private final String prompt;
    private final String modelId;
    private final long seed;
    private final int[] promptTokens;
    private final ChunkCache chunkCache = new ChunkCache();

    private ModelSession modelSession;
    private ModelInfo modelInfo;
    private ModelLoader modelLoader;
    private BlockPalette blockPalette = new BlockPalette(List.of());

    /** Plain construction with no prompt/model selected yet - resolves whatever default model it can find. */
    public ImagineChunkGenerator(BiomeSource biomeSource, Holder<NoiseGeneratorSettings> settings) {
        this(biomeSource, settings, "", "", 0L);
    }

    /**
     * The CODEC- and UI-driven constructor: carries the prompt string, selected model id, and world seed so
     * they persist into {@code level.dat} and survive a reload (docs/poc-plan.md Phase 2 gate).
     */
    public ImagineChunkGenerator(BiomeSource biomeSource, Holder<NoiseGeneratorSettings> settings,
                                 String prompt, String modelId, long seed) {
        super(biomeSource);
        this.settings = settings;
        this.vanillaDelegate = new NoiseBasedChunkGenerator(biomeSource, settings);
        this.prompt = prompt != null ? prompt : "";
        this.modelId = modelId != null ? modelId : "";
        this.seed = seed;
        initModel(null);
        this.promptTokens = tokenizePrompt();
        attachBiomeSourceIfPossible(biomeSource);
    }

    /** Test/debug convenience constructor: resolves a model directly from a directory or {@code .mcim} path. */
    public ImagineChunkGenerator(BiomeSource biomeSource, Holder<NoiseGeneratorSettings> settings, Path customModelPath) {
        super(biomeSource);
        this.settings = settings;
        this.vanillaDelegate = new NoiseBasedChunkGenerator(biomeSource, settings);
        this.prompt = "";
        this.modelId = "";
        this.seed = 0L;
        initModel(customModelPath);
        this.promptTokens = tokenizePrompt();
        attachBiomeSourceIfPossible(biomeSource);
    }

    /** Test convenience constructor for an explicit, already-constructed {@link ModelSession}. */
    public ImagineChunkGenerator(BiomeSource biomeSource, Holder<NoiseGeneratorSettings> settings, ModelSession modelSession) {
        super(biomeSource);
        this.settings = settings;
        this.vanillaDelegate = new NoiseBasedChunkGenerator(biomeSource, settings);
        this.prompt = "";
        this.modelId = "";
        this.seed = 0L;
        this.modelSession = modelSession;
        this.promptTokens = tokenizePrompt();
        attachBiomeSourceIfPossible(biomeSource);
    }

    private void attachBiomeSourceIfPossible(BiomeSource biomeSource) {
        if (biomeSource instanceof ImagineBiomeSource imagineBiomeSource) {
            imagineBiomeSource.attachGenerator(this);
        }
    }

    private void initModel(Path customModelPath) {
        Path modelsDir = resolveModelsDirectory(customModelPath);

        if (modelsDir != null) {
            if (!modelId.isBlank()) {
                var sessionOpt = ModelRegistry.getOrLoad(modelId, modelsDir);
                if (sessionOpt.isPresent()) {
                    this.modelSession = sessionOpt.get();
                    this.modelInfo = ModelRegistry.getInfo(modelId).orElse(null);
                    this.blockPalette = new BlockPalette(modelInfo != null ? modelInfo.capabilities().blockPalette() : List.of());
                    return;
                }
                McImagine.LOGGER.warn("ImagineChunkGenerator: requested model '{}' not found/available in {}; " +
                        "falling back to the default model selection", modelId, modelsDir);
            }
            if (loadFromDirectory(modelsDir)) {
                return;
            }
        } else if (customModelPath != null && customModelPath.toString().endsWith(".mcim") && Files.exists(customModelPath)) {
            if (loadFromMcimFile(customModelPath)) {
                return;
            }
        }

        // Fallback session - FallbackTerrain sine-wave terrain, no ONNX graph loaded.
        this.modelSession = new ModelSession();
    }

    private Path resolveModelsDirectory(Path customModelPath) {
        if (customModelPath != null && Files.exists(customModelPath) && Files.isDirectory(customModelPath)) {
            return customModelPath;
        }
        String sysPropPath = System.getProperty("mcimagine.model.path");
        if (sysPropPath != null) {
            Path path = Paths.get(sysPropPath);
            if (Files.exists(path) && Files.isDirectory(path)) {
                return path;
            }
        }
        for (Path dir : DEFAULT_MODEL_DIRS) {
            if (Files.exists(dir) && Files.isDirectory(dir)) {
                return dir;
            }
        }
        return null;
    }

    private boolean loadFromDirectory(Path dir) {
        try {
            ModelLoader loader = new ModelLoader(dir);
            List<ModelInfo> models = loader.discoverModels();
            List<ModelInfo> availableModels = models.stream().filter(ModelInfo::available).toList();
            if (availableModels.isEmpty()) {
                return false;
            }
            ModelInfo selected = availableModels.stream()
                    .filter(m -> "dummy-test-v1".equals(m.id()) || (m.path() != null && m.path().contains("dummy-test-v1")))
                    .findFirst()
                    .orElse(availableModels.get(0));
            byte[] bytes = loader.extractModelBytes(selected);
            ModelSession session = new ModelSession(bytes);
            ModelRegistry.register(selected, session);
            this.modelLoader = loader;
            this.modelInfo = selected;
            this.modelSession = session;
            this.blockPalette = new BlockPalette(selected.capabilities().blockPalette());
            return true;
        } catch (Exception e) {
            McImagine.LOGGER.warn("ImagineChunkGenerator: failed to load a model from {}: {}", dir, e.getMessage());
        }
        return false;
    }

    private boolean loadFromMcimFile(Path mcimFile) {
        try {
            Path parent = mcimFile.getParent() != null ? mcimFile.getParent() : Paths.get(".");
            ModelLoader loader = new ModelLoader(parent);
            ModelInfo info = loader.parseModelInfo(mcimFile);
            if (info != null && info.available()) {
                byte[] bytes = loader.extractModelBytes(info);
                ModelSession session = new ModelSession(bytes);
                ModelRegistry.register(info, session);
                this.modelLoader = loader;
                this.modelInfo = info;
                this.modelSession = session;
                this.blockPalette = new BlockPalette(info.capabilities().blockPalette());
                return true;
            }
        } catch (Exception e) {
            McImagine.LOGGER.warn("ImagineChunkGenerator: failed to load {}: {}", mcimFile, e.getMessage());
        }
        return false;
    }

    private int[] tokenizePrompt() {
        int maxTokens = (modelInfo != null && modelInfo.capabilities().maxPromptTokens() > 0)
                ? modelInfo.capabilities().maxPromptTokens() : 128;

        PromptTokenizer tokenizer = loadTokenizerForCurrentModel();
        int[] tokens = tokenizer.tokenize(prompt, maxTokens);
        McImagine.LOGGER.info("ImagineChunkGenerator: tokenized prompt \"{}\" (model={}, vocabSize={}) -> {}",
                prompt, modelId.isBlank() ? "<default>" : modelId, tokenizer.vocabSize(), Arrays.toString(tokens));
        return tokens;
    }

    private PromptTokenizer loadTokenizerForCurrentModel() {
        if (modelInfo != null && modelInfo.path() != null) {
            try {
                Path mcimPath = Path.of(modelInfo.path());
                ModelLoader loader = new ModelLoader(mcimPath.getParent());
                String tokenizerJson = loader.extractTokenizerJson(mcimPath);
                if (tokenizerJson != null) {
                    return PromptTokenizer.fromTokenizerJson(tokenizerJson);
                }
            } catch (Exception ignored) {
                // Fall through to an empty-vocab tokenizer below.
            }
        }
        return new PromptTokenizer(Map.of());
    }

    public ChunkOutput generateOrGetChunkOutput(int chunkX, int chunkZ) {
        long key = ChunkPos.asLong(chunkX, chunkZ);
        return chunkCache.computeIfAbsent(key, k -> {
            try {
                if (modelSession != null) {
                    return modelSession.generateChunk(chunkX, chunkZ, seed, promptTokens);
                }
            } catch (Exception e) {
                McImagine.LOGGER.warn("ImagineChunkGenerator: inference failed for chunk ({}, {}): {}", chunkX, chunkZ, e.getMessage());
            }
            return FallbackTerrain.generateFallbackChunkOutput(chunkX, chunkZ);
        });
    }

    @Override
    protected Codec<? extends ChunkGenerator> codec() {
        return CODEC;
    }

    public Holder<NoiseGeneratorSettings> generatorSettings() {
        return settings;
    }

    // --- Vanilla generation steps, delegated to vanillaDelegate (see class javadoc) ---
    // This is what gives the §3 "none" row for free: vanilla caves/ravines carve the AI stone volume,
    // SurfaceSystem dresses whatever blocks the AI wrote (biome-correct snow/sand/podzol), and original
    // mobs spawn normally - all running unmodified against AI-authored terrain.

    @Override
    public void applyCarvers(WorldGenRegion region, long carverSeed, RandomState randomState, BiomeManager biomeManager,
                              StructureManager structureManager, ChunkAccess chunk, GenerationStep.Carving step) {
        vanillaDelegate.applyCarvers(region, carverSeed, randomState, biomeManager, structureManager, chunk, step);
    }

    @Override
    public void buildSurface(WorldGenRegion region, StructureManager structureManager, RandomState randomState, ChunkAccess chunk) {
        vanillaDelegate.buildSurface(region, structureManager, randomState, chunk);
    }

    @Override
    public void spawnOriginalMobs(WorldGenRegion region) {
        vanillaDelegate.spawnOriginalMobs(region);
    }

    @Override
    public int getGenDepth() {
        return vanillaDelegate.getGenDepth();
    }

    @Override
    public int getSeaLevel() {
        return vanillaDelegate.getSeaLevel();
    }

    @Override
    public int getMinY() {
        return vanillaDelegate.getMinY();
    }

    @Override
    public CompletableFuture<ChunkAccess> fillFromNoise(Executor executor, Blender blender, RandomState randomState,
                                                         StructureManager structureManager, ChunkAccess chunk) {
        return CompletableFuture.supplyAsync(() -> {
            int chunkX = chunk.getPos().x;
            int chunkZ = chunk.getPos().z;

            ChunkOutput output = generateOrGetChunkOutput(chunkX, chunkZ);
            int[] heightmap = validateHeightmap(output.heightmap(), chunkX, chunkZ);
            ValidatedVolume volume = validateBlockVolume(output, heightmap, chunkX, chunkZ);

            int minBuildHeight = chunk.getMinBuildHeight();
            int maxBuildHeight = chunk.getMaxBuildHeight();
            int seaLevel = getSeaLevel();

            Heightmap worldSurface = chunk.getOrCreateHeightmapUnprimed(Heightmap.Types.WORLD_SURFACE_WG);
            Heightmap oceanFloor = chunk.getOrCreateHeightmapUnprimed(Heightmap.Types.OCEAN_FLOOR_WG);

            LevelChunkSection[] sections = chunk.getSections();
            for (LevelChunkSection section : sections) {
                section.acquire();
            }
            try {
                for (int x = 0; x < 16; x++) {
                    for (int z = 0; z < 16; z++) {
                        for (int y = minBuildHeight; y < maxBuildHeight; y++) {
                            BlockState state = resolveBlockState(volume.blockVolume(), x, y, z, seaLevel,
                                    volume.fromFallback());

                            int sectionIndex = chunk.getSectionIndex(y);
                            if (sectionIndex < 0 || sectionIndex >= sections.length) {
                                continue;
                            }
                            int localY = y & 15;
                            sections[sectionIndex].setBlockState(x, localY, z, state, false);

                            // Bug #4 fix: prime WORLD_SURFACE_WG/OCEAN_FLOOR_WG as blocks are written, exactly
                            // as vanilla's own generation does - without this, vanilla surface dressing,
                            // decoration, and structure placement land in the wrong place or not at all.
                            worldSurface.update(x, y, z, state);
                            oceanFloor.update(x, y, z, state);
                        }
                    }
                }
            } finally {
                for (LevelChunkSection section : sections) {
                    section.release();
                }
            }
            return chunk;
        }, executor);
    }

    /**
     * Resolves the block at a single column position, trusting the (validated) block_volume literally
     * except for one targeted fallback: on a {@link FallbackTerrain}-derived volume, a declared-air cell at
     * or below sea level fills with water, matching vanilla's always-full-to-sea-level ocean convention.
     * That synthesis only ever fills up to the surface and leaves everything above as air, so without this
     * it would produce dry ocean basins.
     *
     * <p>A spec-conformant trained model already encodes water in its own block_volume via its export-time
     * Where-based expansion over its own predicted per-column water level (docs/model-spec.md), so its air
     * below sea level is authoritative and must be taken literally - applying the override there floods
     * every valley, cave mouth and canyon floor the model deliberately left dry (docs/phase2-plan.md §2
     * Phase 1.5).
     */
    private BlockState resolveBlockState(int[] blockVolume, int localX, int y, int localZ, int seaLevel,
                                         boolean volumeFromFallback) {
        int yIdx = y - FallbackTerrain.MIN_Y;
        int index = FallbackTerrain.blockVolumeIndex(localX, yIdx, localZ);
        int id = (blockVolume != null && index >= 0 && index < blockVolume.length) ? blockVolume[index] : 0;
        if (volumeFromFallback && id == 0 && y <= seaLevel) {
            return Blocks.WATER.defaultBlockState();
        }
        return blockPalette.get(id);
    }

    private int[] validateHeightmap(int[] raw, int chunkX, int chunkZ) {
        if (raw == null || raw.length != 256) {
            McImagine.LOGGER.warn("ImagineChunkGenerator: chunk ({}, {}) heightmap has {} elements (expected 256); " +
                    "using fallback terrain for this chunk", chunkX, chunkZ, raw == null ? -1 : raw.length);
            return FallbackTerrain.generateHeightmap(chunkX, chunkZ);
        }
        int[] clamped = new int[256];
        for (int i = 0; i < 256; i++) {
            clamped[i] = clampY(raw[i]);
        }
        return clamped;
    }

    /**
     * A validated block_volume together with where it came from. The provenance has to travel with the
     * array because {@link #resolveBlockState} applies the sea-level water fill only to
     * {@link FallbackTerrain}-synthesized volumes - see that method's javadoc.
     */
    private record ValidatedVolume(int[] blockVolume, boolean fromFallback) {}

    private ValidatedVolume validateBlockVolume(ChunkOutput output, int[] heightmap, int chunkX, int chunkZ) {
        int[] raw = output.blockVolume();
        int expectedLength = 16 * FallbackTerrain.HEIGHT_SPAN * 16;
        if (raw == null || raw.length != expectedLength) {
            if (raw != null && raw.length != 0) {
                McImagine.LOGGER.warn("ImagineChunkGenerator: chunk ({}, {}) block_volume has {} elements " +
                        "(expected {}); using heightmap-derived fallback volume for this chunk",
                        chunkX, chunkZ, raw.length, expectedLength);
            }
            return new ValidatedVolume(FallbackTerrain.buildBlockVolumeFromHeightmap(heightmap), true);
        }
        return new ValidatedVolume(raw, output.blockVolumeFromFallback());
    }

    private int clampY(int y) {
        return Math.max(FallbackTerrain.MIN_Y, Math.min(y, FallbackTerrain.MAX_Y - 1));
    }

    @Override
    public int getBaseHeight(int x, int z, Heightmap.Types type, LevelHeightAccessor level, RandomState randomState) {
        int chunkX = Math.floorDiv(x, 16);
        int chunkZ = Math.floorDiv(z, 16);
        int localX = Math.floorMod(x, 16);
        int localZ = Math.floorMod(z, 16);

        ChunkOutput output = generateOrGetChunkOutput(chunkX, chunkZ);
        int[] heightmap = output.heightmap();
        if (heightmap != null && heightmap.length == 256) {
            return clampY(heightmap[localZ * 16 + localX]);
        }
        return clampY(60);
    }

    @Override
    public NoiseColumn getBaseColumn(int x, int z, LevelHeightAccessor level, RandomState randomState) {
        int chunkX = Math.floorDiv(x, 16);
        int chunkZ = Math.floorDiv(z, 16);
        int localX = Math.floorMod(x, 16);
        int localZ = Math.floorMod(z, 16);

        ChunkOutput output = generateOrGetChunkOutput(chunkX, chunkZ);
        int[] heightmap = validateHeightmap(output.heightmap(), chunkX, chunkZ);
        ValidatedVolume volume = validateBlockVolume(output, heightmap, chunkX, chunkZ);

        int minHeight = level.getMinBuildHeight();
        int maxHeight = level.getMaxBuildHeight();
        int seaLevel = getSeaLevel();

        BlockState[] states = new BlockState[Math.max(0, maxHeight - minHeight)];
        for (int i = 0; i < states.length; i++) {
            int y = minHeight + i;
            states[i] = resolveBlockState(volume.blockVolume(), localX, y, localZ, seaLevel, volume.fromFallback());
        }
        return new NoiseColumn(minHeight, states);
    }

    @Override
    public void addDebugScreenInfo(List<String> list, RandomState randomState, BlockPos pos) {
        vanillaDelegate.addDebugScreenInfo(list, randomState, pos);
        if (modelInfo != null) {
            list.add("Model: " + modelInfo.name() + " v" + modelInfo.version());
        } else {
            list.add("Model: Fallback Engine");
        }
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

    public String getPrompt() {
        return prompt;
    }

    public String getModelId() {
        return modelId;
    }

    public long getSeed() {
        return seed;
    }

    @Override
    public void close() throws Exception {
        // Sessions are owned by ModelRegistry (shared across generator instances) and closed on server
        // stop, not here - closing per-generator-instance would break every other generator/world sharing
        // the same loaded model.
    }
}
