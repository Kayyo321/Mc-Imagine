package com.mcimagine.model;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.mcimagine.McImagine;
import com.mcimagine.api.ModelInfo;
import com.mcimagine.api.ModelInfo.ModelCapabilities;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
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
                return new ModelInfo(defaultName, defaultName, "1.0.0", "Unknown", "A Minecraft Imagine model.",
                        absPath, "model.onnx", "0.1.0", ModelCapabilities.legacyDefaults(), false);
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
                JsonObject json;
                try (InputStream is = zipFile.getInputStream(manifestEntry);
                     InputStreamReader reader = new InputStreamReader(is, StandardCharsets.UTF_8)) {
                    json = JsonParser.parseReader(reader).getAsJsonObject();
                }

                String id = getString(json, "id", defaultName);
                String name = getString(json, "name", defaultName);
                String version = getString(json, "version", "1.0.0");
                String author = getString(json, "author", "Unknown");
                String description = getString(json, "description", "A Minecraft Imagine model.");
                String onnxFilename = getString(json, "onnx_filename", "model.onnx");
                String formatVersion = getString(json, "format_version", "0.1.0");

                if (json.has("model") && json.get("model").isJsonObject()) {
                    JsonObject modelObj = json.getAsJsonObject("model");
                    if (id.equals(defaultName)) id = getString(modelObj, "id", id);
                    if (name.equals(defaultName)) name = getString(modelObj, "name", name);
                    if ("1.0.0".equals(version)) version = getString(modelObj, "version", version);
                    if ("Unknown".equals(author)) author = getString(modelObj, "author", author);
                    if ("A Minecraft Imagine model.".equals(description)) description = getString(modelObj, "description", description);
                    if ("model.onnx".equals(onnxFilename)) onnxFilename = getString(modelObj, "onnx_filename", onnxFilename);
                }

                ModelCapabilities capabilities = parseCapabilities(json);
                List<String> declaredInputs = getChunkIoNames(json, "input_names");
                List<String> declaredOutputs = getChunkIoNames(json, "output_names");

                ModelInfo info = new ModelInfo(id, name, version, author, description, absPath, onnxFilename,
                        formatVersion, capabilities, true);

                // Logged after the trial inference, not before: a model that announces "0.6.0 -
                // volumetric" and is then marked unavailable two lines later reads as a working
                // volumetric model to anyone skimming, which is the confusion this line exists to prevent.
                boolean available = validateAndTrialInference(mcimPath, info, declaredInputs, declaredOutputs);
                logFormatContract(id, formatVersion, available);
                return info.withAvailable(available);
            }
        } catch (Exception e) {
            McImagine.LOGGER.warn("ModelLoader: failed to parse manifest for {}: {}", mcimPath, e.getMessage());
        }

        return new ModelInfo(defaultName, defaultName, "1.0.0", "Unknown", "A Minecraft Imagine model.",
                absPath, "model.onnx", "0.1.0", ModelCapabilities.legacyDefaults(), false);
    }

    /**
     * Which docs/model-spec.md output contract a manifest's {@code format_version} binds its model to.
     *
     * <p>0.6.0 is a purely semantic revision: the tensor names, shapes and dtypes are identical to
     * 0.5.0's, and what changes is only what the numbers in them promise. Under
     * {@link #HEIGHTFIELD_0_5_0}, {@code block_volume} is derivable from
     * {@code (heightmap, profile, water_level)} and {@code heightmap} is the quantity the volume was
     * expanded from. Under {@link #VOLUMETRIC_0_6_0}, {@code block_volume} is authored directly and
     * {@code heightmap} is a *report* of its topmost solid cell (docs/phase4-plan.md §1.4, §6).
     *
     * <p><b>The mod does not branch on this.</b> It reads {@code block_volume} literally either way,
     * which is exactly why a 0.5.0 model keeps working unchanged - a heightfield volume is a valid
     * volume. The classification exists so the distinction is visible in a log: a 0.5.0 model and a
     * 0.6.0 model produce visibly different worlds, and when a 0.6.0 world comes out looking flat,
     * this line plus {@link VolumeFallbackLog}'s are what separate "the operator loaded the old
     * model" from "the new model's volume was rejected".
     */
    public enum FormatContract {
        /** 0.1.0 through 0.5.0: heightmap-derived volume. */
        HEIGHTFIELD_0_5_0("heightfield: block_volume is derived from heightmap, no overhangs possible"),
        /** 0.6.0: model-authored volume, heightmap reports its topmost solid cell. */
        VOLUMETRIC_0_6_0("volumetric: block_volume is authored directly and may contain overhangs"),
        /** Not a published revision - a corrupt manifest, or one written by a newer toolchain. */
        UNKNOWN("unrecognized - the mod will read block_volume literally, as it does for every version");

        private final String description;

        FormatContract(String description) {
            this.description = description;
        }

        public String description() {
            return description;
        }
    }

    /**
     * The {@code format_version} values this loader recognizes, mapped to the contract each carries.
     *
     * <p>docs/model-spec.md's Versioning section documents 0.6.0, 0.5.0 and (as the revision 0.5.0 kept
     * compatible with) 0.4.0. The rest are here for archives that predate it: 0.1.0 is this loader's own
     * default and the dummy test fixture's declared version, and 0.2.0/0.3.0 appear in no document at
     * all - they are listed only so an archive from that era classifies as heightfield rather than as
     * {@link FormatContract#UNKNOWN}, since every version before 0.6.0 shares one output contract.
     *
     * <p>Exact-string matching, not semver ordering, and that is a deliberate choice rather than an
     * omission. The mod has never compared versions - {@code format_version} was parsed into
     * {@link ModelInfo} and otherwise unused - and the one place that does adapt to an older manifest
     * shape ({@link #getChunkIoNames}) does it by probing for the field it needs rather than by
     * checking a version number. An enumerated set keeps that property: it is only ever asked "which
     * of the published revisions is this", a question with no ordering in it, and an unpublished value
     * is reported as {@link FormatContract#UNKNOWN} rather than being silently sorted into a range it
     * was never tested against. If a future revision does need "0.6.0 or later" semantics, that is the
     * point to introduce a comparator - and the point at which one can be tested against a real case.
     */
    private static final Map<String, FormatContract> PUBLISHED_FORMAT_VERSIONS = Map.of(
            "0.1.0", FormatContract.HEIGHTFIELD_0_5_0,
            "0.2.0", FormatContract.HEIGHTFIELD_0_5_0,
            "0.3.0", FormatContract.HEIGHTFIELD_0_5_0,
            "0.4.0", FormatContract.HEIGHTFIELD_0_5_0,
            "0.5.0", FormatContract.HEIGHTFIELD_0_5_0,
            "0.6.0", FormatContract.VOLUMETRIC_0_6_0);

    /** Classifies a manifest's {@code format_version}. Never throws; an unusable value is {@code UNKNOWN}. */
    public static FormatContract formatContract(String formatVersion) {
        if (formatVersion == null) {
            return FormatContract.UNKNOWN;
        }
        return PUBLISHED_FORMAT_VERSIONS.getOrDefault(formatVersion, FormatContract.UNKNOWN);
    }

    /**
     * Records, once per manifest parse, which spec contract the model claims. An unrecognized version
     * is a warning and not a load failure: every field this loader reads is defaulted, so the archive
     * still loads and still generates - the operator just needs to know the mod has no idea what it
     * agreed to.
     */
    private void logFormatContract(String id, String formatVersion, boolean available) {
        FormatContract contract = formatContract(formatVersion);
        String availability = available ? "" : " - but it failed its load-time checks and is unavailable";
        if (contract == FormatContract.UNKNOWN) {
            McImagine.LOGGER.warn("ModelLoader: model '{}' declares format_version '{}', which is not a recognized " +
                    "docs/model-spec.md revision ({}); loading it anyway{}",
                    id, formatVersion, contract.description(), availability);
        } else {
            McImagine.LOGGER.info("ModelLoader: model '{}' declares format_version {} - {}{}",
                    id, formatVersion, contract.description(), availability);
        }
    }

    /**
     * Parses the (optional) nested {@code capabilities} manifest block, extending the existing defensive
     * {@link #getString} pattern to string arrays and nested objects. Every field defaults per
     * docs/model-spec.md's Versioning note so 0.4.0-and-earlier manifests (missing intensity,
     * structure_support, prompt_tags, biome_palette, requires_macro_field, detail_passes entirely) still load.
     */
    private ModelCapabilities parseCapabilities(JsonObject root) {
        ModelCapabilities defaults = ModelCapabilities.legacyDefaults();
        if (!root.has("capabilities") || !root.get("capabilities").isJsonObject()) {
            return defaults;
        }
        JsonObject caps = root.getAsJsonObject("capabilities");

        String intensity = getString(caps, "intensity", defaults.intensity());
        String structureSupport = getString(caps, "structure_support", defaults.structureSupport());
        List<String> promptTags = getStringArray(caps, "prompt_tags", defaults.promptTags());
        List<String> blockPalette = getStringArray(caps, "block_palette", defaults.blockPalette());
        List<String> biomePalette = getStringArray(caps, "biome_palette", defaults.biomePalette());
        int maxPromptTokens = getInt(caps, "max_prompt_tokens", defaults.maxPromptTokens());
        boolean requiresMacroField = getBoolean(caps, "requires_macro_field", defaults.requiresMacroField());
        int detailPasses = getInt(caps, "detail_passes", defaults.detailPasses());

        return new ModelCapabilities(intensity, structureSupport, promptTags, blockPalette, biomePalette,
                maxPromptTokens, requiresMacroField, detailPasses);
    }

    /**
     * Reads {@code io.chunk.<key>} (0.5.0 nested shape); falls back to flat {@code io.<key>} for
     * pre-0.5.0/legacy manifests (e.g. the dummy test model, whose {@code io} block has no {@code chunk}
     * sub-object at all).
     */
    private List<String> getChunkIoNames(JsonObject root, String key) {
        if (!root.has("io") || !root.get("io").isJsonObject()) {
            return List.of();
        }
        JsonObject io = root.getAsJsonObject("io");
        if (io.has("chunk") && io.get("chunk").isJsonObject()) {
            return getStringArray(io.getAsJsonObject("chunk"), key, List.of());
        }
        return getStringArray(io, key, List.of());
    }

    /**
     * Validates the declared {@code io.chunk} input/output names against the actual {@code OrtSession}
     * names, and runs a one-time trial inference with dummy zero-filled inputs (PROJECT.md §"Error Handling
     * & Fallback Chain" / docs/model-spec.md "Load time"). A mismatch or thrown exception marks the model
     * unavailable in the UI-facing list rather than deferring failure to the first real chunk request.
     */
    private boolean validateAndTrialInference(Path mcimPath, ModelInfo info,
                                               List<String> declaredInputs, List<String> declaredOutputs) {
        byte[] modelBytes;
        try {
            modelBytes = extractModelBytes(mcimPath, info.onnxFilename());
        } catch (Exception e) {
            McImagine.LOGGER.warn("ModelLoader: model '{}' has no readable {} - marking unavailable: {}",
                    info.id(), info.onnxFilename(), e.getMessage());
            return false;
        }

        try (ModelSession trialSession = new ModelSession(modelBytes, formatContract(info.formatVersion()))) {
            if (!trialSession.isLoaded()) {
                McImagine.LOGGER.warn("ModelLoader: model '{}' produced an empty ONNX session - marking unavailable", info.id());
                return false;
            }

            Set<String> actualInputs = trialSession.getInputNames();
            Set<String> actualOutputs = trialSession.getOutputNames();
            if (!declaredInputs.isEmpty() && !actualInputs.containsAll(declaredInputs)) {
                McImagine.LOGGER.warn("ModelLoader: model '{}' declares io.chunk.input_names {} but the ONNX " +
                        "graph only exposes {} - proceeding, but this manifest may be stale", info.id(), declaredInputs, actualInputs);
            }
            if (!declaredOutputs.isEmpty() && !actualOutputs.containsAll(declaredOutputs)) {
                McImagine.LOGGER.warn("ModelLoader: model '{}' declares io.chunk.output_names {} but the ONNX " +
                        "graph only exposes {} - proceeding, but this manifest may be stale", info.id(), declaredOutputs, actualOutputs);
            }

            boolean ok = trialSession.trialInference(info.capabilities().maxPromptTokens());
            if (!ok) {
                McImagine.LOGGER.warn("ModelLoader: model '{}' failed its trial inference - marking unavailable", info.id());
            }
            return ok;
        } catch (Exception e) {
            McImagine.LOGGER.warn("ModelLoader: model '{}' failed to load its ONNX session - marking unavailable: {}",
                    info.id(), e.getMessage());
            return false;
        }
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

    /** Extracts {@code tokenizer.json} from a {@code .mcim} archive, or null if the archive has none. */
    public String extractTokenizerJson(Path mcimPath) {
        try (ZipFile zipFile = new ZipFile(mcimPath.toFile())) {
            ZipEntry entry = zipFile.getEntry("tokenizer.json");
            if (entry == null) {
                return null;
            }
            try (InputStream is = zipFile.getInputStream(entry)) {
                return new String(is.readAllBytes(), StandardCharsets.UTF_8);
            }
        } catch (Exception e) {
            return null;
        }
    }

    private String getString(JsonObject json, String memberName, String fallback) {
        if (json.has(memberName) && !json.get(memberName).isJsonNull()) {
            return json.get(memberName).getAsString();
        }
        return fallback;
    }

    private int getInt(JsonObject json, String memberName, int fallback) {
        if (json.has(memberName) && !json.get(memberName).isJsonNull()) {
            try {
                return json.get(memberName).getAsInt();
            } catch (Exception ignored) {}
        }
        return fallback;
    }

    private boolean getBoolean(JsonObject json, String memberName, boolean fallback) {
        if (json.has(memberName) && !json.get(memberName).isJsonNull()) {
            try {
                return json.get(memberName).getAsBoolean();
            } catch (Exception ignored) {}
        }
        return fallback;
    }

    private List<String> getStringArray(JsonObject json, String memberName, List<String> fallback) {
        if (json.has(memberName) && json.get(memberName).isJsonArray()) {
            JsonArray array = json.getAsJsonArray(memberName);
            List<String> result = new ArrayList<>();
            for (JsonElement el : array) {
                if (!el.isJsonNull()) {
                    result.add(el.getAsString());
                }
            }
            return result;
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
