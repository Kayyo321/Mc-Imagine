package com.mcimagine.model;

import com.mcimagine.api.ModelInfo;
import com.mcimagine.model.ModelLoader.FormatContract;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Pins docs/phase4-plan.md §6's format-version handling: 0.6.0 loads, and 0.5.0 keeps loading
 * unchanged alongside it.
 *
 * <p>The compatibility half is the half that matters. 0.6.0 changes what {@code heightmap} and
 * {@code block_volume} <em>mean</em>, not their shapes or names, and the mod reads {@code block_volume}
 * literally under both contracts - so a 0.5.0 model is not merely still loadable but still correct, and
 * a version check that rejected or downgraded it would be a regression invented by the version check
 * itself. These tests exist so the next version bump cannot quietly acquire one.
 *
 * <p>Each case is a synthetic {@code .mcim} built around the dummy model's real {@code model.onnx}, so
 * {@code parseModelInfo}'s load-time trial inference runs for real rather than being skipped.
 */
public class ModelLoaderFormatVersionTest {

    private static Path findDummyMcim() {
        Path[] candidates = new Path[]{
                Paths.get("fabric/run/mcimagine/models"),
                Paths.get("../fabric/run/mcimagine/models"),
                Paths.get("mod/fabric/run/mcimagine/models"),
                Paths.get("../../mod/fabric/run/mcimagine/models")
        };
        for (Path candidate : candidates) {
            Path mcim = candidate.resolve("dummy-test-v1.mcim");
            if (Files.exists(mcim)) {
                return mcim.toAbsolutePath();
            }
        }
        return null;
    }

    /**
     * Writes a {@code .mcim} declaring {@code formatVersion}, reusing the dummy model's ONNX graph so the
     * archive is loadable end to end. Everything else in the manifest is held constant across cases, so a
     * difference between two of these tests can only be the format version.
     */
    private static Path writeMcim(Path dir, String id, String formatVersion) throws IOException {
        Path dummy = findDummyMcim();
        assertNotNull(dummy, "dummy-test-v1.mcim should be found in the workspace");

        byte[] onnxBytes;
        try (ZipFile source = new ZipFile(dummy.toFile());
             InputStream is = source.getInputStream(source.getEntry("model.onnx"))) {
            onnxBytes = is.readAllBytes();
        }

        String manifest = "{\n"
                + "  \"format_version\": \"" + formatVersion + "\",\n"
                + "  \"id\": \"" + id + "\",\n"
                + "  \"name\": \"" + id + "\",\n"
                + "  \"version\": \"1.1.2\",\n"
                + "  \"author\": \"Mc-Imagine Team\",\n"
                + "  \"onnx_filename\": \"model.onnx\",\n"
                + "  \"capabilities\": {\n"
                + "    \"intensity\": \"low\",\n"
                + "    \"caves\": true,\n"
                + "    \"block_palette\": [\"minecraft:air\", \"minecraft:stone\"],\n"
                + "    \"max_prompt_tokens\": 128\n"
                + "  },\n"
                + "  \"io\": { \"chunk\": { \"input_names\": [\"prompt_tokens\"], \"output_names\": [\"heightmap\"] } }\n"
                + "}\n";

        Path mcim = dir.resolve(id + ".mcim");
        try (OutputStream fos = Files.newOutputStream(mcim);
             ZipOutputStream zip = new ZipOutputStream(fos)) {
            zip.putNextEntry(new ZipEntry("manifest.json"));
            zip.write(manifest.getBytes(StandardCharsets.UTF_8));
            zip.closeEntry();
            zip.putNextEntry(new ZipEntry("model.onnx"));
            zip.write(onnxBytes);
            zip.closeEntry();
        }
        return mcim;
    }

    @Test
    public void formatVersion060IsAcceptedAndClassifiedVolumetric(@TempDir Path dir) throws IOException {
        Path mcim = writeMcim(dir, "volumetric-v060", "0.6.0");

        ModelInfo info = new ModelLoader(dir).parseModelInfo(mcim);

        assertNotNull(info);
        assertEquals("0.6.0", info.formatVersion());
        assertTrue(info.available(), "a 0.6.0 manifest must load and pass trial inference");
        assertEquals(FormatContract.VOLUMETRIC_0_6_0, ModelLoader.formatContract(info.formatVersion()));
    }

    @Test
    public void formatVersion050RemainsLoadableAndUnchanged(@TempDir Path dir) throws IOException {
        Path mcim = writeMcim(dir, "heightfield-v050", "0.5.0");

        ModelInfo info = new ModelLoader(dir).parseModelInfo(mcim);

        assertNotNull(info);
        assertEquals("0.5.0", info.formatVersion());
        assertTrue(info.available(), "0.6.0 must not make an existing 0.5.0 model unloadable");
        assertEquals(FormatContract.HEIGHTFIELD_0_5_0, ModelLoader.formatContract(info.formatVersion()));
        assertEquals(128, info.capabilities().maxPromptTokens(), "0.5.0 capability parsing is untouched");
        assertEquals(2, info.capabilities().blockPalette().size(), "0.5.0 capability parsing is untouched");
    }

    @Test
    public void theTwoContractsAreDistinguishable(@TempDir Path dir) throws IOException {
        // The point of the classification: the same archive under two version strings must report two
        // different contracts, because that is all an operator has to tell an old model from a new one.
        ModelInfo old = new ModelLoader(dir).parseModelInfo(writeMcim(dir, "same-graph-v050", "0.5.0"));
        ModelInfo current = new ModelLoader(dir).parseModelInfo(writeMcim(dir, "same-graph-v060", "0.6.0"));

        assertNotEquals(ModelLoader.formatContract(old.formatVersion()),
                ModelLoader.formatContract(current.formatVersion()));
    }

    @Test
    public void everyPublishedVersionClassifies() {
        assertEquals(FormatContract.HEIGHTFIELD_0_5_0, ModelLoader.formatContract("0.1.0"),
                "0.1.0 is this loader's own format_version default and must never be UNKNOWN");
        assertEquals(FormatContract.HEIGHTFIELD_0_5_0, ModelLoader.formatContract("0.4.0"));
        assertEquals(FormatContract.HEIGHTFIELD_0_5_0, ModelLoader.formatContract("0.5.0"));
        assertEquals(FormatContract.VOLUMETRIC_0_6_0, ModelLoader.formatContract("0.6.0"));
    }

    @Test
    public void unpublishedVersionsAreReportedNotGuessedAt() {
        // Exact matching, so an unpublished value is never sorted into a range it was not tested against.
        assertEquals(FormatContract.UNKNOWN, ModelLoader.formatContract("0.7.0"));
        assertEquals(FormatContract.UNKNOWN, ModelLoader.formatContract("0.6"));
        assertEquals(FormatContract.UNKNOWN, ModelLoader.formatContract("not a version"));
        assertEquals(FormatContract.UNKNOWN, ModelLoader.formatContract(null));
    }

    @Test
    public void anUnpublishedVersionStillLoads(@TempDir Path dir) throws IOException {
        // UNKNOWN is a diagnosis, not a rejection: every manifest field the loader reads is defaulted,
        // so refusing the archive would break a model that generates perfectly well.
        ModelInfo info = new ModelLoader(dir).parseModelInfo(writeMcim(dir, "future-v999", "9.9.9"));

        assertNotNull(info);
        assertEquals("9.9.9", info.formatVersion());
        assertTrue(info.available(), "an unrecognized format_version must warn, not disable the model");
    }
}
