package com.mcimagine.prompt;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The real end-to-end tokenizer correctness gate docs/poc-plan.md's Phase 2/6 call for: unlike
 * {@link PromptTokenizerTest} (which only proves the WordPiece algorithm is internally consistent
 * against a small hand-built vocab), this test loads the trained model's *actual*
 * {@code all-MiniLM-L6-v2} {@code tokenizer.json} and asserts the Java implementation reproduces
 * the ~54 (prompt, token_ids) pairs the Python training pipeline emitted from the real HuggingFace
 * tokenizer bit-for-bit. A mismatch here is invisible at runtime (no crash, just silently wrong
 * prompt embeddings) and would quietly destroy prompt fidelity for every real player prompt.
 */
public class PromptTokenizerGoldenVectorTest {

    private static Path findFirstExisting(String... relativePaths) {
        for (String rel : relativePaths) {
            Path p = Paths.get(rel).toAbsolutePath().normalize();
            if (Files.exists(p)) {
                return p;
            }
        }
        return null;
    }

    @Test
    public void javaTokenizerMatchesPythonGoldenVectorsAgainstRealMiniLMVocab() throws IOException {
        Path tokenizerJsonPath = findFirstExisting(
                "../../model/checkpoints/all-MiniLM-L6-v2/tokenizer.json",
                "../model/checkpoints/all-MiniLM-L6-v2/tokenizer.json",
                "model/checkpoints/all-MiniLM-L6-v2/tokenizer.json"
        );
        assertNotNull(tokenizerJsonPath,
                "Could not locate model/checkpoints/all-MiniLM-L6-v2/tokenizer.json relative to the test working directory");

        String tokenizerJson = Files.readString(tokenizerJsonPath, StandardCharsets.UTF_8);
        PromptTokenizer tokenizer = PromptTokenizer.fromTokenizerJson(tokenizerJson);
        assertTrue(tokenizer.vocabSize() > 1000, "Real MiniLM vocab should have thousands of entries, got " + tokenizer.vocabSize());

        JsonArray golden;
        try (InputStream in = getClass().getClassLoader().getResourceAsStream("tokenizer_golden_vectors.json")) {
            assertNotNull(in, "tokenizer_golden_vectors.json test resource not found on classpath");
            golden = JsonParser.parseReader(new java.io.InputStreamReader(in, StandardCharsets.UTF_8)).getAsJsonArray();
        }
        assertTrue(golden.size() >= 10, "Expected a substantial golden-vector set, got " + golden.size());

        int mismatches = 0;
        StringBuilder details = new StringBuilder();
        for (JsonElement el : golden) {
            JsonObject entry = el.getAsJsonObject();
            String prompt = entry.get("prompt").getAsString();
            JsonArray idsJson = entry.getAsJsonArray("token_ids");
            int[] expected = new int[idsJson.size()];
            for (int i = 0; i < expected.length; i++) {
                expected[i] = idsJson.get(i).getAsInt();
            }

            int[] actual = tokenizer.tokenize(prompt, expected.length);
            if (!java.util.Arrays.equals(expected, actual)) {
                mismatches++;
                details.append("\nprompt=\"").append(prompt).append("\"\n  expected=")
                        .append(firstN(expected, 20)).append("\n  actual=  ").append(firstN(actual, 20));
            }
        }

        assertTrue(mismatches == 0, mismatches + " / " + golden.size()
                + " golden vectors mismatched Java tokenizer output against the real MiniLM vocab:" + details);
    }

    private static List<Integer> firstN(int[] arr, int n) {
        List<Integer> out = new ArrayList<>();
        for (int i = 0; i < Math.min(n, arr.length); i++) {
            out.add(arr[i]);
        }
        return out;
    }
}
