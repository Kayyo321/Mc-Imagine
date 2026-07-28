package com.mcimagine.prompt;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Pure-Java, dependency-free BERT-family WordPiece tokenizer, matching docs/model-spec.md's pinned
 * tokenizer scheme exactly (uncased, matching the MiniLM/{@code all-MiniLM-L6-v2} foundation):
 * lowercase -&gt; strip accents -&gt; split on whitespace/punctuation -&gt; greedy longest-match-first
 * subword splitting with {@code ##} continuation prefixes -&gt; wrap {@code [CLS] ... [SEP]} -&gt; pad to
 * {@code max_prompt_tokens} with {@code [PAD] = 0}.
 *
 * <p>Loads its vocabulary from the {@code tokenizer.json} shipped inside a {@code .mcim} archive (standard
 * HuggingFace {@code tokenizers} format: {@code model.vocab} is a flat {@code token -> id} object).
 */
public class PromptTokenizer {

    public static final int PAD_ID = 0;
    public static final int CLS_ID = 101;
    public static final int SEP_ID = 102;
    public static final String UNK_TOKEN = "[UNK]";
    public static final int DEFAULT_UNK_ID = 100;
    public static final String CONTINUATION_PREFIX = "##";
    private static final int MAX_INPUT_CHARS_PER_WORD = 100;

    private static final Pattern PUNCT_OR_WHITESPACE = Pattern.compile("([\\p{Punct}])|(\\s+)");

    private final Map<String, Integer> vocab;
    private final int unkId;

    public PromptTokenizer(Map<String, Integer> vocab) {
        this.vocab = vocab != null ? vocab : Map.of();
        this.unkId = this.vocab.getOrDefault(UNK_TOKEN, DEFAULT_UNK_ID);
    }

    /** Parses a HuggingFace-format {@code tokenizer.json} string and extracts its WordPiece vocab. */
    public static PromptTokenizer fromTokenizerJson(String tokenizerJson) {
        Map<String, Integer> vocab = new LinkedHashMap<>();
        try {
            JsonObject root = JsonParser.parseString(tokenizerJson).getAsJsonObject();
            if (root.has("model") && root.get("model").isJsonObject()) {
                JsonObject model = root.getAsJsonObject("model");
                if (model.has("vocab") && model.get("vocab").isJsonObject()) {
                    JsonObject vocabObj = model.getAsJsonObject("vocab");
                    for (Map.Entry<String, JsonElement> entry : vocabObj.entrySet()) {
                        vocab.put(entry.getKey(), entry.getValue().getAsInt());
                    }
                }
            }
        } catch (Exception ignored) {
            // Malformed/missing tokenizer.json - fall back to an empty vocab; every word becomes [UNK].
        }
        return new PromptTokenizer(vocab);
    }

    /**
     * Tokenizes {@code text} into {@code [CLS] token... [SEP]}, truncated so the whole sequence (including
     * the two special tokens) never exceeds {@code maxTokens}, then padded with {@code [PAD] = 0} out to
     * exactly {@code maxTokens} elements.
     */
    public int[] tokenize(String text, int maxTokens) {
        int[] out = new int[Math.max(maxTokens, 0)];
        if (out.length == 0) {
            return out;
        }

        List<Integer> ids = new ArrayList<>();
        ids.add(CLS_ID);

        int budget = Math.max(0, maxTokens - 2); // room left for [CLS] (already added) and [SEP]
        int added = 0;
        outer:
        for (String word : basicTokenize(text)) {
            for (int id : wordPieceTokenize(word)) {
                if (added >= budget) {
                    break outer;
                }
                ids.add(id);
                added++;
            }
        }
        if (maxTokens >= 2) {
            ids.add(SEP_ID);
        }

        for (int i = 0; i < maxTokens; i++) {
            out[i] = i < ids.size() ? ids.get(i) : PAD_ID;
        }
        return out;
    }

    /** Lowercase, strip accents (Unicode combining marks), split on whitespace and punctuation. */
    private List<String> basicTokenize(String text) {
        if (text == null || text.isEmpty()) {
            return List.of();
        }
        String normalized = Normalizer.normalize(text, Normalizer.Form.NFD)
                .replaceAll("\\p{M}+", "")
                .toLowerCase();

        List<String> tokens = new ArrayList<>();
        Matcher matcher = PUNCT_OR_WHITESPACE.matcher(normalized);
        int last = 0;
        while (matcher.find()) {
            if (matcher.start() > last) {
                tokens.add(normalized.substring(last, matcher.start()));
            }
            String punct = matcher.group(1);
            if (punct != null) {
                tokens.add(punct);
            }
            last = matcher.end();
        }
        if (last < normalized.length()) {
            tokens.add(normalized.substring(last));
        }
        tokens.removeIf(String::isBlank);
        return tokens;
    }

    /** Greedy longest-match-first WordPiece subword splitting with {@code ##} continuation prefixes. */
    private List<Integer> wordPieceTokenize(String word) {
        if (word.isEmpty()) {
            return List.of();
        }
        if (word.length() > MAX_INPUT_CHARS_PER_WORD) {
            return List.of(unkId);
        }

        List<Integer> subTokenIds = new ArrayList<>();
        int start = 0;
        while (start < word.length()) {
            int end = word.length();
            String matchedToken = null;
            while (end > start) {
                String substr = word.substring(start, end);
                String candidate = start > 0 ? CONTINUATION_PREFIX + substr : substr;
                if (vocab.containsKey(candidate)) {
                    matchedToken = candidate;
                    break;
                }
                end--;
            }
            if (matchedToken == null) {
                return List.of(unkId);
            }
            subTokenIds.add(vocab.get(matchedToken));
            start = end;
        }
        return subTokenIds;
    }

    public int vocabSize() {
        return vocab.size();
    }
}
