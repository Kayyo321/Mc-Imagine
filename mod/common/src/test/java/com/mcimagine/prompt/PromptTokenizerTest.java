package com.mcimagine.prompt;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

/**
 * Golden-vector test for {@link PromptTokenizer} against a small, hand-built vocab (docs/model-spec.md's
 * "Tokenizer Format" section calls for exactly this kind of shared (prompt, token_ids) golden set - the
 * Python side will eventually emit its own set from the real MiniLM/WordPiece vocab, but this repo's Java
 * implementation must be independently, deterministically correct against a known vocabulary first).
 */
public class PromptTokenizerTest {

    private PromptTokenizer tokenizer;

    @BeforeEach
    public void setUp() {
        Map<String, Integer> vocab = new LinkedHashMap<>();
        vocab.put("[PAD]", 0);
        vocab.put("[UNK]", 100);
        vocab.put("[CLS]", 101);
        vocab.put("[SEP]", 102);
        vocab.put("hello", 200);
        vocab.put("world", 201);
        vocab.put("play", 300);
        vocab.put("##ing", 301);
        vocab.put("##ed", 302);
        vocab.put("castle", 303);
        tokenizer = new PromptTokenizer(vocab);
    }

    @Test
    public void emptyStringProducesJustClsAndSepThenPadding() {
        int[] expected = {101, 102, 0, 0, 0, 0, 0, 0};
        assertArrayEquals(expected, tokenizer.tokenize("", 8));
    }

    @Test
    public void singleKnownWord() {
        int[] expected = {101, 200, 102, 0, 0, 0, 0, 0};
        assertArrayEquals(expected, tokenizer.tokenize("hello", 8));
    }

    @Test
    public void unknownWordProducesUnkToken() {
        // "xyzzy" has no matching prefix of any length in the vocab, so it falls back to [UNK].
        int[] expected = {101, 100, 102, 0, 0, 0, 0, 0};
        assertArrayEquals(expected, tokenizer.tokenize("xyzzy", 8));
    }

    @Test
    public void multiSubwordGreedyLongestMatchFirst() {
        // "playing" -> "play" + "##ing" (greedy longest-match-first WordPiece splitting).
        int[] expected = {101, 300, 301, 102, 0, 0, 0, 0};
        assertArrayEquals(expected, tokenizer.tokenize("playing", 8));

        // "played" -> "play" + "##ed"
        int[] expectedPlayed = {101, 300, 302, 102, 0, 0, 0, 0};
        assertArrayEquals(expectedPlayed, tokenizer.tokenize("played", 8));
    }

    @Test
    public void multipleWordsWithPadding() {
        int[] expected = {101, 200, 201, 102, 0, 0, 0, 0, 0, 0};
        assertArrayEquals(expected, tokenizer.tokenize("hello world", 10));
    }

    @Test
    public void truncatesAtMaxPromptTokens() {
        // 5 content words requested but maxTokens=5 only leaves room for 3 content tokens
        // ([CLS] + 3 + [SEP] == 5); the rest of the prompt is dropped rather than overflowing.
        int[] expected = {101, 200, 201, 200, 102};
        assertArrayEquals(expected, tokenizer.tokenize("hello world hello world hello", 5));
    }

    @Test
    public void lowercasesAndStripsAccents() {
        // "Héllo" -> normalize(NFD) strips the combining accent -> lowercase -> "hello", same id as plain "hello".
        int[] expected = {101, 200, 102, 0, 0, 0, 0, 0};
        assertArrayEquals(expected, tokenizer.tokenize("Héllo", 8));
    }

    @Test
    public void wholeWordMatchPreferredOverSubwordSplit() {
        int[] expected = {101, 303, 102, 0, 0, 0, 0, 0};
        assertArrayEquals(expected, tokenizer.tokenize("castle", 8));
    }
}
