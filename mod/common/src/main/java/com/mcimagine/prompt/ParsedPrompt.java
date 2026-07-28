package com.mcimagine.prompt;

import java.util.List;
import java.util.Set;

/**
 * Result of {@link PromptParser#parse}: the prompt text after any unsupported clauses have been stripped
 * (ready for {@link PromptTokenizer}), which semantic tags it actually exercises, and human-readable warnings
 * for anything that was dropped because the selected model doesn't declare support for it
 * (docs/model-spec.md's {@code capabilities.prompt_tags}).
 */
public record ParsedPrompt(
    String normalizedText,
    Set<String> tags,
    List<String> warnings
) {}
