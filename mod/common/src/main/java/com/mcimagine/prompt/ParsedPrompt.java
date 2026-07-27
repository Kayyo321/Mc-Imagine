package com.mcimagine.prompt;

import java.util.Map;

public record ParsedPrompt(
    int[] tokens,
    Map<String, Float> structureWeights,
    String normalizedText
) {}\n