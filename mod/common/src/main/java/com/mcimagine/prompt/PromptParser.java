package com.mcimagine.prompt;

import java.util.Map;
import java.util.Collections;

public class PromptParser {
    
    public ParsedPrompt parse(String rawPrompt) {
        return new ParsedPrompt(new int[0], Collections.emptyMap(), rawPrompt.trim().toLowerCase());
    }

    public void extractStructureConstraints() {
        // Blacklist/whitelist logic
    }
}\n