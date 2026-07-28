package com.mcimagine.prompt;

import com.mcimagine.api.ModelInfo.ModelCapabilities;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * The M7 prompt tagger (PROJECT.md §"Model Selection & Prompt Validation"): a regex/keyword rule set that
 * splits a prompt into clauses, tags each with the semantic vocabulary docs/model-spec.md's
 * {@code capabilities.prompt_tags} uses ({@code terrain}, {@code biome_blend}, {@code structures},
 * {@code loot}, {@code redstone}), and - when checked against a specific model's declared tags - strips any
 * clause the model can't act on. Unsupported clauses are never a hard failure, only a warning: a
 * low-intensity/no-structures model asked for "castles with hidden rooms" drops that clause and keeps
 * generating the terrain clause.
 */
public class PromptParser {

    private static final Map<String, Pattern> TAG_KEYWORDS = buildTagKeywords();

    private static Map<String, Pattern> buildTagKeywords() {
        Map<String, Pattern> map = new LinkedHashMap<>();
        map.put("terrain", Pattern.compile(
                "\\b(valley|valleys|mountain|mountains|hill|hills|plain|plains|desert|ocean|river|cave|caves|" +
                "cliff|cliffs|terrain|landscape|plateau|plateaus|canyon|crater|craters|dune|dunes|peak|peaks|" +
                "archipelago|mesa)\\b", Pattern.CASE_INSENSITIVE));
        map.put("biome_blend", Pattern.compile(
                "\\b(biome|biomes|forest|jungle|swamp|tundra|savanna|taiga|blend|flavor|flavors|multiple biomes)\\b",
                Pattern.CASE_INSENSITIVE));
        map.put("structures", Pattern.compile(
                "\\b(castle|castles|ruin|ruins|dungeon|dungeons|temple|temples|village|villages|tower|towers|" +
                "fortress|structure|structures|building|buildings|hidden room|hidden rooms|abandoned)\\b",
                Pattern.CASE_INSENSITIVE));
        map.put("loot", Pattern.compile(
                "\\b(loot|treasure|treasures|chest|chests|reward|rewards)\\b", Pattern.CASE_INSENSITIVE));
        map.put("redstone", Pattern.compile(
                "\\b(redstone|puzzle|puzzles|trap|traps|contraption|contraptions|piston|pistons|lever|levers|" +
                "escape.room)\\b", Pattern.CASE_INSENSITIVE));
        return map;
    }

    private static final Pattern CLAUSE_SPLIT = Pattern.compile("\\s*[,;.]\\s*|\\s+\\band\\b\\s+", Pattern.CASE_INSENSITIVE);

    /** Tags and normalizes a prompt without validating it against any particular model's capabilities. */
    public ParsedPrompt parse(String rawPrompt) {
        return parse(rawPrompt, null);
    }

    /**
     * Tags a prompt, and - if {@code capabilities} is non-null - strips any clause whose only matching
     * tag(s) aren't in {@code capabilities.promptTags()}, producing a warning for each stripped clause.
     */
    public ParsedPrompt parse(String rawPrompt, ModelCapabilities capabilities) {
        String trimmed = rawPrompt == null ? "" : rawPrompt.trim();
        if (trimmed.isEmpty()) {
            return new ParsedPrompt("", Set.of(), List.of());
        }

        // An absent or empty prompt_tags list means "no restriction declared" (true for every legacy/pre-
        // capabilities manifest, including the current dummy model) rather than "supports zero tags" -
        // otherwise every legacy model would warn-and-strip ordinary terrain prompts it never opted into
        // restricting in the first place.
        List<String> declaredTags = capabilities != null ? capabilities.promptTags() : null;
        List<String> allowedTags = (declaredTags == null || declaredTags.isEmpty()) ? null : declaredTags;
        Set<String> keptTags = new LinkedHashSet<>();
        List<String> warnings = new ArrayList<>();
        List<String> keptClauses = new ArrayList<>();

        for (String clause : CLAUSE_SPLIT.split(trimmed)) {
            String clauseTrimmed = clause.trim();
            if (clauseTrimmed.isEmpty()) continue;

            Set<String> clauseTags = tagsFor(clauseTrimmed);
            if (clauseTags.isEmpty()) {
                // No recognized semantic tag - harmless prose (connective phrasing, adjectives); keep as-is.
                keptClauses.add(clauseTrimmed);
                continue;
            }

            if (allowedTags == null) {
                keptTags.addAll(clauseTags);
                keptClauses.add(clauseTrimmed);
                continue;
            }

            Set<String> supported = new LinkedHashSet<>(clauseTags);
            supported.retainAll(allowedTags);

            if (supported.isEmpty()) {
                warnings.add("This model doesn't support " + String.join("/", clauseTags) +
                        "; ignoring clause: \"" + clauseTrimmed + "\"");
            } else {
                keptTags.addAll(supported);
                keptClauses.add(clauseTrimmed);
                if (supported.size() < clauseTags.size()) {
                    Set<String> unsupported = new LinkedHashSet<>(clauseTags);
                    unsupported.removeAll(allowedTags);
                    warnings.add("This model partially supports clause \"" + clauseTrimmed + "\" (missing " +
                            String.join("/", unsupported) + "); generating what it can.");
                }
            }
        }

        String normalized = String.join(", ", keptClauses).trim().toLowerCase();
        return new ParsedPrompt(normalized, keptTags, warnings);
    }

    private Set<String> tagsFor(String clause) {
        Set<String> tags = new LinkedHashSet<>();
        for (Map.Entry<String, Pattern> entry : TAG_KEYWORDS.entrySet()) {
            if (entry.getValue().matcher(clause).find()) {
                tags.add(entry.getKey());
            }
        }
        return tags;
    }
}
