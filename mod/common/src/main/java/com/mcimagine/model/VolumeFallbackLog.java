package com.mcimagine.model;

import com.mcimagine.McImagine;

import java.util.EnumMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Reports the one failure mode docs/phase4-plan.md §8 calls "the hardest possible bug to notice":
 * a model-authored {@code block_volume} that could not be used, silently replaced by
 * {@link FallbackTerrain#buildBlockVolumeFromHeightmap}.
 *
 * <p>The replacement is correct - a heightfield volume is a valid volume and generation continues -
 * but it converts a volumetric model into a heightfield one, so the world comes out looking exactly
 * like a pre-volumetric release rather than obviously broken. Nothing in the terrain, the F3 screen,
 * or the model list distinguishes "the model authored flat terrain" from "the model's terrain was
 * thrown away". This log line is the only distinguishing signal, which is why it is worded loudly.
 *
 * <p><b>Deliberately not reported here:</b> the no-model-loaded path
 * ({@link FallbackTerrain#generateFallbackChunkOutput}, reached when {@code ImagineChunkGenerator}
 * has no ONNX graph at all), and a heightmap-only graph whose manifest does not claim the volumetric
 * contract. Both are expected configurations with their own log lines at their own sites, and folding
 * them in here would make the two findings indistinguishable in a log - which is the whole point of
 * this class.
 *
 * <p><b>Why the throttle.</b> The call sites run per chunk generated <em>and</em> per
 * {@code getBaseColumn} query, which vanilla issues per column during spawn search and structure
 * placement - so a plain per-occurrence WARN emits thousands of lines per session; that either drowns
 * the rest of the log or gets filtered out wholesale by the operator, and in both cases the finding is
 * lost again. Instead the first occurrence carries the full diagnosis and later ones are reduced to a
 * running total at occurrence 10, 100, and then every 1000 - O(log n) lines up to the first thousand
 * and one line per thousand thereafter, so the count keeps advancing (an operator who joins the log
 * mid-session still sees it) without ever becoming the bulk of the output.
 *
 * <p>The running total counts <b>occurrences, not distinct chunks</b>, and says so: one broken chunk
 * queried a hundred times during structure placement reports a hundred occurrences. De-duplicating by
 * chunk key would need an unbounded set of keys held for the life of the process, which is a worse
 * trade than a number the log labels honestly.
 *
 * <p>Counts are process-wide per {@link Cause}, not per world or per session: two causes firing at
 * once each get their own loud first line rather than one hiding behind the other.
 */
public final class VolumeFallbackLog {

    /** What was wrong with the model-authored volume. One independent counter and first-report each. */
    public enum Cause {
        /** The {@code ChunkOutput} carried a null {@code block_volume}. */
        VOLUME_NULL("block_volume was null"),
        /** The volume decoded, but not to the {@code 16 x HEIGHT_SPAN x 16} cell count the mod indexes. */
        VOLUME_WRONG_LENGTH("block_volume had the wrong element count"),
        /**
         * The ONNX graph declared a {@code block_volume} output and it decoded to zero elements -
         * an unsupported dtype or a failed buffer read in {@code ModelSession.extractTensorAsInts},
         * both of which log their own cause at WARN and then return an empty array.
         */
        VOLUME_EMPTY("the graph emitted block_volume but it decoded to zero elements"),
        /**
         * The graph emitted no {@code block_volume} tensor at all, from a model whose manifest declares
         * the volumetric 0.6.0 contract - so the volume is not recoverable from the heightmap and the
         * export dropped the one output that contract is about. The same graph shape from a model
         * declaring an older contract is an expected legacy configuration and is not reported here.
         */
        VOLUME_OUTPUT_MISSING("the graph emitted no block_volume output, but the manifest declares "
                + "the volumetric 0.6.0 contract");

        private final String description;

        Cause(String description) {
            this.description = description;
        }

        public String description() {
            return description;
        }
    }

    /**
     * Mutable per-cause state. The map is fully populated in the static initializer and never
     * structurally modified afterwards, so concurrent chunk-generation threads only ever read it.
     */
    private static final class Occurrences {
        private final AtomicLong count = new AtomicLong();
        /** Null until the first report publishes it - see {@link #heightfieldDowngrade} on the ordering. */
        private final AtomicReference<String> firstChunk = new AtomicReference<>(null);
    }

    private static final Map<Cause, Occurrences> STATE = new EnumMap<>(Cause.class);

    static {
        for (Cause cause : Cause.values()) {
            STATE.put(cause, new Occurrences());
        }
    }

    private VolumeFallbackLog() {}

    /**
     * Records one occurrence of a model-authored volume being replaced by a heightmap-derived one,
     * logging per this class's throttle policy.
     *
     * <p>{@code firstChunk} is published <em>before</em> the counter is incremented, and that order is
     * load-bearing: every thread has therefore already published a location by the time its own
     * increment returns, so no throttled line can report an unset one. Doing it the other way round
     * leaves a window in which a thread that increments to 2 reads the location before the thread that
     * incremented to 1 has written it (reproduced: 2 stale reads in 200,000 two-thread trials). The
     * winner of the publish race is not necessarily the thread that increments to 1, so the banner
     * prints the published value rather than its own coordinates - one reported location, consistent
     * across every line about this cause.
     *
     * @param actualLength the rejected volume's element count, or {@code -1} when there was no array
     */
    public static void heightfieldDowngrade(Cause cause, int chunkX, int chunkZ,
                                            int actualLength, int expectedLength) {
        Occurrences state = STATE.get(cause);
        state.firstChunk.compareAndSet(null, "(" + chunkX + ", " + chunkZ + ")");
        long n = state.count.incrementAndGet();
        if (n == 1) {
            String detail = actualLength < 0
                    ? cause.description()
                    : cause.description() + " (got " + actualLength + ", expected " + expectedLength + ")";
            // One call, not a banner built from several: chunk generation is multithreaded and separate
            // LOGGER calls interleave with other threads' output. The headline is on the first line, with
            // no leading newline, so `grep -i warn latest.log` returns the finding rather than the blank
            // line that a leading newline would put the level prefix on.
            McImagine.LOGGER.warn(
                    "*** Mc-Imagine: MODEL VOLUME REJECTED - THIS WORLD IS NOW HEIGHTFIELD TERRAIN ***\n"
                    + "*** {}. First occurrence at chunk {}.\n"
                    + "*** The loaded model's block_volume is being discarded and rebuilt from its heightmap, so the\n"
                    + "*** terrain will have no overhangs, undercuts or open cave mouths - it will look like a working\n"
                    + "*** pre-volumetric release rather than like a failure. The model or its export is at fault;\n"
                    + "*** world settings are not. See docs/phase4-plan.md 5.1.\n"
                    + "*** Further occurrences are throttled to a running total.",
                    detail, state.firstChunk.get());
        } else if (shouldReport(n)) {
            McImagine.LOGGER.warn("Mc-Imagine: still discarding model volumes - {} occurrences so far ({}); "
                    + "first was chunk {}. This counts occurrences, not distinct chunks: one chunk queried "
                    + "repeatedly during structure placement counts each time. See the first occurrence for "
                    + "the diagnosis.",
                    n, cause.description(), state.firstChunk.get());
        }
    }

    /**
     * The throttle policy, separated from the logging so it can be asserted directly: report
     * occurrences 1, 10 and 100, then every 1000th.
     */
    static boolean shouldReport(long occurrence) {
        if (occurrence <= 100) {
            return occurrence == 1 || occurrence == 10 || occurrence == 100;
        }
        return occurrence % 1000 == 0;
    }

    /**
     * How many occurrences of this cause this process has seen. Public rather than package-private
     * because {@code ImagineChunkGeneratorTest} asserts on it from another package.
     */
    public static long occurrences(Cause cause) {
        return STATE.get(cause).count.get();
    }

    /**
     * The location published for this cause's first occurrence, or null if it has none yet. Exists so
     * the publish-before-increment ordering above can be asserted rather than argued.
     */
    static String firstChunk(Cause cause) {
        return STATE.get(cause).firstChunk.get();
    }

    /** Clears every counter. For tests, which share one JVM and therefore one set of counters. */
    public static void reset() {
        for (Occurrences state : STATE.values()) {
            state.count.set(0);
            state.firstChunk.set(null);
        }
    }
}
