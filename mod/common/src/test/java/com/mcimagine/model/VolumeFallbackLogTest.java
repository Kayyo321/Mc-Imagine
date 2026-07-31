package com.mcimagine.model;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Pins the throttle policy, which is the part of docs/phase4-plan.md §5.1's "loudly" that is easy to
 * get wrong in the direction that loses the finding: both call sites run per chunk (and
 * {@code getBaseColumn} runs per column query), so an unthrottled WARN would emit thousands of lines
 * a session and be filtered out by the operator, leaving the downgrade as invisible as it was before.
 *
 * <p>These assert the schedule, not the log output - a log-capturing test would pin the message
 * wording, which is the part that should be free to change.
 */
public class VolumeFallbackLogTest {

    @BeforeEach
    public void resetCounters() {
        // Process-wide counters shared with every other test in this JVM.
        VolumeFallbackLog.reset();
    }

    @Test
    public void theFirstOccurrenceAlwaysReports() {
        assertTrue(VolumeFallbackLog.shouldReport(1),
                "the first occurrence carries the whole diagnosis and must never be throttled");
    }

    @Test
    public void thePerChunkFloodIsThrottled() {
        for (long n = 2; n <= 9; n++) {
            assertFalse(VolumeFallbackLog.shouldReport(n), "occurrence " + n + " should be silent");
        }
        assertTrue(VolumeFallbackLog.shouldReport(10));
        assertFalse(VolumeFallbackLog.shouldReport(11));
        assertTrue(VolumeFallbackLog.shouldReport(100));
        assertTrue(VolumeFallbackLog.shouldReport(1000));

        long reported = 0;
        for (long n = 1; n <= 10_000; n++) {
            if (VolumeFallbackLog.shouldReport(n)) {
                reported++;
            }
        }
        assertEquals(13, reported, "10,000 occurrences must cost 13 log lines, not 10,000");
    }

    @Test
    public void theCountKeepsAdvancingForever() {
        // Not a geometric backoff: an operator who joins the log after a million chunks still gets a
        // running total rather than a log whose last mention of the problem scrolled past hours ago.
        assertTrue(VolumeFallbackLog.shouldReport(1_000_000));
        assertTrue(VolumeFallbackLog.shouldReport(1_001_000));
    }

    @Test
    public void causesAreCountedIndependently() {
        // Two failure modes at once must not hide behind each other: each gets its own loud first line,
        // which is the same reason the no-model path is kept out of this log entirely.
        VolumeFallbackLog.heightfieldDowngrade(VolumeFallbackLog.Cause.VOLUME_NULL, 1, 2, -1, 98_304);
        VolumeFallbackLog.heightfieldDowngrade(VolumeFallbackLog.Cause.VOLUME_NULL, 1, 3, -1, 98_304);
        VolumeFallbackLog.heightfieldDowngrade(VolumeFallbackLog.Cause.VOLUME_EMPTY, 4, 5, 0, 98_304);

        assertEquals(2, VolumeFallbackLog.occurrences(VolumeFallbackLog.Cause.VOLUME_NULL));
        assertEquals(1, VolumeFallbackLog.occurrences(VolumeFallbackLog.Cause.VOLUME_EMPTY));
        assertEquals(0, VolumeFallbackLog.occurrences(VolumeFallbackLog.Cause.VOLUME_WRONG_LENGTH));
    }

    @Test
    public void theFirstChunkIsPublishedBeforeTheCountThatReportsIt() throws Exception {
        // The throttled line quotes the first chunk, so a count that is visible before that location is
        // a line that says "unknown" - losing the one coordinate an operator would go and look at.
        // Publishing before incrementing makes "count >= 1 implies location != null" hold for every
        // observer.
        //
        // Be clear about what this proves: the window it targets exists only around the very first
        // occurrence, so one run is one trial of that race no matter how many calls follow, and a pass
        // is evidence rather than proof (the reversed order was measured elsewhere at ~1 stale read in
        // 100,000 trials). A failure, on the other hand, is always real - the invariant cannot fail on
        // correct code. The bulk of the calls are here for the second assertion, which is not racy.
        VolumeFallbackLog.Cause cause = VolumeFallbackLog.Cause.VOLUME_WRONG_LENGTH;
        AtomicBoolean sawCountWithoutLocation = new AtomicBoolean(false);
        AtomicBoolean done = new AtomicBoolean(false);

        Thread observer = new Thread(() -> {
            while (!done.get()) {
                if (VolumeFallbackLog.occurrences(cause) >= 1 && VolumeFallbackLog.firstChunk(cause) == null) {
                    sawCountWithoutLocation.set(true);
                    return;
                }
            }
        });
        observer.start();

        int threads = 4;
        int callsPerThread = 2_500;
        Thread[] producers = new Thread[threads];
        for (int t = 0; t < threads; t++) {
            int id = t;
            producers[t] = new Thread(() -> {
                for (int i = 0; i < callsPerThread; i++) {
                    VolumeFallbackLog.heightfieldDowngrade(cause, id, i, 7, 98_304);
                }
            });
            producers[t].start();
        }
        for (Thread producer : producers) {
            producer.join();
        }
        done.set(true);
        observer.join();

        assertFalse(sawCountWithoutLocation.get(), "a counted occurrence was visible before its location");
        assertEquals((long) threads * callsPerThread, VolumeFallbackLog.occurrences(cause),
                "AtomicLong must not lose increments under contention");
        assertNotNull(VolumeFallbackLog.firstChunk(cause));
    }
}
