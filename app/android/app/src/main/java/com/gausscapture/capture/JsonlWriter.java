package com.gausscapture.capture;

import android.util.Log;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;

/**
 * A line writer that never blocks the thread producing the lines.
 *
 * <p>The previous version wrote every sensor event and every capture result synchronously, on one
 * shared background thread, straight into a {@code BufferedWriter} on external storage. Five
 * sensors at the fastest rate produce roughly 2000 events a second on an S22. A single flush stall
 * therefore delayed camera metadata <em>and</em> back-pressured the sensor queue, which the
 * framework resolves by dropping samples (audit D10). The most valuable stream in the project was
 * the one being starved.
 *
 * <p>So producers here only append to an in-memory buffer, and a dedicated thread does the I/O.
 * Two buffers are used alternately: a producer appends to one while the writer drains the other,
 * so the lock is held for a pointer swap rather than for a disk write. No object is allocated per
 * event, which matters at 2 kHz — the old path allocated a {@code JSONObject}, a {@code JSONArray}
 * and a {@code String} for every sample.
 *
 * <p>Two smaller things follow from the same concern. The file is flushed on a timer rather than
 * only at the end, so process death costs a fraction of a second rather than the last 8 KB of
 * every stream (audit D24). And when the buffer does overflow, the count of lost lines is kept and
 * written into the manifest: a gap that is known about is a limitation, and a gap that is not is a
 * wrong answer.
 */
final class JsonlWriter {

    private static final String TAG = "JsonlWriter";

    /** How often the writer thread wakes to move whatever has accumulated onto disk. */
    private static final long FLUSH_INTERVAL_MS = 500;

    /**
     * Cap on buffered characters before lines start being dropped. At roughly 120 bytes a line and
     * 2000 lines a second, this is about eight seconds of IMU — far more headroom than a healthy
     * flush needs, and a bound that keeps a stalled disk from becoming an out-of-memory kill.
     */
    private static final int MAX_BUFFERED_CHARS = 2_000_000;

    private final File file;
    private final Thread thread;

    private final Object lock = new Object();
    private StringBuilder active = new StringBuilder(1 << 16);
    private StringBuilder draining = new StringBuilder(1 << 16);

    private volatile boolean running = true;
    private volatile long linesWritten;
    private volatile long linesDropped;
    private volatile String failure;

    JsonlWriter(File file) {
        this.file = file;
        this.thread = new Thread(new Runnable() {
            @Override
            public void run() {
                pump();
            }
        }, "jsonl-" + file.getName());
        this.thread.start();
    }

    /**
     * Queue one line. Cheap enough to call from a sensor callback at 2 kHz.
     *
     * <p>The newline is added here so callers cannot forget it and produce a file that parses as
     * one enormous record.
     */
    void append(CharSequence line) {
        synchronized (lock) {
            if (active.length() >= MAX_BUFFERED_CHARS) {
                linesDropped++;
                return;
            }
            active.append(line).append('\n');
        }
    }

    private void pump() {
        BufferedWriter writer = null;
        try {
            writer = new BufferedWriter(
                    new OutputStreamWriter(new FileOutputStream(file), StandardCharsets.UTF_8),
                    1 << 16);
            while (running) {
                try {
                    Thread.sleep(FLUSH_INTERVAL_MS);
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    break;
                }
                drainOnce(writer);
            }
            drainOnce(writer);   // whatever arrived after the last tick
        } catch (IOException error) {
            failure = error.getMessage();
            Log.e(TAG, "writing " + file.getName(), error);
        } finally {
            if (writer != null) {
                try {
                    writer.close();
                } catch (IOException ignored) {
                    // Nothing useful is left to do about a close failure at this point.
                }
            }
        }
    }

    private void drainOnce(BufferedWriter writer) throws IOException {
        StringBuilder ready;
        synchronized (lock) {
            if (active.length() == 0) {
                return;
            }
            // Swap rather than copy: the producer is free again immediately, and the write below
            // happens with no lock held at all.
            ready = active;
            active = draining;
            draining = ready;
        }
        writer.append(ready);
        writer.flush();
        long lines = 0;
        for (int i = 0; i < ready.length(); i++) {
            if (ready.charAt(i) == '\n') {
                lines++;
            }
        }
        linesWritten += lines;
        ready.setLength(0);
    }

    /** Stop accepting lines and wait for everything already queued to reach the disk. */
    void close() {
        running = false;
        thread.interrupt();
        try {
            thread.join(5000);
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    long linesWritten() {
        return linesWritten;
    }

    /** Lines lost to a full buffer. Zero on a healthy capture; recorded either way. */
    long linesDropped() {
        return linesDropped;
    }

    /** The I/O error that stopped this stream, or null. */
    String failure() {
        return failure;
    }
}
