package com.gausscapture.capture;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;

import java.io.File;

/**
 * Records inertial data alongside the video, on a thread of its own.
 *
 * <p>This stream is the most valuable thing the app produces. Pairing each frame's camera pose
 * with the accelerometer reading taken at that instant is what recovers which way is up — measured
 * at 0.998 agreement across a capture, against heuristics that were wrong by as much as 41 degrees.
 *
 * <p>Which is why it no longer shares a handler with the camera. The previous version delivered
 * five sensors' callbacks and every capture result onto one background thread that also did
 * synchronous disk writes: roughly 2000 events a second behind a single blocking writer, where any
 * flush stall back-pressured the sensor queue and the framework responded by dropping samples
 * (audit D10). Here the sensors get a dedicated high-priority thread, and the disk is somebody
 * else's problem ({@link JsonlWriter}).
 *
 * <p>Five sensors are registered rather than two. The calibrated accelerometer and gyroscope are
 * what the pipeline reads today; the <em>uncalibrated</em> variants carry the raw signal alongside
 * the bias the system is subtracting, which is the only way a consumer can ever redo the
 * integration its own way; the rotation vector is the platform's own fused orientation and serves
 * as an independent check on anything derived from the others. None of it can be recovered after
 * the fact, and together they cost a few megabytes a minute.
 */
final class SensorLogger {

    /**
     * Rolling window for the angular-rate estimate, in samples.
     *
     * <p>The gyroscope runs at a few hundred hertz. A single sample is far too twitchy to drive a
     * user-facing warning, so the guidance uses a short mean over roughly a tenth of a second:
     * long enough to be stable, short enough to react before a whole pass is ruined.
     */
    private static final int RATE_WINDOW = 32;

    private final SensorManager sensorManager;
    private final Clocks clocks;

    private HandlerThread thread;
    private Handler handler;
    private JsonlWriter writer;
    private volatile long startRealtimeNanos;
    private volatile int sampleCount;

    private final float[] rateWindow = new float[RATE_WINDOW];
    private volatile int rateIndex;
    private volatile int rateFilled;

    /**
     * Reusable line buffer. Only the sensor thread touches it, so it needs no synchronisation and
     * costs no allocation per event — which is the point at two kilohertz. The old path built a
     * JSONObject, a JSONArray and a String for every single sample.
     */
    private final StringBuilder line = new StringBuilder(192);

    SensorLogger(SensorManager sensorManager, Clocks clocks) {
        this.sensorManager = sensorManager;
        this.clocks = clocks;
    }

    int sampleCount() {
        return sampleCount;
    }

    /** Smoothed angular speed in radians per second, or 0 before any gyro data arrives. */
    float angularRate() {
        int filled = rateFilled;
        if (filled == 0) {
            return 0f;
        }
        float sum = 0f;
        for (int i = 0; i < filled; i++) {
            sum += rateWindow[i];
        }
        return sum / filled;
    }

    /**
     * Achieved sample rate in hertz.
     *
     * <p>Measured, because the requested rate is not the delivered one: without
     * {@code HIGH_SAMPLING_RATE_SENSORS} the platform silently caps sensors at 200 Hz from API 31,
     * and thermal throttling can cut it further mid-capture. A preflight check that reported the
     * requested rate would always pass.
     */
    int measuredRateHz() {
        long start = startRealtimeNanos;
        if (start == 0) {
            return 0;
        }
        long elapsed = SystemClock.elapsedRealtimeNanos() - start;
        return elapsed <= 0 ? 0 : (int) (sampleCount * 1_000_000_000L / elapsed);
    }

    long linesWritten() {
        return writer == null ? 0 : writer.linesWritten();
    }

    /** Samples lost to a full buffer. Zero on a healthy capture; recorded either way. */
    long linesDropped() {
        return writer == null ? 0 : writer.linesDropped();
    }

    String failure() {
        return writer == null ? null : writer.failure();
    }

    boolean hasRequiredSensors() {
        return sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER) != null
                && sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE) != null;
    }

    /** Begin logging into {@code output}. */
    void start(File output, long startRealtimeNanos) {
        this.startRealtimeNanos = startRealtimeNanos;
        this.sampleCount = 0;
        this.rateIndex = 0;
        this.rateFilled = 0;
        this.writer = new JsonlWriter(output);

        thread = new HandlerThread("gausscapture-imu");
        thread.start();
        handler = new Handler(thread.getLooper());

        register(Sensor.TYPE_ACCELEROMETER);
        register(Sensor.TYPE_GYROSCOPE);
        register(Sensor.TYPE_ROTATION_VECTOR);
        register(Sensor.TYPE_GYROSCOPE_UNCALIBRATED);
        register(Sensor.TYPE_ACCELEROMETER_UNCALIBRATED);
    }

    private void register(int type) {
        Sensor sensor = sensorManager.getDefaultSensor(type);
        if (sensor == null) {
            return;
        }
        sensorManager.registerListener(listener, sensor, SensorManager.SENSOR_DELAY_FASTEST, handler);
    }

    void stop() {
        sensorManager.unregisterListener(listener);
        if (writer != null) {
            writer.close();
        }
        if (thread != null) {
            thread.quitSafely();
            try {
                thread.join(2000);
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
            }
            thread = null;
            handler = null;
        }
    }

    private final SensorEventListener listener = new SensorEventListener() {
        @Override
        public void onSensorChanged(SensorEvent event) {
            // The first event settles which clock the sensors are on. Measuring it rather than
            // trusting the documentation is the whole of audit D12: several OEM builds report
            // CLOCK_MONOTONIC where CLOCK_BOOTTIME is specified, and the resulting error is the
            // accumulated deep-sleep time since boot.
            clocks.observeSensorEvent(event.timestamp);

            if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE) {
                float omega = (float) Math.sqrt(
                        event.values[0] * event.values[0]
                                + event.values[1] * event.values[1]
                                + event.values[2] * event.values[2]);
                rateWindow[rateIndex] = omega;
                rateIndex = (rateIndex + 1) % RATE_WINDOW;
                if (rateFilled < RATE_WINDOW) {
                    rateFilled++;
                }
            }

            // The absolute t_ns is the primary key -- it is what lets a frame find its sample --
            // and t_rel_ns is alongside it only because it is readable. Recording both is what
            // makes the alignment possible at all.
            line.setLength(0);
            line.append("{\"type\":\"").append(event.sensor.getStringType())
                    .append("\",\"t_ns\":").append(event.timestamp)
                    .append(",\"t_rel_ns\":").append(event.timestamp - startRealtimeNanos)
                    .append(",\"v\":[");
            for (int i = 0; i < event.values.length; i++) {
                if (i > 0) {
                    line.append(',');
                }
                line.append(event.values[i]);
            }
            line.append("],\"accuracy\":").append(event.accuracy).append('}');

            writer.append(line);
            sampleCount++;
        }

        @Override
        public void onAccuracyChanged(Sensor sensor, int accuracy) {
            // Recorded per row in the "accuracy" field, so there is nothing to do here.
        }
    };

    /**
     * Predicted motion blur, in pixels, from angular rate and exposure time.
     *
     * <p>This is geometry rather than a fitted score: a camera rotating at {@code omega} radians
     * per second during an exposure of {@code exposureSeconds} sweeps the image across
     * {@code omega * exposureSeconds} radians, and at a focal length of {@code focalPixels} that
     * is {@code omega * exposureSeconds * focalPixels} pixels of smear. Nothing here is calibrated
     * or guessed, which is why it is honest to show it during capture while the project's actual
     * quality score is still unvalidated.
     *
     * <p>Translation also blurs, but converting that to pixels needs scene depth, which we do not
     * have. Rotation dominates in handheld capture anyway.
     */
    static float blurPixels(float omega, float exposureSeconds, float focalPixels) {
        if (exposureSeconds <= 0 || focalPixels <= 0) {
            return 0f;
        }
        return Math.abs(omega) * exposureSeconds * focalPixels;
    }
}
