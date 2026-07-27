package com.gausscapture.capture;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Handler;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Records inertial data alongside the video, and estimates motion blur from it.
 *
 * <p>Timestamps are written in {@code elapsedRealtimeNanos}, the same clock the camera reports
 * when its timestamp source is {@code REALTIME}. On such a device video frames and IMU samples
 * share one hardware clock, so they align without any correlation step -- which a browser cannot
 * do at any price, and which is the reason this is a native app.
 */
public final class SensorLogger {

    /**
     * Rolling window for the angular-rate estimate, in samples.
     *
     * <p>The gyroscope runs at a few hundred hertz. A single sample is far too twitchy to drive a
     * user-facing warning, so the guidance uses a short median-free mean over roughly a tenth of a
     * second: long enough to be stable, short enough to react before a whole pass is ruined.
     */
    private static final int RATE_WINDOW = 32;

    private final SensorManager sensorManager;
    private final List<Sensor> registered = new ArrayList<>();

    private BufferedWriter writer;
    private volatile int sampleCount;
    private long startRealtimeNanos;

    private final float[] rateWindow = new float[RATE_WINDOW];
    private int rateIndex;
    private int rateFilled;

    public SensorLogger(SensorManager sensorManager) {
        this.sensorManager = sensorManager;
    }

    public int sampleCount() {
        return sampleCount;
    }

    /** Smoothed angular speed in radians per second, or 0 before any gyro data arrives. */
    public float angularRate() {
        if (rateFilled == 0) {
            return 0f;
        }
        float sum = 0f;
        for (int i = 0; i < rateFilled; i++) {
            sum += rateWindow[i];
        }
        return sum / rateFilled;
    }

    public void start(File output, long startRealtimeNanos) throws IOException {
        this.startRealtimeNanos = startRealtimeNanos;
        this.sampleCount = 0;
        this.rateIndex = 0;
        this.rateFilled = 0;
        this.writer = new BufferedWriter(new FileWriter(output));
    }

    /**
     * Begins listening. Registered even when not recording, so that the guidance has data to work
     * with before the first take starts.
     */
    public void listen(Handler handler) {
        registered.clear();
        add(Sensor.TYPE_ACCELEROMETER);
        add(Sensor.TYPE_GYROSCOPE);
        add(Sensor.TYPE_ROTATION_VECTOR);
        // The uncalibrated variants carry the raw signal plus the bias the system is subtracting,
        // which is what anyone doing their own integration needs.
        add(Sensor.TYPE_GYROSCOPE_UNCALIBRATED);
        add(Sensor.TYPE_ACCELEROMETER_UNCALIBRATED);
        for (Sensor sensor : registered) {
            sensorManager.registerListener(listener, sensor,
                    SensorManager.SENSOR_DELAY_FASTEST, handler);
        }
    }

    public void stopListening() {
        sensorManager.unregisterListener(listener);
        registered.clear();
    }

    /** Stops writing but keeps listening, so guidance survives between takes. */
    public synchronized void closeFile() {
        if (writer == null) {
            return;
        }
        try {
            writer.flush();
            writer.close();
        } catch (IOException ignored) {
            // Nothing useful to do at this point.
        }
        writer = null;
    }

    private void add(int type) {
        Sensor sensor = sensorManager.getDefaultSensor(type);
        if (sensor != null) {
            registered.add(sensor);
        }
    }

    private final SensorEventListener listener = new SensorEventListener() {
        @Override
        public void onSensorChanged(SensorEvent event) {
            if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE && event.values.length >= 3) {
                float x = event.values[0];
                float y = event.values[1];
                float z = event.values[2];
                rateWindow[rateIndex] = (float) Math.sqrt(x * x + y * y + z * z);
                rateIndex = (rateIndex + 1) % RATE_WINDOW;
                rateFilled = Math.min(rateFilled + 1, RATE_WINDOW);
            }

            BufferedWriter target = writer;
            if (target == null) {
                return;  // Listening for guidance only; not inside a take.
            }
            try {
                JSONObject row = new JSONObject();
                row.put("type", event.sensor.getStringType());
                // Absolute value aligns with frame timestamps; the relative one is readable.
                row.put("t_ns", event.timestamp);
                row.put("t_rel_ns", event.timestamp - startRealtimeNanos);
                JSONArray values = new JSONArray();
                for (float v : event.values) {
                    values.put((double) v);
                }
                row.put("v", values);
                row.put("accuracy", event.accuracy);
                synchronized (SensorLogger.this) {
                    if (writer != null) {
                        writer.write(row.toString());
                        writer.write("\n");
                    }
                }
                sampleCount++;
            } catch (Exception ignored) {
                // A dropped sample must never interrupt a recording.
            }
        }

        @Override
        public void onAccuracyChanged(Sensor sensor, int accuracy) { }
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
    public static float blurPixels(float omega, float exposureSeconds, float focalPixels) {
        if (exposureSeconds <= 0 || focalPixels <= 0) {
            return 0f;
        }
        return Math.abs(omega) * exposureSeconds * focalPixels;
    }
}
