package com.gausscapture.capture;

import android.app.Activity;
import android.os.Build;
import android.os.SystemClock;
import android.view.Surface;
import android.view.TextureView;
import android.hardware.SensorManager;
import android.hardware.camera2.CameraManager;
import android.content.Context;

import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.TimeZone;
import java.util.UUID;

/**
 * Everything a capture consists of, with no interface attached.
 *
 * <p>In the Java version this lived inside the Activity, mixed together with buttons and layout
 * inflation. Flutter now owns the interface, which forces a separation that was always the right
 * one: the parts that decide what gets recorded, and the parts that decide what gets drawn, are
 * different problems with different failure modes. Nothing here knows a widget exists.
 *
 * <p>What it does own is the guarantee that a take is either complete or absent. Starting means a
 * directory, an IMU stream, an encoder and a metadata stream all coming up together; stopping means
 * all of them coming down and a manifest describing what actually happened, measured rather than
 * asserted.
 */
public final class CaptureEngine {

    /** Bumped whenever the recorded schema changes, so a pack can be traced to what produced it. */
    static final String APP_VERSION = "gausscapture-flutter/0.3";

    public interface Events {
        /** A take ended for a reason the operator did not choose. */
        void onProblem(String message);
        /** A take started or stopped; the interface should re-read state. */
        void onStateChanged();
    }

    private final Activity activity;
    private final Clocks clocks = new Clocks();
    private final SensorLogger sensors;
    private final CameraManager cameraManager;
    private Storage storage;
    private CameraController camera;
    private Events events;

    /**
     * Where the chosen protocol is remembered.
     *
     * <p>It was not remembered at all before. The operator picked C, the process was recreated for
     * any of the ordinary reasons, and the next take was silently written as A -- and a
     * mislabelled capture is not a weaker data point, it is a wrong one. Preset's own
     * documentation makes that argument; this is the case it did not defend against (audit D26).
     */
    private static final String PREFERENCES = "gausscapture";
    private static final String KEY_PRESET = "preset";

    private File sessionDir;
    private Preset preset = Preset.ALL[0];
    private volatile boolean recording;
    private volatile boolean starting;
    private long startRealtimeNanos;
    private long startUptimeNanos;
    private long startWallMillis;

    public CaptureEngine(Activity activity) {
        this.activity = activity;
        this.cameraManager = (CameraManager) activity.getSystemService(Context.CAMERA_SERVICE);
        this.sensors = new SensorLogger(
                (SensorManager) activity.getSystemService(Context.SENSOR_SERVICE), clocks);
        this.storage = Storage.open(activity);

        String remembered = activity.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
                .getString(KEY_PRESET, null);
        if (remembered != null) {
            setPreset(remembered);
        }
    }

    public void setEvents(Events events) {
        this.events = events;
    }

    /** Re-read the storage location: the all-files grant can change while backgrounded. */
    public void refreshStorage() {
        storage = Storage.open(activity);
    }

    public Storage storage() {
        return storage;
    }

    public Clocks clocks() {
        return clocks;
    }

    public boolean isRecording() {
        return recording;
    }

    public Preset preset() {
        return preset;
    }

    public void setPreset(String id) {
        for (Preset candidate : Preset.ALL) {
            if (candidate.id.equals(id)) {
                preset = candidate;
                activity.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
                        .edit()
                        .putString(KEY_PRESET, id)
                        .apply();
                return;
            }
        }
    }

    // ------------------------------------------------------------------ camera

    /** Bind to the preview surface Flutter is hosting, and open the camera behind it. */
    public void attachPreview(TextureView view, CameraController.Listener listener) {
        camera = new CameraController(activity, view, cameraManager, listener);
        camera.setRecorderProblem(new CameraController.RecorderProblem() {
            @Override
            public void onRecorderProblem(String message, boolean fatal) {
                if (recording) {
                    stop();
                    if (events != null) {
                        events.onProblem(message);
                    }
                }
            }
        });
    }

    public CameraController camera() {
        return camera;
    }

    public void openCamera(int displayRotation) {
        if (camera != null) {
            camera.open(displayRotation);
            clocks.observeCamera(camera.characteristics());
        }
    }

    public void closeCamera() {
        if (recording) {
            stop();
        }
        if (camera != null) {
            camera.close();
        }
    }

    // --------------------------------------------------------------- recording

    /** @return null on success, or why the take could not start. */
    public String start(int displayRotation) {
        if (recording || starting || camera == null) {
            return "A take is already under way";
        }
        starting = true;

        sessionDir = storage.newSessionDir(preset.id);
        if (sessionDir == null) {
            starting = false;
            return "Cannot create a folder for this take";
        }

        startRealtimeNanos = SystemClock.elapsedRealtimeNanos();
        startUptimeNanos = System.nanoTime();
        startWallMillis = System.currentTimeMillis();
        sensors.start(new File(sessionDir, "imu.jsonl"), startRealtimeNanos);

        final String[] failure = new String[1];
        final Object latch = new Object();
        final boolean[] settled = new boolean[1];

        camera.startRecording(sessionDir, displayRotation, new CameraController.RecordingCallback() {
            @Override
            public void onStarted() {
                synchronized (latch) {
                    recording = true;
                    settled[0] = true;
                    latch.notifyAll();
                }
                if (events != null) {
                    events.onStateChanged();
                }
            }

            @Override
            public void onFailed(String message) {
                sensors.stop();
                deleteRecursively(sessionDir);
                synchronized (latch) {
                    failure[0] = message;
                    settled[0] = true;
                    latch.notifyAll();
                }
            }
        });

        // Session configuration is asynchronous and takes 100-500 ms. Waiting for the outcome here
        // means the method channel answers with the truth rather than with an optimistic "started"
        // that the interface would have to unwind.
        synchronized (latch) {
            long deadline = System.currentTimeMillis() + 8000;
            while (!settled[0] && System.currentTimeMillis() < deadline) {
                try {
                    latch.wait(deadline - System.currentTimeMillis());
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }
        starting = false;
        if (!settled[0]) {
            return "The camera did not respond";
        }
        return failure[0];
    }

    /** @return a description of the finished take, or null when nothing usable was produced. */
    public JSONObject stop() {
        if (!recording) {
            return null;
        }
        recording = false;

        boolean encoded = camera.stopRecording();
        sensors.stop();

        File video = new File(sessionDir, "video.mp4");
        boolean hasVideo = video.exists() && video.length() > 0;

        // The old code deleted the whole session -- video and IMU together -- whenever stop() threw,
        // on the theory that it only throws for sub-frame takes. It also throws when the muxer
        // fails at the end of a long recording, and a minute of walking is not worth discarding
        // because the trailer did not write (audit D13). So the file is judged by whether it
        // exists, not by whether the call was clean.
        if (!hasVideo) {
            deleteRecursively(sessionDir);
            if (events != null) {
                events.onStateChanged();
            }
            return null;
        }

        JSONObject manifest = writeManifest(encoded);
        if (events != null) {
            events.onStateChanged();
        }
        return manifest;
    }

    public long elapsedMillis() {
        return recording ? (SystemClock.elapsedRealtimeNanos() - startRealtimeNanos) / 1_000_000L : 0;
    }

    /**
     * Predicted motion blur in pixels for the current instant.
     *
     * <p>Geometry, not a score: angular rate times exposure time times focal length. It is the only
     * quality signal this project can defend today, which is why it is the only one shown.
     */
    public float blurPixels() {
        if (camera == null) {
            return 0f;
        }
        float focal = camera.focalPixels();
        float exposureSeconds = camera.lastExposureNanos() / 1_000_000_000f;
        return SensorLogger.blurPixels(sensors.angularRate(), exposureSeconds, focal);
    }

    public SensorLogger sensors() {
        return sensors;
    }

    // ---------------------------------------------------------------- manifest

    private JSONObject writeManifest(boolean cleanStop) {
        try {
            JSONObject manifest = new JSONObject();
            manifest.put("capturepack_version", "0.1");
            manifest.put("session_id", UUID.randomUUID().toString());
            manifest.put("session_name", sessionDir.getName());
            manifest.put("preset", preset.id);
            manifest.put("preset_target_seconds", preset.targetSeconds);
            manifest.put("expected_to_fail", preset.expectedToFail);
            manifest.put("capture_type", "E_subject".equals(preset.id)
                    ? "dynamic_subject" : "static_scene");
            manifest.put("created_at", isoTime(startWallMillis));
            manifest.put("app", APP_VERSION);
            manifest.put("clean_stop", cleanStop);

            JSONObject device = new JSONObject();
            device.put("manufacturer", Build.MANUFACTURER);
            device.put("model", Build.MODEL);
            device.put("os", "Android " + Build.VERSION.RELEASE
                    + " (API " + Build.VERSION.SDK_INT + ")");
            device.put("camera_id", camera.cameraId());
            device.put("sensor_orientation", camera.sensorOrientation());
            manifest.put("device", device);

            JSONObject video = new JSONObject();
            video.put("main_file", "video.mp4");
            video.put("width", camera.videoSize().getWidth());
            video.put("height", camera.videoSize().getHeight());
            video.put("fps", 30);
            video.put("codec", "h264");
            video.put("orientation_hint", camera.previewRotation(displayRotation()));
            video.put("frames_recorded", camera.frameCount());
            video.put("has_audio", false);
            video.put("bytes", new File(sessionDir, "video.mp4").length());
            manifest.put("video", video);

            // Requested, not asserted. A Samsung HAL honours these only partially, and frames.jsonl
            // carries the per-frame 3A and lens state that says what actually happened (audit D11).
            JSONObject settings = new JSONObject();
            settings.put("exposure_lock_requested", true);
            settings.put("white_balance_lock_requested", true);
            settings.put("focus_lock_requested", true);
            settings.put("stabilisation_disable_requested", true);
            settings.put("verified_per_frame_in", "frames.jsonl");
            settings.put("exposure", camera.exposureSettings());
            settings.put("storage_mode", storage.isVisibleToDesktop() ? "shared" : "app_private");
            settings.put("storage_path", storage.describePath());
            manifest.put("capture_settings", settings);

            manifest.put("time_base", "nanoseconds_elapsed_realtime");
            manifest.put("clocks", clocks.toJson(startRealtimeNanos, startUptimeNanos,
                    startWallMillis, camera.firstEncodedFrameNs()));

            JSONObject files = new JSONObject();
            files.put("intrinsics", "intrinsics.json");
            files.put("imu", "imu.jsonl");
            files.put("frames", "frames.jsonl");
            files.put("poses", JSONObject.NULL);
            files.put("light", JSONObject.NULL);
            manifest.put("metadata_files", files);
            manifest.put("imu_samples", sensors.sampleCount());

            JSONObject health = new JSONObject();
            health.put("imu_lines_written", sensors.linesWritten());
            health.put("imu_lines_dropped", sensors.linesDropped());
            health.put("imu_rate_hz", sensors.measuredRateHz());
            health.put("frame_rows_dropped", camera.framesDropped());
            String imuFailure = sensors.failure();
            health.put("imu_write_error", imuFailure == null ? JSONObject.NULL : imuFailure);
            manifest.put("stream_health", health);

            write(new File(sessionDir, "manifest.json"), manifest.toString(2));
            write(new File(sessionDir, "intrinsics.json"), camera.intrinsics().toString(2));
            return manifest;
        } catch (Exception error) {
            if (events != null) {
                events.onProblem("Could not write metadata: " + error.getMessage());
            }
            return null;
        }
    }

    // ------------------------------------------------------------------- takes

    /** Every take on disk, described well enough for a list and a detail screen. */
    public List<JSONObject> takes() {
        List<JSONObject> out = new ArrayList<>();
        for (File dir : storage.sessions()) {
            try {
                JSONObject entry = new JSONObject();
                entry.put("name", dir.getName());
                entry.put("path", dir.getAbsolutePath());
                entry.put("bytes", Storage.directorySize(dir));
                entry.put("offloaded", Storage.isOffloaded(dir));
                entry.put("incomplete", Storage.isIncomplete(dir));

                File manifest = new File(dir, "manifest.json");
                if (manifest.exists()) {
                    JSONObject data = new JSONObject(read(manifest));
                    entry.put("manifest", data);
                }
                out.add(entry);
            } catch (Exception ignored) {
                // A take that cannot be described is still shown by name above; skipping the whole
                // row would make it invisible and therefore undeletable.
            }
        }
        return out;
    }

    public boolean deleteTake(String name) {
        File dir = new File(storage.root(), name);
        // Guard against a name that climbs out of the storage root.
        if (!dir.getParentFile().equals(storage.root()) || !dir.isDirectory()) {
            return false;
        }
        return deleteRecursively(dir);
    }

    // ------------------------------------------------------------------ helper

    private int displayRotation() {
        return activity.getWindowManager().getDefaultDisplay().getRotation();
    }

    static String isoTime(long millis) {
        SimpleDateFormat format =
                new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US);
        format.setTimeZone(TimeZone.getTimeZone("UTC"));
        return format.format(new Date(millis));
    }

    private static void write(File file, String content) throws Exception {
        try (OutputStreamWriter writer = new OutputStreamWriter(
                new FileOutputStream(file), StandardCharsets.UTF_8)) {
            writer.write(content);
        }
    }

    private static String read(File file) throws Exception {
        byte[] bytes = new byte[(int) file.length()];
        try (java.io.FileInputStream stream = new java.io.FileInputStream(file)) {
            int read = stream.read(bytes);
            return new String(bytes, 0, Math.max(0, read), StandardCharsets.UTF_8);
        }
    }

    private static boolean deleteRecursively(File file) {
        if (file == null) {
            return false;
        }
        File[] children = file.listFiles();
        if (children != null) {
            for (File child : children) {
                deleteRecursively(child);
            }
        }
        return file.delete();
    }
}
