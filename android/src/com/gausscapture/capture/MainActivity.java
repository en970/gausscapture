package com.gausscapture.capture;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.graphics.SurfaceTexture;
import android.hardware.SensorManager;
import android.hardware.camera2.CameraManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.os.VibratorManager;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONObject;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.UUID;

/**
 * Capture screen.
 *
 * <p>The operator is holding a phone at arm's length and walking in a circle. Everything here
 * follows from that: what to shoot is stated in words at the top, elapsed time is shown against
 * the take's target so nobody has to count, controls are 48dp and spaced, and the only thing that
 * ever interrupts is a warning that the phone is moving fast enough to blur the frame.
 *
 * <p>Technical readouts are deliberately marginal. During a take they are noise; the useful
 * information is "am I moving too fast" and "am I done yet".
 */
public class MainActivity extends Activity implements CameraController.Listener {

    private static final int PERMISSION_REQUEST = 1;

    /**
     * Blur, in pixels, above which the capture is warned about.
     *
     * <p>Around one pixel of smear is invisible and around three is clearly soft; two is a
     * defensible line that warns before a pass is ruined without nagging during normal walking.
     * It is a display threshold on a physically computed quantity, not a quality score.
     */
    private static final float BLUR_WARN_PIXELS = 2.0f;
    /** Hysteresis, so the banner does not flicker at the boundary. */
    private static final float BLUR_CLEAR_PIXELS = 1.4f;
    private static final long BLUR_MIN_VISIBLE_MS = 900;

    private TextureView previewView;
    private TextView presetNameView;
    private TextView presetHintView;
    private TextView coachView;
    private TextView timerView;
    private TextView techView;
    private LinearLayout hudView;
    private ProgressBar progressView;
    private Button recordButton;
    private Button presetButton;
    private Button takesButton;

    private CameraController camera;
    private SensorLogger sensors;
    private HandlerThread backgroundThread;
    private Handler backgroundHandler;
    private final Handler uiHandler = new Handler();

    private Preset preset = Preset.ALL[0];
    private volatile boolean recording;
    private File sessionDir;
    private long startRealtimeNanos;
    private long startUptimeNanos;
    private long startWallMillis;
    private long coachShownAt;
    private boolean coachVisible;

    private final Runnable tick = new Runnable() {
        @Override
        public void run() {
            if (!recording) {
                return;
            }
            long elapsedMs =
                    (SystemClock.elapsedRealtimeNanos() - startRealtimeNanos) / 1_000_000L;
            int seconds = (int) (elapsedMs / 1000);
            timerView.setText(String.format(Locale.US, "%d:%02d / %d:%02d",
                    seconds / 60, seconds % 60,
                    preset.targetSeconds / 60, preset.targetSeconds % 60));
            progressView.setProgress(
                    (int) Math.min(1000, elapsedMs * 1000 / (preset.targetSeconds * 1000L)));
            updateCoach();
            uiHandler.postDelayed(this, 200);
        }
    };

    // ---------------------------------------------------------------- lifecycle

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        previewView = findViewById(R.id.preview);
        presetNameView = findViewById(R.id.presetName);
        presetHintView = findViewById(R.id.presetHint);
        coachView = findViewById(R.id.coach);
        timerView = findViewById(R.id.timer);
        techView = findViewById(R.id.tech);
        hudView = findViewById(R.id.hud);
        progressView = findViewById(R.id.progress);
        recordButton = findViewById(R.id.record);
        presetButton = findViewById(R.id.presetButton);
        takesButton = findViewById(R.id.takesButton);

        sensors = new SensorLogger((SensorManager) getSystemService(Context.SENSOR_SERVICE));
        camera = new CameraController(this, previewView,
                (CameraManager) getSystemService(Context.CAMERA_SERVICE), this);

        recordButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (recording) {
                    stopRecording();
                } else {
                    startRecording();
                }
            }
        });
        presetButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                showPresetChooser();
            }
        });
        takesButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                showTakes();
            }
        });

        showPreset(preset);
        refreshTakesButton();

        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, PERMISSION_REQUEST);
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        backgroundThread = new HandlerThread("GaussCaptureCamera");
        backgroundThread.start();
        backgroundHandler = new Handler(backgroundThread.getLooper());
        camera.setHandler(backgroundHandler);
        sensors.listen(backgroundHandler);

        if (previewView.isAvailable()) {
            camera.open(displayRotation());
        } else {
            previewView.setSurfaceTextureListener(new TextureView.SurfaceTextureListener() {
                @Override
                public void onSurfaceTextureAvailable(SurfaceTexture s, int w, int h) {
                    camera.open(displayRotation());
                }

                @Override
                public void onSurfaceTextureSizeChanged(SurfaceTexture s, int w, int h) {
                    camera.applyTransform(displayRotation());
                }

                @Override
                public boolean onSurfaceTextureDestroyed(SurfaceTexture s) {
                    return true;
                }

                @Override
                public void onSurfaceTextureUpdated(SurfaceTexture s) { }
            });
        }
    }

    @Override
    protected void onPause() {
        if (recording) {
            stopRecording();
        }
        sensors.stopListening();
        camera.close();
        if (backgroundThread != null) {
            backgroundThread.quitSafely();
            try {
                backgroundThread.join();
            } catch (InterruptedException ignored) {
                Thread.currentThread().interrupt();
            }
            backgroundThread = null;
            backgroundHandler = null;
        }
        super.onPause();
    }

    /**
     * The app follows the phone rather than forcing a grip.
     *
     * <p>The pipeline does not care whether a capture is portrait or landscape; what matters is
     * that the orientation is constant within a take. So rotation is free while idle and frozen
     * during a recording, because the encoder's frame geometry is fixed when the session is
     * configured and cannot change underneath it.
     */
    @Override
    public void onConfigurationChanged(android.content.res.Configuration config) {
        super.onConfigurationChanged(config);
        camera.applyTransform(displayRotation());
    }

    private int displayRotation() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && getDisplay() != null) {
            return getDisplay().getRotation();
        }
        return getWindowManager().getDefaultDisplay().getRotation();
    }

    // ---------------------------------------------------------------- camera callbacks

    @Override
    public void onReady(final String summary) {
        uiHandler.post(new Runnable() {
            @Override
            public void run() {
                techView.setText(summary);
            }
        });
    }

    @Override
    public void onError(final String message) {
        uiHandler.post(new Runnable() {
            @Override
            public void run() {
                techView.setText(message);
                Toast.makeText(MainActivity.this, message, Toast.LENGTH_LONG).show();
            }
        });
    }

    @Override
    public void onFrameMetadata(long exposureNanos, int iso) {
        // Consumed by updateCoach on the UI tick; nothing to do per frame.
    }

    // ---------------------------------------------------------------- guidance

    /**
     * Shows a warning when the phone is rotating fast enough to smear the frame.
     *
     * <p>The quantity is computed, not guessed: a camera turning at omega radians per second
     * during an exposure of t seconds sweeps the image by omega*t radians, which at a focal length
     * of f pixels is omega*t*f pixels of blur. All three come from the device -- gyroscope,
     * CaptureResult, and reported intrinsics -- so this is geometry rather than a heuristic, which
     * is what makes it honest to show while the project's own quality score is still unvalidated.
     */
    private void updateCoach() {
        float focalPixels = camera.focalPixels();
        if (focalPixels <= 0) {
            return;  // Device does not report intrinsics; no defensible number to show.
        }
        float exposureSeconds = camera.lastExposureNanos() / 1e9f;
        float blur = SensorLogger.blurPixels(sensors.angularRate(), exposureSeconds, focalPixels);

        long now = SystemClock.uptimeMillis();
        if (!coachVisible && blur > BLUR_WARN_PIXELS) {
            coachView.setText("Slow down");
            coachView.setVisibility(View.VISIBLE);
            coachVisible = true;
            coachShownAt = now;
        } else if (coachVisible && blur < BLUR_CLEAR_PIXELS
                && now - coachShownAt > BLUR_MIN_VISIBLE_MS) {
            coachView.setVisibility(View.INVISIBLE);
            coachVisible = false;
        }
    }

    // ---------------------------------------------------------------- recording

    private void startRecording() {
        sessionDir = new File(getExternalFilesDir(null), preset.id + "_"
                + new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date()));
        if (!sessionDir.mkdirs() && !sessionDir.isDirectory()) {
            onError("Cannot create a folder for this take");
            return;
        }

        startRealtimeNanos = SystemClock.elapsedRealtimeNanos();
        startUptimeNanos = System.nanoTime();
        startWallMillis = System.currentTimeMillis();

        try {
            sensors.start(new File(sessionDir, "imu.jsonl"), startRealtimeNanos);
        } catch (Exception e) {
            onError("Cannot write sensor data: " + e.getMessage());
            return;
        }

        camera.startRecording(sessionDir, displayRotation(), new CameraController.RecordingCallback() {
            @Override
            public void onStarted() {
                uiHandler.post(new Runnable() {
                    @Override
                    public void run() {
                        recording = true;
                        buzz(30);
                        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LOCKED);
                        recordButton.setBackgroundResource(R.drawable.btn_stop);
                        recordButton.setContentDescription("Stop recording");
                        hudView.setVisibility(View.VISIBLE);
                        presetButton.setEnabled(false);
                        takesButton.setEnabled(false);
                        uiHandler.post(tick);
                    }
                });
            }

            @Override
            public void onFailed(String message) {
                sensors.closeFile();
                deleteRecursively(sessionDir);
                onError(message);
            }
        });
    }

    private void stopRecording() {
        recording = false;
        uiHandler.removeCallbacks(tick);
        boolean ok = camera.stopRecording();
        sensors.closeFile();
        buzz(20);

        String message;
        if (!ok) {
            deleteRecursively(sessionDir);
            message = "Take was too short, discarded";
        } else {
            writeManifest();
            long seconds = (SystemClock.elapsedRealtimeNanos() - startRealtimeNanos) / 1_000_000_000L;
            message = String.format(Locale.US, "Saved %s · %ds · %d frames",
                    preset.id, seconds, camera.frameCount());
        }

        final String toast = message;
        uiHandler.post(new Runnable() {
            @Override
            public void run() {
                setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_FULL_USER);
                recordButton.setBackgroundResource(R.drawable.btn_record);
                recordButton.setContentDescription("Start recording");
                hudView.setVisibility(View.INVISIBLE);
                coachView.setVisibility(View.INVISIBLE);
                coachVisible = false;
                presetButton.setEnabled(true);
                takesButton.setEnabled(true);
                progressView.setProgress(0);
                refreshTakesButton();
                Toast.makeText(MainActivity.this, toast, Toast.LENGTH_SHORT).show();
            }
        });

        camera.startPreview(displayRotation());
    }

    /**
     * Writes what the desktop pipeline needs to interpret the recording.
     *
     * <p>The timestamp source matters most. If it is not REALTIME, video and IMU timestamps live
     * in different clocks and alignment needs the offset recorded here; saying so explicitly is
     * the difference between data that can be used and data that merely looks usable.
     */
    private void writeManifest() {
        try {
            JSONObject manifest = new JSONObject();
            manifest.put("capturepack_version", "0.1");
            manifest.put("session_id", UUID.randomUUID().toString());
            manifest.put("session_name", sessionDir.getName());
            manifest.put("preset", preset.id);
            manifest.put("preset_target_seconds", preset.targetSeconds);
            manifest.put("expected_to_fail", preset.expectedToFail);
            manifest.put("capture_type", preset.expectedToFail && "E_subject".equals(preset.id)
                    ? "dynamic_subject" : "static_scene");
            manifest.put("created_at", isoTime(startWallMillis));
            manifest.put("app", "gausscapture-android/0.2");

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
            manifest.put("video", video);

            JSONObject settings = new JSONObject();
            settings.put("exposure_locked", true);
            settings.put("white_balance_locked", true);
            settings.put("focus_locked", true);
            settings.put("stabilisation_disabled", true);
            settings.put("lens_switching_disabled", true);
            settings.put("storage_mode", "phone");
            manifest.put("capture_settings", settings);

            manifest.put("time_base", "nanoseconds_elapsed_realtime");
            JSONObject clocks = new JSONObject();
            clocks.put("recording_start_elapsed_realtime_ns", startRealtimeNanos);
            clocks.put("recording_start_uptime_ns", startUptimeNanos);
            clocks.put("recording_start_wall_ms", startWallMillis);
            clocks.put("camera_timestamp_source",
                    camera.timestampsAligned() ? "REALTIME" : "UNKNOWN");
            clocks.put("camera_imu_same_clock", camera.timestampsAligned());
            manifest.put("clocks", clocks);

            JSONObject files = new JSONObject();
            files.put("intrinsics", "intrinsics.json");
            files.put("imu", "imu.jsonl");
            files.put("frames", "frames.jsonl");
            files.put("poses", JSONObject.NULL);
            files.put("light", JSONObject.NULL);
            manifest.put("metadata_files", files);
            manifest.put("imu_samples", sensors.sampleCount());

            write(new File(sessionDir, "manifest.json"), manifest.toString(2));
            write(new File(sessionDir, "intrinsics.json"), camera.intrinsics().toString(2));
        } catch (Exception e) {
            onError("Could not write metadata: " + e.getMessage());
        }
    }

    // ---------------------------------------------------------------- preset & takes

    private void showPreset(Preset chosen) {
        preset = chosen;
        presetNameView.setText(chosen.title);
        presetHintView.setText(chosen.hint);
        presetButton.setText(chosen.shortLabel());
        timerView.setText(String.format(Locale.US, "0:00 / %d:%02d",
                chosen.targetSeconds / 60, chosen.targetSeconds % 60));
    }

    private void showPresetChooser() {
        new AlertDialog.Builder(this)
                .setTitle("Which shot?")
                .setItems(Preset.menuLabels(), new android.content.DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(android.content.DialogInterface dialog, int which) {
                        showPreset(Preset.ALL[which]);
                    }
                })
                .show();
    }

    private List<File> takes() {
        File root = getExternalFilesDir(null);
        File[] children = root == null ? null : root.listFiles();
        List<File> result = new ArrayList<>();
        if (children != null) {
            for (File child : children) {
                if (child.isDirectory() && new File(child, "manifest.json").exists()) {
                    result.add(child);
                }
            }
        }
        java.util.Collections.sort(result, new Comparator<File>() {
            @Override
            public int compare(File a, File b) {
                return a.getName().compareTo(b.getName());
            }
        });
        return result;
    }

    private void refreshTakesButton() {
        int count = takes().size();
        takesButton.setText(count == 0 ? "Takes" : "Takes · " + count);
    }

    private void showTakes() {
        final List<File> recorded = takes();
        if (recorded.isEmpty()) {
            // An empty state should say what to do next, not just be blank.
            new AlertDialog.Builder(this)
                    .setTitle("No takes yet")
                    .setMessage("Pick a shot, then press the red button. "
                            + "Start with A and work down the list.")
                    .setPositiveButton("OK", null)
                    .show();
            return;
        }

        String[] labels = new String[recorded.size()];
        for (int i = 0; i < recorded.size(); i++) {
            File take = recorded.get(i);
            labels[i] = take.getName() + "\n" + humanSize(directorySize(take));
        }

        new AlertDialog.Builder(this)
                .setTitle(recorded.size() + " take" + (recorded.size() == 1 ? "" : "s"))
                .setItems(labels, new android.content.DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(android.content.DialogInterface dialog, int which) {
                        confirmDelete(recorded.get(which));
                    }
                })
                .setPositiveButton("Done", null)
                .show();
    }

    /** Deleting a take destroys minutes of walking, so it is always confirmed. */
    private void confirmDelete(final File take) {
        new AlertDialog.Builder(this)
                .setTitle("Delete this take?")
                .setMessage(take.getName() + "\nThis cannot be undone.")
                .setNegativeButton("Keep", null)
                .setPositiveButton("Delete", new android.content.DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(android.content.DialogInterface dialog, int which) {
                        deleteRecursively(take);
                        refreshTakesButton();
                        Toast.makeText(MainActivity.this, "Deleted", Toast.LENGTH_SHORT).show();
                    }
                })
                .show();
    }

    // ---------------------------------------------------------------- helpers

    /** Tactile confirmation for start and stop, where a glance at the screen is inconvenient. */
    private void buzz(long millis) {
        Vibrator vibrator;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            VibratorManager vm = (VibratorManager) getSystemService(Context.VIBRATOR_MANAGER_SERVICE);
            vibrator = vm == null ? null : vm.getDefaultVibrator();
        } else {
            vibrator = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
        }
        if (vibrator != null && vibrator.hasVibrator()) {
            vibrator.vibrate(VibrationEffect.createOneShot(millis,
                    VibrationEffect.DEFAULT_AMPLITUDE));
        }
    }

    private static long directorySize(File dir) {
        long total = 0;
        File[] children = dir.listFiles();
        if (children != null) {
            for (File child : children) {
                total += child.isDirectory() ? directorySize(child) : child.length();
            }
        }
        return total;
    }

    private static String humanSize(long bytes) {
        if (bytes < 1024) {
            return bytes + " B";
        }
        if (bytes < 1024 * 1024) {
            return (bytes / 1024) + " KB";
        }
        return String.format(Locale.US, "%.0f MB", bytes / 1024.0 / 1024.0);
    }

    private static String isoTime(long millis) {
        SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US);
        format.setTimeZone(java.util.TimeZone.getTimeZone("UTC"));
        return format.format(new Date(millis));
    }

    private static void write(File file, String content) throws Exception {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
            writer.write(content);
        }
    }

    private static void deleteRecursively(File file) {
        if (file == null || !file.exists()) {
            return;
        }
        File[] children = file.listFiles();
        if (children != null) {
            for (File child : children) {
                deleteRecursively(child);
            }
        }
        file.delete();
    }

    @Override
    public void onRequestPermissionsResult(int code, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(code, permissions, results);
        if (code == PERMISSION_REQUEST && previewView.isAvailable()) {
            camera.open(displayRotation());
        }
    }
}
