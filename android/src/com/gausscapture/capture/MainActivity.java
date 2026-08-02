package com.gausscapture.capture;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.Context;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.graphics.SurfaceTexture;
import android.hardware.SensorManager;
import android.hardware.camera2.CameraManager;
import android.net.Uri;
import android.provider.Settings;
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
import android.view.KeyEvent;
import android.view.View;
import android.view.WindowManager;
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
    private TextView timerTargetView;
    private View perimeterView;
    /** Which quarters of the target have already been marked with a haptic tick. */
    private int pacingTicked;
    private TextView techView;
    private LinearLayout hudView;
    private ProgressBar progressView;
    private Button recordButton;
    private Button presetButton;
    private Button takesButton;

    private CameraController camera;
    private SensorLogger sensors;
    private final Clocks clocks = new Clocks();
    /** True from the moment record is tapped until the session is configured or has failed. */
    private volatile boolean starting;
    private Storage storage;
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
            timerView.setText(String.format(Locale.US, "%d:%02d", seconds / 60, seconds % 60));
            int permille = (int) Math.min(1000, elapsedMs * 1000 / (preset.targetSeconds * 1000L));
            progressView.setProgress(permille);

            // Pacing ticks answer "am I done yet" with no glance at all, which is the point when
            // the operator's gaze belongs on the subject.
            int quarter = Math.min(4, permille / 250);
            while (pacingTicked < quarter) {
                pacingTicked++;
                buzz(pacingTicked == 4 ? 60 : 15);
            }
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
        timerTargetView = findViewById(R.id.timerTarget);
        perimeterView = findViewById(R.id.perimeter);
        techView = findViewById(R.id.tech);
        hudView = findViewById(R.id.hud);
        progressView = findViewById(R.id.progress);
        recordButton = findViewById(R.id.record);
        presetButton = findViewById(R.id.presetButton);
        takesButton = findViewById(R.id.takesButton);

        sensors = new SensorLogger(
                (SensorManager) getSystemService(Context.SENSOR_SERVICE), clocks);
        storage = Storage.open(this);
        offerSharedStorage();
        camera = new CameraController(this, previewView,
                (CameraManager) getSystemService(Context.CAMERA_SERVICE), this);

        recordButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onRecordTapped();
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
        // The grant can be given or revoked while the app is backgrounded, so the location is
        // chosen again here rather than once at startup.
        storage = Storage.open(this);
        backgroundThread = new HandlerThread("GaussCaptureCamera");
        backgroundThread.start();
        backgroundHandler = new Handler(backgroundThread.getLooper());
        camera.setHandler(backgroundHandler);

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
            perimeterView.setBackgroundResource(R.drawable.perimeter_warn);
            // One buzz on entering the state, never while it persists. Repetition is how an alert
            // teaches the person to ignore it.
            buzz(40);
            coachVisible = true;
            coachShownAt = now;
        } else if (coachVisible && blur < BLUR_CLEAR_PIXELS
                && now - coachShownAt > BLUR_MIN_VISIBLE_MS) {
            coachView.setVisibility(View.INVISIBLE);
            perimeterView.setBackgroundResource(R.drawable.perimeter_recording);
            coachVisible = false;
        }
    }

    /**
     * The one entry point for starting and stopping, whatever triggered it.
     *
     * <p>The button is disabled for the whole start transition rather than only once recording is
     * under way. Session configuration takes 100-500 ms, and a second tap inside that window used
     * to open a second session directory and a second writer over the same imu.jsonl, dropping the
     * first unflushed -- reproducible with a double-tap (audit D07).
     */
    private void onRecordTapped() {
        if (starting) {
            return;
        }
        if (recording) {
            stopRecording();
        } else {
            starting = true;
            recordButton.setEnabled(false);
            startRecording();
        }
    }

    /**
     * Volume keys start and stop a take.
     *
     * <p>The largest one-handed win available for a few lines: during a take the priority inverts,
     * because starting no longer matters and stopping has to be trivial while walking with the
     * phone held out. Deliberately not tap-anywhere-to-stop -- a stray palm ending a
     * sixty-second walk is worse than a fumbled stop.
     */
    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        if (keyCode == KeyEvent.KEYCODE_VOLUME_DOWN || keyCode == KeyEvent.KEYCODE_VOLUME_UP) {
            if (event.getRepeatCount() == 0) {
                onRecordTapped();
            }
            return true;
        }
        return super.onKeyDown(keyCode, event);
    }

    @Override
    public boolean onKeyUp(int keyCode, KeyEvent event) {
        // Swallowed so the system volume panel never appears over the viewfinder.
        return keyCode == KeyEvent.KEYCODE_VOLUME_DOWN || keyCode == KeyEvent.KEYCODE_VOLUME_UP
                || super.onKeyUp(keyCode, event);
    }

    /**
     * Offer to make captures visible to a computer, once.
     *
     * <p>Without all-files access the app writes to its private directory, which One UI hides from
     * both the file manager and MTP -- so the phone can be plugged in and the takes simply will not
     * appear. Recording still works either way, so this asks rather than blocks, and asks only when
     * the answer would change something.
     */
    private void offerSharedStorage() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R || Storage.canUseSharedStorage()) {
            return;
        }
        new AlertDialog.Builder(this)
                .setTitle("Make takes visible on a computer")
                .setMessage("Without file access, takes are stored where macOS cannot see them "
                        + "and can only be copied with adb. Grant access to write them to "
                        + "Documents/GaussCapture instead.")
                .setPositiveButton("Grant", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        try {
                            Intent intent = new Intent(
                                    Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                                    Uri.parse("package:" + getPackageName()));
                            startActivity(intent);
                        } catch (Exception error) {
                            startActivity(new Intent(
                                    Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION));
                        }
                    }
                })
                .setNegativeButton("Use adb", null)
                .show();
    }

    // ---------------------------------------------------------------- recording

    private void startRecording() {
        sessionDir = storage.newSessionDir(preset.id);
        if (sessionDir == null) {
            onError("Cannot create a folder for this take");
            return;
        }

        // The routine is "record all the time", and the default display timeout is 30 seconds.
        // Without this the screen sleeps mid-walk, onPause fires and the take is truncated
        // (audit D04). Full brightness for the duration is the other half: it is roughly 1300 nits
        // on an S22, and no colour choice can compensate for a dim panel in sunlight.
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        WindowManager.LayoutParams attributes = getWindow().getAttributes();
        attributes.screenBrightness = 1.0f;
        getWindow().setAttributes(attributes);

        startRealtimeNanos = SystemClock.elapsedRealtimeNanos();
        startUptimeNanos = System.nanoTime();
        startWallMillis = System.currentTimeMillis();

        sensors.start(new File(sessionDir, "imu.jsonl"), startRealtimeNanos);

        camera.setRecorderProblem(new CameraController.RecorderProblem() {
            @Override
            public void onRecorderProblem(final String message, final boolean fatal) {
                uiHandler.post(new Runnable() {
                    @Override
                    public void run() {
                        if (!recording) {
                            return;
                        }
                        // Encoding has already stopped. Ending the take now keeps the file
                        // finalised and tells the operator, instead of counting up over a video
                        // that is no longer being written (audit D05).
                        onError(message);
                        stopRecording();
                    }
                });
            }
        });

        camera.startRecording(sessionDir, displayRotation(), new CameraController.RecordingCallback() {
            @Override
            public void onStarted() {
                uiHandler.post(new Runnable() {
                    @Override
                    public void run() {
                        recording = true;
                        starting = false;
                        pacingTicked = 0;
                        recordButton.setEnabled(true);
                        perimeterView.setBackgroundResource(R.drawable.perimeter_recording);
                        perimeterView.setVisibility(View.VISIBLE);
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
            public void onFailed(final String message) {
                sensors.stop();
                deleteRecursively(sessionDir);
                uiHandler.post(new Runnable() {
                    @Override
                    public void run() {
                        starting = false;
                        recordButton.setEnabled(true);
                        onError(message);
                    }
                });
            }
        });
    }

    private void stopRecording() {
        recording = false;
        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        WindowManager.LayoutParams attributes = getWindow().getAttributes();
        attributes.screenBrightness = WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE;
        getWindow().setAttributes(attributes);
        uiHandler.removeCallbacks(tick);
        boolean ok = camera.stopRecording();
        sensors.stop();
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
                perimeterView.setVisibility(View.INVISIBLE);
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

            // These used to be hardcoded true (audit D11). A Samsung HAL routinely honours such
            // requests only partially, and a manifest that asserts what it did not check is worse
            // than one that admits it does not know: downstream code trusts it either way. What is
            // recorded now is what was *requested*; frames.jsonl carries the per-frame 3A and lens
            // state that says what actually happened.
            JSONObject settings = new JSONObject();
            settings.put("exposure_lock_requested", true);
            settings.put("white_balance_lock_requested", true);
            settings.put("focus_lock_requested", true);
            settings.put("stabilisation_disable_requested", true);
            settings.put("verified_per_frame_in", "frames.jsonl");
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

            // A gap that is known about is a limitation; a gap that is not is a wrong answer.
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
        timerView.setText("0:00");
        timerTargetView.setText(String.format(Locale.US, "of %d:%02d",
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
        File root = storage.root();
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
