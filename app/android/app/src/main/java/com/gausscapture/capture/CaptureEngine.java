package com.gausscapture.capture;

import android.app.Activity;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.view.TextureView;
import android.hardware.SensorManager;
import android.hardware.camera2.CameraManager;
import android.content.Context;

import org.json.JSONArray;
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
 *
 * <p>It also owns the <em>script</em>, for the presets that have one. A scripted take is several
 * phases inside one recording, and which phase a frame belongs to is not something the desktop can
 * work out afterwards from timestamps — so the engine drives the phase machine, tags every frame,
 * and writes the frame bounds into the manifest. The two moments that cannot be reconstructed
 * later are here for the same reason: when the phone was measured to have come to rest, and when
 * it stopped being at rest.
 */
public final class CaptureEngine implements SensorLogger.MotionSink {

    /** Bumped whenever the recorded schema changes, so a pack can be traced to what produced it. */
    static final String APP_VERSION = "gausscapture-flutter/0.4";

    /**
     * The capturepack schema this app writes.
     *
     * <p>0.3 adds the blocks a scripted take needs — {@code protocol}, {@code arc},
     * {@code fixed_camera}, {@code photometric}, {@code camera} — and a {@code phase} on every
     * frame row. They are additions: a pack from preset A carries the ones that apply to it and
     * omits the rest.
     */
    static final String CAPTUREPACK_VERSION = "0.3";

    /** How often the script is advanced. Twenty times a second is well under one frame. */
    private static final long TICK_MS = 50;

    /** How often the rest detector's own view of the world is written down. */
    private static final int STILLNESS_EVERY_TICKS = 4;

    public interface Events {
        /** A take ended for a reason the operator did not choose. */
        void onProblem(String message);
        /** A take started or stopped; the interface should re-read state. */
        void onStateChanged();
        /**
         * A scripted take ended by itself.
         *
         * <p>Not the same as a problem, and not the same as the operator pressing stop. The
         * interface has to hear about it because it did not ask for it: the dynamic phase
         * auto-stops precisely so that nobody reaches for the phone, which means nothing on the
         * Flutter side knows the take is over unless it is told.
         */
        void onFinished(JSONObject manifest, String reason, boolean aborted);
    }

    private final Activity activity;
    private final Clocks clocks = new Clocks();
    private final SensorLogger sensors;
    private final CameraManager cameraManager;
    private final Handler ticker = new Handler(Looper.getMainLooper());
    private final Stillness stillness = new Stillness();
    private final Arc arc = new Arc();
    private final PhaseMachine.Motion motion = new PhaseMachine.Motion();
    private Storage storage;
    private CameraController camera;
    private Events events;

    private PhaseMachine machine;
    private JsonlWriter eventWriter;
    private JsonlWriter stillnessWriter;
    private final StringBuilder eventLine = new StringBuilder(160);
    private final StringBuilder stillnessLine = new StringBuilder(192);
    private int ticks;
    private boolean restMarked;
    private String finishedReason;
    private boolean finishedByAbort;

    /** Quarter turns the operator has added to the preview. Display only; see nudgePreviewTurn. */
    private int previewTurn;

    /**
     * Where the chosen protocol and the preview orientation are remembered.
     *
     * <p>The protocol was not remembered at all before. The operator picked C, the process was
     * recreated for any of the ordinary reasons, and the next take was silently written as A --
     * and a mislabelled capture is not a weaker data point, it is a wrong one. Preset's own
     * documentation makes that argument; this is the case it did not defend against (audit D26).
     */
    private static final String PREFERENCES = "gausscapture";
    private static final String KEY_PRESET = "preset";
    private static final String KEY_PREVIEW_TURN = "preview_turn";

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
        previewTurn = activity.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
                .getInt(KEY_PREVIEW_TURN, 0);
    }

    /**
     * Turn the preview a quarter and remember it.
     *
     * <p>Which way the preview has to be turned depends on how the platform's compositor
     * presents the surface, and that is not something this process can read: the derived
     * angle was tried at every quarter and each one left the image on its side on this
     * handset. So the operator sets it once, it is kept across launches, and it is stored
     * in the manifest so a take carries the orientation it was framed at.
     *
     * <p>This changes what is displayed and nothing else. The recording, its
     * orientation hint and the intrinsics are untouched.
     */
    public int nudgePreviewTurn() {
        previewTurn = (previewTurn + 1) % 4;
        activity.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
                .edit().putInt(KEY_PREVIEW_TURN, previewTurn).apply();
        if (camera != null) {
            camera.setPreviewTurn(previewTurn);
            camera.applyTransform(displayRotation());
        }
        return previewTurn;
    }

    public int previewTurn() {
        return previewTurn;
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
                boolean lensChanged = !candidate.lensFacing.equals(preset.lensFacing);
                preset = candidate;
                activity.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
                        .edit()
                        .putString(KEY_PRESET, id)
                        .apply();
                // The 4D protocol films the operator's own face, and which lens that is has to be
                // decided before the device is opened. Switching presets therefore reopens the
                // camera rather than leaving the preview pointed at the wrong side of the phone.
                if (lensChanged && camera != null && !recording) {
                    camera.close();
                    camera.setLensFacing(preset.lensFacing);
                    camera.open(displayRotation());
                }
                return;
            }
        }
    }

    // ------------------------------------------------------------------ camera

    /** Bind to the preview surface Flutter is hosting, and open the camera behind it. */
    public void attachPreview(TextureView view, CameraController.Listener listener) {
        // Flutter recreates the platform view on rotation and on returning from the
        // background, so this runs more than once per session. Each controller owns a
        // callback thread; dropping the previous one on the floor would leak a thread
        // per attach, and the abandoned one would still hold the camera device.
        if (camera != null) {
            camera.dispose();
        }
        camera = new CameraController(activity, view, cameraManager, listener);
        camera.setPreviewTurn(previewTurn);
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
            camera.setLensFacing(preset.lensFacing);
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

        ticks = 0;
        restMarked = false;
        finishedReason = null;
        finishedByAbort = false;
        stillness.reset();
        machine = preset.isScripted() ? new PhaseMachine(preset) : null;
        camera.setPhase(preset.phases[0].id);
        camera.setAnomalyListener(anomalies);

        eventWriter = new JsonlWriter(new File(sessionDir, "events.jsonl"));
        if (machine != null) {
            stillnessWriter = new JsonlWriter(new File(sessionDir, "stillness.jsonl"));
        }
        // Guidance reads the same samples that are being written, rather than subscribing to the
        // sensors a second time.
        sensors.setMotionSink(this);
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
                if (machine != null) {
                    ticker.post(tick);
                }
                if (events != null) {
                    events.onStateChanged();
                }
            }

            @Override
            public void onFailed(String message) {
                sensors.setMotionSink(null);
                sensors.stop();
                closeSidecars();
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
        ticker.removeCallbacks(tick);

        boolean encoded = camera.stopRecording();
        sensors.setMotionSink(null);
        sensors.stop();
        closeSidecars();

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

    // ------------------------------------------------------------------ script

    /**
     * Advance the script, and do the two things that cannot be reconstructed afterwards.
     *
     * <p>The first is marking the rest window: the run of frames the camera was measured to be at
     * rest for, at the pose the dynamic phase is about to be shot from. Those frames are static
     * scene content from the final pose, which is what makes them the tier-one evidence for that
     * pose on the desktop. A timestamp written down later cannot identify them, because "at rest"
     * is a measurement and not a moment.
     *
     * <p>The second is auto-stopping. Reaching for the phone to press stop moves it, and the
     * frames that would be corrupted are the last ones — the ones closest to the motion the take
     * exists to record.
     */
    private final Runnable tick = new Runnable() {
        @Override
        public void run() {
            PhaseMachine current = machine;
            if (!recording || current == null) {
                return;
            }
            long elapsed = elapsedMillis();
            motion.atRest = stillness.atRest();
            motion.restForMillis = stillness.restForMillis();
            motion.sweptDeg = arc.sweptDeg();
            motion.orbitLikely = arc.orbitLikely();

            String before = current.phaseId();
            current.update(elapsed, motion);
            String after = current.phaseId();
            if (!before.equals(after)) {
                onPhaseChanged(before, after, elapsed);
            }
            if (current.restConfirmed() && !restMarked) {
                restMarked = true;
                // From here on, any change in the gravity direction is a camera that moved.
                stillness.markReference();
                camera.markRestWindowStart();
                event("rest_confirmed", "\"rest_for_ms\":" + stillness.restForMillis());
            }
            if (ticks % STILLNESS_EVERY_TICKS == 0) {
                writeStillnessRow(current.phaseId());
            }
            ticks++;

            if (current.finished()) {
                finish(current.finishedReason(), current.aborted());
                return;
            }
            ticker.postDelayed(this, TICK_MS);
        }
    };

    private void onPhaseChanged(String from, String to, long elapsed) {
        camera.setPhase(to);
        if ("arc".equals(to)) {
            arc.begin();
        }
        if ("arc".equals(from)) {
            arc.end();
            event("arc_finished", "\"swept_deg\":" + arc.sweptDeg()
                    + ",\"return_error_deg\":" + arc.returnErrorDeg());
        }
        if (machine != null && machine.phase().subjectMoves) {
            camera.markRestWindowEnd();
        }
        event("phase", "\"from\":\"" + from + "\",\"to\":\"" + to
                + "\",\"elapsed_ms\":" + elapsed);
    }

    /** A scripted take ending by itself, either because it finished or because it cannot go on. */
    private void finish(String reason, boolean aborted) {
        finishedReason = reason;
        finishedByAbort = aborted;
        event("auto_stop", "\"reason\":\"" + reason + "\",\"aborted\":" + aborted);
        JSONObject manifest = stop();
        if (events != null) {
            events.onFinished(manifest, reason, aborted);
        }
    }

    // ------------------------------------------------------------ motion sink

    @Override
    public void onGyro(long tNs, float x, float y, float z) {
        stillness.onGyro(tNs, x, y, z);
        arc.onGyro(tNs, x, y, z);
    }

    @Override
    public void onAccel(long tNs, float x, float y, float z) {
        stillness.onAccel(tNs, x, y, z);
        arc.onAccel(tNs, x, y, z);
    }

    @Override
    public void onRotation(float[] quaternion) {
        arc.onRotation(quaternion);
    }

    // ------------------------------------------------------------- side files

    /** Anything that invalidates something the manifest would otherwise assert. */
    private final CameraController.Anomaly anomalies = new CameraController.Anomaly() {
        @Override
        public void onAnomaly(String event, String detail, long tNs) {
            JsonlWriter writer = eventWriter;
            if (writer == null) {
                return;
            }
            synchronized (eventLine) {
                eventLine.setLength(0);
                eventLine.append("{\"t_ns\":").append(tNs)
                        .append(",\"event\":\"").append(event)
                        .append("\",\"detail\":\"").append(detail.replace('"', '\''))
                        .append("\"}");
                writer.append(eventLine);
            }
        }
    };

    /** One line of {@code events.jsonl}. {@code fields} is raw JSON, already escaped. */
    private void event(String name, String fields) {
        JsonlWriter writer = eventWriter;
        if (writer == null) {
            return;
        }
        synchronized (eventLine) {
            eventLine.setLength(0);
            eventLine.append("{\"t_ns\":").append(SystemClock.elapsedRealtimeNanos())
                    .append(",\"event\":\"").append(name).append('"');
            if (fields != null && !fields.isEmpty()) {
                eventLine.append(',').append(fields);
            }
            eventLine.append('}');
            writer.append(eventLine);
        }
    }

    private void writeStillnessRow(String phase) {
        JsonlWriter writer = stillnessWriter;
        if (writer == null) {
            return;
        }
        synchronized (stillnessLine) {
            stillness.appendRow(stillnessLine, SystemClock.elapsedRealtimeNanos(), phase);
            writer.append(stillnessLine);
        }
    }

    private void closeSidecars() {
        JsonlWriter events = eventWriter;
        eventWriter = null;
        if (events != null) {
            events.close();
        }
        JsonlWriter rest = stillnessWriter;
        stillnessWriter = null;
        if (rest != null) {
            rest.close();
        }
    }

    // -------------------------------------------------------- live script state

    /** What the interface needs to render the script. Neutral values when there is no script. */
    public String phaseId() {
        PhaseMachine current = machine;
        return current == null ? preset.phases[0].id : current.phaseId();
    }

    public int phaseIndex() {
        PhaseMachine current = machine;
        return current == null ? 0 : current.phaseIndex();
    }

    public int phaseCount() {
        return preset.phases.length;
    }

    public String phaseCue() {
        PhaseMachine current = machine;
        return current == null ? "" : current.cue();
    }

    public String phaseDetail() {
        PhaseMachine current = machine;
        return current == null ? preset.hint : current.phase().detail;
    }

    public String phaseWarning() {
        PhaseMachine current = machine;
        return current == null ? null : current.warning();
    }

    public long phaseElapsedMs() {
        PhaseMachine current = machine;
        return current == null ? elapsedMillis() : current.phaseElapsedMs();
    }

    public long phaseTargetMs() {
        PhaseMachine current = machine;
        return current == null ? preset.targetSeconds * 1000L : current.phaseTargetMs();
    }

    public long countdownMillis() {
        PhaseMachine current = machine;
        return current == null ? -1 : current.countdownMillis();
    }

    public int phaseTransitions() {
        PhaseMachine current = machine;
        return current == null ? 0 : current.transitions();
    }

    public boolean atRest() {
        return stillness.atRest();
    }

    /** How long the phone has been continuously still, which is what the countdown waits on. */
    public long restForMillis() {
        return stillness.restForMillis();
    }

    /** True once rest was confirmed and then lost. A latch: the take can no longer claim it. */
    public boolean restDisturbed() {
        return stillness.disturbed();
    }

    /** Whether this preset is a script with phases, or one undivided stretch of recording. */
    public boolean isScripted() {
        return preset.isScripted();
    }

    /** The phase ids in order, so the interface can draw the script before it starts. */
    public List<String> phaseIds() {
        List<String> out = new ArrayList<>(preset.phases.length);
        for (Preset.Phase phase : preset.phases) {
            out.add(phase.id);
        }
        return out;
    }

    public float arcSweptDeg() {
        return arc.sweptDeg();
    }

    public int arcTargetDeg() {
        return preset.arcTargetDeg;
    }

    /** −1 left, +1 right, 0 when both sides of the sweep have their share. */
    public int arcDirection() {
        return preset.arcTargetDeg > 0 ? arc.suggestedDirection(preset.arcTargetDeg) : 0;
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
            JSONArray spans = camera.phaseSpansJson();
            double durationSec = recordedSeconds(spans);
            double achievedFps = durationSec > 0 && camera.frameCount() > 1
                    ? (camera.frameCount() - 1) / durationSec : 0;

            JSONObject manifest = new JSONObject();
            manifest.put("capturepack_version", CAPTUREPACK_VERSION);
            manifest.put("session_id", UUID.randomUUID().toString());
            manifest.put("session_name", sessionDir.getName());
            manifest.put("preset", preset.id);
            manifest.put("preset_target_seconds", preset.targetSeconds);
            manifest.put("expected_to_fail", preset.expectedToFail);
            // The preset declares its own capture type. Deriving it from the preset id here is how
            // an F_bullet take used to be labelled static_scene, which sent the desktop down the
            // wrong pipeline for a recording whose whole point was the other one.
            //
            // Except when the script did not get that far. A take that aborted during the perch or
            // the sweep contains no footage of a moving subject in front of a stationary camera,
            // which is the only thing fixed_camera_4d means -- so labelling it that way would be a
            // claim about content that is not in the file. What was recorded is a hand-held sweep,
            // which is a static scene and is usable as one.
            String captureType = preset.captureType;
            if (preset.isScripted() && !hasPhase(spans, "hold")) {
                captureType = hasPhase(spans, "arc") ? "static_scene" : "unknown";
                manifest.put("capture_type_intended", preset.captureType);
            }
            manifest.put("capture_type", captureType);
            manifest.put("created_at", isoTime(startWallMillis));
            manifest.put("app", APP_VERSION);
            manifest.put("clean_stop", cleanStop);

            JSONObject device = new JSONObject();
            device.put("manufacturer", Build.MANUFACTURER);
            device.put("model", Build.MODEL);
            device.put("os", "Android " + Build.VERSION.RELEASE
                    + " (API " + Build.VERSION.SDK_INT + ")");
            device.put("app_version", APP_VERSION);
            device.put("camera_id", camera.cameraId());
            device.put("sensor_orientation", camera.sensorOrientation());
            manifest.put("device", device);

            JSONObject video = new JSONObject();
            video.put("main_file", "video.mp4");
            video.put("width", camera.videoSize().getWidth());
            video.put("height", camera.videoSize().getHeight());
            // Achieved, not requested. The encoder is asked for 30 and delivers what the thermal
            // and light budget allowed; a phase bound quoted in frames is meaningless against a
            // nominal rate the recording never ran at.
            video.put("fps", achievedFps > 0 ? round(achievedFps, 3) : JSONObject.NULL);
            video.put("fps_requested", 30);
            video.put("duration_sec", durationSec > 0 ? round(durationSec, 3) : JSONObject.NULL);
            video.put("codec", "h264");
            video.put("orientation_hint", camera.previewRotation(displayRotation()));
            video.put("frames_recorded", camera.frameCount());
            video.put("has_audio", false);
            video.put("bytes", new File(sessionDir, "video.mp4").length());
            manifest.put("video", video);

            JSONObject camera4d = new JSONObject();
            camera4d.put("lens_facing", camera.lensFacing());
            camera4d.put("logical_id", camera.cameraId());
            camera4d.put("physical_id", camera.activePhysicalId() == null
                    ? JSONObject.NULL : camera.activePhysicalId());
            camera4d.put("sensor_orientation_deg", camera.sensorOrientation());
            camera4d.put("zoom_ratio", 1.0);
            manifest.put("camera", camera4d);

            JSONObject photometric = camera.photometricSettings();
            photometric.put("screen_brightness", preset.screenBrightness < 0
                    ? JSONObject.NULL : (double) preset.screenBrightness);
            manifest.put("photometric", photometric);

            if (preset.isScripted()) {
                manifest.put("protocol", protocolBlock(spans, achievedFps));
            }
            if (preset.arcTargetDeg > 0) {
                manifest.put("arc", arcBlock());
            }
            if (preset.dynamicPhase() != null) {
                manifest.put("fixed_camera", fixedCameraBlock());
            }

            // Requested, not asserted. A Samsung HAL honours these only partially, and frames.jsonl
            // carries the per-frame 3A and lens state that says what actually happened (audit D11).
            JSONObject settings = new JSONObject();
            settings.put("exposure_lock_requested", true);
            settings.put("white_balance_lock_requested", true);
            settings.put("focus_lock_requested", true);
            settings.put("stabilisation_disable_requested", true);
            settings.put("verified_per_frame_in", "frames.jsonl");

            // ...and achieved, under the names capturepack.schema.json publishes. These are the
            // keys pack/manifest.py and telemetry/report.py read; writing only the `_requested`
            // spelling above left them absent, the manifest still validated because the schema
            // allows extra properties, and every take was reported as having no locks.
            JSONObject achieved = camera.achievedLocks();
            for (java.util.Iterator<String> keys = achieved.keys(); keys.hasNext(); ) {
                String key = keys.next();
                settings.put(key, achieved.get(key));
            }
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
            files.put("events", "events.jsonl");
            files.put("stillness", preset.isScripted() ? "stillness.jsonl" : JSONObject.NULL);
            files.put("poses", JSONObject.NULL);
            files.put("light", JSONObject.NULL);
            manifest.put("metadata_files", files);
            manifest.put("imu_samples", sensors.sampleCount());

            JSONObject health = new JSONObject();
            health.put("imu_lines_written", sensors.linesWritten());
            health.put("imu_lines_dropped", sensors.linesDropped());
            health.put("imu_rate_hz", sensors.measuredRateHz());
            health.put("frame_rows_written", camera.frameCount());
            health.put("frame_rows_dropped", camera.framesDropped());
            String imuFailure = sensors.failure();
            health.put("imu_write_error", imuFailure == null ? JSONObject.NULL : imuFailure);
            String frameFailure = camera.frameWriteError();
            health.put("frame_write_error", frameFailure == null ? JSONObject.NULL : frameFailure);
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

    /**
     * Where each phase begins and ends, in frames.
     *
     * <p>This is the block the whole scripted protocol exists to produce. {@code prep4d} splits the
     * recording on it, and there is no way to recover it afterwards: the boundary between the phone
     * settling and the subject starting to move is a measurement taken while it happened, and a
     * desktop looking at timestamps a week later can only guess at it.
     *
     * <p>Bounds are frame ordinals among the rows of {@code frames.jsonl}, which is the same
     * ordering as the video's samples, and both ends are inclusive. Phase ids are the schema's
     * enumeration; a preset with a single unscripted stretch has no protocol block at all rather
     * than one naming a phase the format does not define.
     */
    private JSONObject protocolBlock(JSONArray spans, double achievedFps) throws Exception {
        JSONObject protocol = new JSONObject();
        protocol.put("name", preset.id);
        // The phase machine counted the frames it actually received, so the bounds must be read
        // against the rate the recording achieved rather than the rate it asked for.
        protocol.put("fps", achievedFps > 0 ? round(achievedFps, 3) : JSONObject.NULL);

        JSONArray phases = new JSONArray();
        for (int i = 0; i < spans.length(); i++) {
            JSONObject span = spans.getJSONObject(i);
            String id = span.getString("id");
            if (!isDeclarablePhase(id)) {
                continue;
            }
            JSONObject phase = new JSONObject();
            phase.put("name", id);
            phase.put("start_frame", span.getInt("frame_first"));
            phase.put("end_frame", span.getInt("frame_last"));
            // Beyond the schema, and the only evidence that a frame was dropped inside a phase
            // rather than between two of them.
            phase.put("capture_frame_first", span.getLong("capture_frame_first"));
            phase.put("capture_frame_last", span.getLong("capture_frame_last"));
            phase.put("t_ns_first", span.getLong("t_ns_first"));
            phase.put("t_ns_last", span.getLong("t_ns_last"));
            phases.put(phase);
        }
        protocol.put("phases", phases);

        JSONObject rest = camera.restWindowJson();
        if (rest != null) {
            JSONObject window = new JSONObject();
            window.put("start_frame", rest.getInt("frame_first"));
            window.put("end_frame", rest.getInt("frame_last"));
            window.put("capture_frame_first", rest.getLong("capture_frame_first"));
            window.put("capture_frame_last", rest.getLong("capture_frame_last"));
            protocol.put("rest_window", window);
        }

        protocol.put("auto_stopped", finishedReason != null);
        protocol.put("aborted_reason", finishedByAbort ? finishedReason : JSONObject.NULL);
        protocol.put("phase_transitions", machine == null ? 0 : machine.transitions());
        return protocol;
    }

    /** Whether any frame was actually tagged with this phase. */
    private static boolean hasPhase(JSONArray spans, String id) throws Exception {
        for (int i = 0; i < spans.length(); i++) {
            if (id.equals(spans.getJSONObject(i).getString("id"))) {
                return true;
            }
        }
        return false;
    }

    /** Only the four phases the capturepack schema defines may be declared under that name. */
    private static boolean isDeclarablePhase(String id) {
        return "perch".equals(id) || "arc".equals(id)
                || "reseat".equals(id) || "hold".equals(id);
    }

    /**
     * What the phone's own sensors made of the sweep.
     *
     * <p>A cross-check and never the answer. The published view cone is measured offline from the
     * cameras structure-from-motion solved, and an integrated gyro drifts about seven degrees over
     * twelve seconds — so these numbers may narrow that cone and must never widen it.
     */
    private JSONObject arcBlock() throws Exception {
        JSONObject block = new JSONObject();
        block.put("swept_deg_left", round(arc.leftDeg(), 2));
        block.put("swept_deg_right", round(arc.rightDeg(), 2));
        block.put("swept_deg_total", round(arc.sweptDeg(), 2));
        block.put("implied_cone_deg", round(arc.impliedConeDeg(), 2));
        block.put("return_error_deg", round(arc.returnErrorDeg(), 2));
        block.put("median_rate_deg_per_sec", round(arc.medianRateDegPerSec(), 2));
        block.put("duration_sec", round(arc.durationSec(), 3));
        block.put("target_deg", preset.arcTargetDeg);
        block.put("floor_deg", preset.arcFloorDeg);
        block.put("reached_floor", arc.sweptDeg() >= preset.arcFloorDeg);
        // Whether the orbit-versus-pan question was answerable at all on this device, so nobody
        // downstream reads a default as a measurement.
        block.put("orbit_measured", arc.orbitMeasured());
        block.put("orbit_likely", arc.orbitLikely());
        block.put("orbit_samples", arc.orbitSamples());
        block.put("pan_samples", arc.panSamples());
        return block;
    }

    /** What the phone observed about being at rest while the subject moved. */
    private JSONObject fixedCameraBlock() throws Exception {
        JSONObject block = new JSONObject();
        // The app cannot know how the phone was held still, and guessing would put a claim in the
        // record that nothing measured.
        block.put("support", "unknown");
        block.put("rest_confirmed", machine != null && machine.restConfirmed());
        block.put("rest_residual_deg", round(stillness.residualDeg(), 3));
        block.put("auto_stopped", finishedReason != null);
        // Beyond the schema. `disturbed` is the latch that says rest was confirmed and then lost,
        // which is the difference between a take that may be presented as fixed-camera and one
        // that must not be.
        block.put("rest_disturbed", stillness.disturbed());
        block.put("max_tilt_deg", round(stillness.maxTiltDeg(), 3));
        block.put("tilt_budget_deg", Stillness.MAX_TILT_DEG);
        block.put("rest_confirmed_at_ms", machine == null ? -1 : machine.restConfirmedAtMs());
        return block;
    }

    /** Seconds between the first and last recorded frame, from their capture timestamps. */
    private static double recordedSeconds(JSONArray spans) throws Exception {
        if (spans.length() == 0) {
            return 0;
        }
        long first = spans.getJSONObject(0).getLong("t_ns_first");
        long last = spans.getJSONObject(spans.length() - 1).getLong("t_ns_last");
        return last > first ? (last - first) / 1_000_000_000.0 : 0;
    }

    /** JSON has no notion of significant figures, and a raw float prints fifteen of them. */
    private static double round(double value, int places) {
        double scale = Math.pow(10, places);
        return Math.round(value * scale) / scale;
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
