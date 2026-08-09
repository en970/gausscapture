package com.gausscapture.capture;

import android.Manifest;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.graphics.SurfaceTexture;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;
import android.view.TextureView;
import android.view.WindowManager;

import androidx.annotation.NonNull;

import org.json.JSONObject;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import io.flutter.embedding.android.FlutterActivity;
import io.flutter.embedding.engine.FlutterEngine;
import io.flutter.plugin.common.EventChannel;
import io.flutter.plugin.common.MethodCall;
import io.flutter.plugin.common.MethodChannel;

/**
 * The seam between Flutter and the capture engine.
 *
 * <p>Flutter owns every pixel; nothing below this line draws anything except the camera preview,
 * which is a platform view because it has to be a real {@code TextureView} — the transform that
 * makes the preview upright and correctly proportioned is written against one, and that code is
 * the single most error-prone thing in the app.
 *
 * <p>The division is deliberate rather than incidental. Flutter's own camera plugin cannot give a
 * per-frame {@code SENSOR_TIMESTAMP}, and its sensor plugins do not expose {@code
 * SensorEvent.timestamp} at all. Those two values are the entire reason this app exists: pairing
 * them recovers which way is up, measured at 0.998 agreement across a capture. A pure-Dart rewrite
 * would have produced a nicer-looking app that records scientifically useless data.
 *
 * <p>So: three channels. Commands go one way, a stream of live telemetry comes back, and the
 * preview is a texture Flutter composites like any other widget.
 */
public final class MainActivity extends FlutterActivity implements CaptureEngine.Events {

    private static final String COMMANDS = "gausscapture/commands";
    private static final String TELEMETRY = "gausscapture/telemetry";
    private static final String PREVIEW_VIEW = "gausscapture/preview";

    private static final int PERMISSION_REQUEST = 41;

    /** Fast enough to feel live, slow enough to leave the UI thread alone. */
    private static final long TELEMETRY_INTERVAL_MS = 100;

    private CaptureEngine engine;
    private MethodChannel commands;
    private EventChannel.EventSink telemetry;
    private final Handler main = new Handler(Looper.getMainLooper());
    private PreviewFactory previewFactory;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        engine = new CaptureEngine(this);
        engine.setEvents(this);

        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, PERMISSION_REQUEST);
        }
    }

    @Override
    public void configureFlutterEngine(@NonNull FlutterEngine flutterEngine) {
        super.configureFlutterEngine(flutterEngine);

        previewFactory = new PreviewFactory(this, engine);
        flutterEngine.getPlatformViewsController().getRegistry()
                .registerViewFactory(PREVIEW_VIEW, previewFactory);

        commands = new MethodChannel(flutterEngine.getDartExecutor().getBinaryMessenger(), COMMANDS);
        commands.setMethodCallHandler(this::onCommand);

        new EventChannel(flutterEngine.getDartExecutor().getBinaryMessenger(), TELEMETRY)
                .setStreamHandler(new EventChannel.StreamHandler() {
                    @Override
                    public void onListen(Object arguments, EventChannel.EventSink sink) {
                        telemetry = sink;
                        main.post(pump);
                    }

                    @Override
                    public void onCancel(Object arguments) {
                        telemetry = null;
                        main.removeCallbacks(pump);
                    }
                });
    }

    // ---------------------------------------------------------------- commands

    private void onCommand(MethodCall call, MethodChannel.Result result) {
        try {
            switch (call.method) {
                case "status":
                    result.success(status());
                    break;

                case "start": {
                    String failure = engine.start(displayRotation());
                    if (failure == null) {
                        // The routine is "record all the time" and the default display timeout is
                        // 30 seconds; without this the screen sleeps and the take is truncated.
                        // Full brightness is the other half — no colour choice compensates for a
                        // dim panel in sunlight.
                        holdScreen(true);
                        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LOCKED);
                        result.success(true);
                    } else {
                        result.error("start_failed", failure, null);
                    }
                    break;
                }

                case "stop": {
                    JSONObject manifest = engine.stop();
                    holdScreen(false);
                    setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED);
                    result.success(manifest == null ? null : toMap(manifest));
                    break;
                }

                case "setPreset":
                    engine.setPreset(call.argument("id"));
                    result.success(status());
                    break;

                case "presets": {
                    List<Map<String, Object>> out = new ArrayList<>();
                    for (Preset preset : Preset.ALL) {
                        Map<String, Object> row = new HashMap<>();
                        row.put("id", preset.id);
                        row.put("title", preset.title);
                        row.put("hint", preset.hint);
                        row.put("targetSeconds", preset.targetSeconds);
                        row.put("expectedToFail", preset.expectedToFail);
                        row.put("scripted", preset.isScripted());
                        row.put("captureType", preset.captureType);
                        row.put("phaseCount", preset.phases.length);
                        out.add(row);
                    }
                    result.success(out);
                    break;
                }

                case "takes": {
                    List<Object> out = new ArrayList<>();
                    for (JSONObject take : engine.takes()) {
                        out.add(toMap(take));
                    }
                    result.success(out);
                    break;
                }

                case "deleteTake":
                    result.success(engine.deleteTake(call.argument("name")));
                    break;

                case "requestSharedStorage":
                    result.success(requestSharedStorage());
                    break;

                default:
                    result.notImplemented();
            }
        } catch (Exception error) {
            result.error("engine_error", String.valueOf(error.getMessage()), null);
        }
    }

    /**
     * Everything the interface needs to decide what to show, in one call.
     *
     * <p>Preflight is part of it rather than a separate screen's private business: whether the
     * camera and the sensors share a clock decides whether a take is worth anything at all, and
     * discovering that on a Mac a week later is the most expensive failure this app can produce.
     */
    private Map<String, Object> status() {
        Map<String, Object> out = new HashMap<>();
        out.put("recording", engine.isRecording());
        out.put("preset", engine.preset().id);
        out.put("presetTitle", engine.preset().title);
        out.put("presetHint", engine.preset().hint);
        out.put("targetSeconds", engine.preset().targetSeconds);
        out.put("expectedToFail", engine.preset().expectedToFail);

        out.put("storagePath", engine.storage().describePath());
        out.put("storageVisible", engine.storage().isVisibleToDesktop());
        out.put("minutesRemaining", engine.storage().minutesRemaining(24_000_000));

        out.put("clockAligned", engine.clocks().alignable());
        out.put("clockDescription", engine.clocks().describe());
        out.put("cameraRealtime", engine.clocks().cameraUsesRealtime());
        out.put("sensorClock", engine.clocks().sensorClock().name());

        // The script, so the interface can draw the four phases before a take starts rather than
        // discovering them one at a time as they arrive in telemetry.
        out.put("scripted", engine.isScripted());
        out.put("phaseIds", engine.phaseIds());
        out.put("phaseCount", engine.phaseCount());
        out.put("arcTargetDeg", engine.arcTargetDeg());

        CameraController camera = engine.camera();
        out.put("focalPixels", camera == null ? 0f : camera.focalPixels());
        out.put("cameraReady", camera != null);
        out.put("imuRateHz", engine.sensors().measuredRateHz());
        return out;
    }

    // --------------------------------------------------------------- telemetry

    private final Runnable pump = new Runnable() {
        @Override
        public void run() {
            if (telemetry == null) {
                return;
            }
            Map<String, Object> frame = new HashMap<>();
            frame.put("recording", engine.isRecording());
            frame.put("elapsedMs", engine.elapsedMillis());
            frame.put("blurPixels", engine.blurPixels());
            frame.put("angularRate", engine.sensors().angularRate());
            frame.put("imuSamples", engine.sensors().sampleCount());
            frame.put("imuRateHz", engine.sensors().measuredRateHz());

            // The script. Ten values rather than one because each drives a different layer of the
            // guidance, and an interface that had to infer the phase from the elapsed time would
            // be wrong exactly at the boundaries, which are the moments it exists to signal.
            frame.put("phaseId", engine.phaseId());
            frame.put("phaseIndex", engine.phaseIndex());
            frame.put("phaseCount", engine.phaseCount());
            frame.put("phaseCue", engine.phaseCue());
            frame.put("phaseDetail", engine.phaseDetail());
            frame.put("phaseWarning", engine.phaseWarning());
            frame.put("phaseElapsedMs", engine.phaseElapsedMs());
            frame.put("phaseTargetMs", engine.phaseTargetMs());
            frame.put("phaseTransitions", engine.phaseTransitions());
            frame.put("countdownMs", engine.countdownMillis());
            frame.put("atRest", engine.atRest());
            frame.put("restForMs", engine.restForMillis());
            frame.put("restDisturbed", engine.restDisturbed());
            frame.put("arcSweptDeg", engine.arcSweptDeg());
            frame.put("arcDirection", engine.arcDirection());

            CameraController camera = engine.camera();
            frame.put("frames", camera == null ? 0 : camera.frameCount());
            frame.put("exposureNs", camera == null ? 0L : camera.lastExposureNanos());
            telemetry.success(frame);
            main.postDelayed(this, TELEMETRY_INTERVAL_MS);
        }
    };

    @Override
    public void onProblem(final String message) {
        main.post(new Runnable() {
            @Override
            public void run() {
                holdScreen(false);
                if (commands != null) {
                    commands.invokeMethod("problem", message);
                }
            }
        });
    }

    @Override
    public void onStateChanged() {
        main.post(new Runnable() {
            @Override
            public void run() {
                if (commands != null) {
                    commands.invokeMethod("stateChanged", status());
                }
            }
        });
    }

    /**
     * A scripted take ended by itself, and nothing on the Flutter side asked for it.
     *
     * <p>This is the callback the whole auto-stop exists for. The dynamic phase ends without anyone
     * touching the phone — that is the point, because reaching for it moves it and the frames that
     * would corrupt are the last ones — so the interface has no way to learn the take is over
     * except by being told. Without this the record button would sit there claiming to be
     * recording, and the operator would press it and start a second take.
     *
     * <p>The screen hold and the orientation lock are released here for the same reason: they were
     * taken in {@code start} and the ordinary release path runs in the {@code stop} command, which
     * never ran.
     */
    @Override
    public void onFinished(final JSONObject manifest, final String reason, final boolean aborted) {
        main.post(new Runnable() {
            @Override
            public void run() {
                holdScreen(false);
                setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED);
                if (commands == null) {
                    return;
                }
                Map<String, Object> out = new HashMap<>();
                out.put("reason", reason);
                out.put("aborted", aborted);
                out.put("manifest", manifest == null ? null : toMap(manifest));
                out.put("status", status());
                commands.invokeMethod("finished", out);
            }
        });
    }

    // ----------------------------------------------------------------- storage

    private boolean requestSharedStorage() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R || Storage.canUseSharedStorage()) {
            return true;
        }
        try {
            startActivity(new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                    Uri.parse("package:" + getPackageName())));
        } catch (Exception error) {
            startActivity(new Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION));
        }
        return false;
    }

    // --------------------------------------------------------------- lifecycle

    @Override
    protected void onResume() {
        super.onResume();
        // The all-files grant can be given or revoked while backgrounded, so where captures land
        // is decided again here rather than once at startup.
        engine.refreshStorage();
        if (previewFactory != null) {
            previewFactory.resume(displayRotation());
        }
    }

    @Override
    protected void onPause() {
        // A take must not survive into the background: the camera will be taken away and the file
        // left without its trailer.
        if (engine.isRecording()) {
            engine.stop();
            holdScreen(false);
        }
        engine.closeCamera();
        super.onPause();
    }

    @Override
    public void onRequestPermissionsResult(int code, @NonNull String[] permissions,
                                           @NonNull int[] results) {
        super.onRequestPermissionsResult(code, permissions, results);
        if (code == PERMISSION_REQUEST && previewFactory != null) {
            previewFactory.resume(displayRotation());
        }
    }

    /**
     * Keep the screen alive for the take, at the brightness the preset asks for.
     *
     * <p>Not one brightness for every preset. On the back camera the panel faces away from the
     * scene and full brightness is the only thing that makes the preview readable outdoors. On the
     * front camera the panel <em>is</em> a light source about 300 mm from the subject's face, and
     * during the arc it travels with the camera — so a bright panel writes a moving specular
     * highlight into every frame and breaks the brightness constancy the reconstruction assumes.
     * {@link Preset#screenBrightness} carries that decision; forcing 1.0 here would quietly
     * override it.
     */
    private void holdScreen(boolean hold) {
        WindowManager.LayoutParams attributes = getWindow().getAttributes();
        if (hold) {
            getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            float wanted = engine.preset().screenBrightness;
            attributes.screenBrightness = wanted < 0
                    ? WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE
                    : Math.max(0f, Math.min(1f, wanted));
        } else {
            getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            attributes.screenBrightness = WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE;
        }
        getWindow().setAttributes(attributes);
    }

    private int displayRotation() {
        return getWindowManager().getDefaultDisplay().getRotation();
    }

    @SuppressWarnings("unchecked")
    private static Object toMap(Object value) {
        if (value instanceof JSONObject) {
            JSONObject json = (JSONObject) value;
            Map<String, Object> out = new HashMap<>();
            for (java.util.Iterator<String> it = json.keys(); it.hasNext(); ) {
                String key = it.next();
                Object child = json.opt(key);
                out.put(key, child == JSONObject.NULL ? null : toMap(child));
            }
            return out;
        }
        if (value instanceof org.json.JSONArray) {
            org.json.JSONArray array = (org.json.JSONArray) value;
            List<Object> out = new ArrayList<>();
            for (int i = 0; i < array.length(); i++) {
                Object child = array.opt(i);
                out.add(child == JSONObject.NULL ? null : toMap(child));
            }
            return out;
        }
        return value;
    }
}
