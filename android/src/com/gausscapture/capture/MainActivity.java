package com.gausscapture.capture;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.OutputConfiguration;
import android.hardware.camera2.params.SessionConfiguration;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Range;
import android.util.Size;
import android.util.SizeF;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.UUID;
import java.util.concurrent.Executor;

/**
 * Capture app for GaussCapture.
 *
 * <p>This app does not try to take nicer video than the stock camera. It records what the stock
 * camera discards, and removes the operator mistakes that no amount of careful instruction
 * reliably prevents:
 *
 * <ul>
 *   <li><b>Locked exposure, white balance and focus.</b> Auto algorithms drift across a capture and
 *       break the brightness-constancy assumption reconstruction relies on. Correcting that
 *       afterwards is worth several dB; not breaking it costs nothing.
 *   <li><b>Stabilisation off.</b> Optical and digital stabilisation warp the image per frame, which
 *       invalidates a single shared intrinsic model. A correctness requirement, not a preference.
 *   <li><b>One fixed lens.</b> The stock camera silently switches between ultra-wide, wide and
 *       telephoto as you zoom, changing intrinsics mid-capture.
 *   <li><b>Camera intrinsics</b> read from the device rather than solved for.
 *   <li><b>IMU at the hardware maximum</b>, timestamped in the same clock as the video frames
 *       wherever the device reports {@code TIMESTAMP_SOURCE_REALTIME} -- alignment a browser
 *       cannot provide at any price.
 * </ul>
 *
 * <p>Sessions land under the app's external files dir, retrievable with {@code adb pull} and
 * needing no runtime storage permission.
 */
public class MainActivity extends Activity {

    private static final int PERMISSION_REQUEST = 1;

    private static final int TARGET_WIDTH = 1920;
    private static final int TARGET_HEIGHT = 1080;
    private static final int TARGET_FPS = 30;
    /** Generous, because compression artefacts are a confound we would rather not introduce. */
    private static final int VIDEO_BITRATE = 24_000_000;

    /**
     * The capture protocol, one tap each.
     *
     * <p>Typing a filename one-handed while holding a phone at arm's length is exactly the kind of
     * friction that produces mislabelled data, and mislabelled data is worse than none. The last
     * two entries are expected to fail: a model trained only on successes learns nothing, and
     * zero-parallax and moving-subject captures are the two failure modes worth having.
     */
    private static final String[][] PRESETS = {
            {"A_good", "Locked settings. Slow full orbit, subject centred. ~60s"},
            {"B_normal", "Normal walking pace, one loop. ~30s"},
            {"C_fast", "Deliberately fast and jerky. ~20s"},
            {"D_rotation", "Stand still, rotate the phone only. Should fail. ~20s"},
            {"E_subject", "Phone still, subject moves. Should fail. ~20s"},
    };

    private TextureView previewView;
    private TextView statusView;
    private TextView timerView;
    private TextView hintView;
    private LinearLayout presetRow;
    private Button recordButton;

    private CameraManager cameraManager;
    private CameraDevice cameraDevice;
    private CameraCaptureSession captureSession;
    private CaptureRequest.Builder requestBuilder;
    private String cameraId;
    private CameraCharacteristics characteristics;
    private int sensorOrientation;
    private Size previewSize = new Size(TARGET_WIDTH, TARGET_HEIGHT);

    private MediaRecorder mediaRecorder;
    private HandlerThread backgroundThread;
    private Handler backgroundHandler;
    private Executor backgroundExecutor;

    private SensorManager sensorManager;
    private final List<Sensor> sensors = new ArrayList<>();
    private BufferedWriter imuWriter;
    private BufferedWriter frameWriter;
    private volatile int imuSamples;
    private volatile int frameCount;

    private volatile boolean recording;
    private File sessionDir;
    private String selectedPreset = PRESETS[0][0];
    private long recordingStartRealtimeNanos;
    private long recordingStartUptimeNanos;
    private long recordingStartWallMillis;
    private Float lockedFocusDistance;
    private Long lastIso;
    private Long lastExposureNanos;

    private final Handler uiHandler = new Handler();
    private final Runnable timerTick = new Runnable() {
        @Override
        public void run() {
            if (!recording) {
                return;
            }
            long seconds =
                    (SystemClock.elapsedRealtimeNanos() - recordingStartRealtimeNanos) / 1_000_000_000L;
            timerView.setText(String.format(Locale.US, "%02d:%02d", seconds / 60, seconds % 60));
            statusView.setText(String.format(Locale.US,
                    "REC %s\nframes %d   imu %d\niso %s   exp %s ms",
                    selectedPreset, frameCount, imuSamples,
                    lastIso == null ? "?" : lastIso.toString(),
                    lastExposureNanos == null
                            ? "?" : String.format(Locale.US, "%.1f", lastExposureNanos / 1e6)));
            uiHandler.postDelayed(this, 250);
        }
    };

    // ---------------------------------------------------------------- lifecycle

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        previewView = findViewById(R.id.preview);
        statusView = findViewById(R.id.status);
        timerView = findViewById(R.id.timer);
        hintView = findViewById(R.id.hint);
        presetRow = findViewById(R.id.presets);
        recordButton = findViewById(R.id.record);

        cameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);

        buildPresetChips();
        timerView.setVisibility(View.GONE);

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

        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, PERMISSION_REQUEST);
        }
    }

    private void buildPresetChips() {
        presetRow.removeAllViews();
        for (int i = 0; i < PRESETS.length; i++) {
            final String name = PRESETS[i][0];
            final String hint = PRESETS[i][1];
            Button chip = new Button(this);
            chip.setText(name);
            chip.setAllCaps(false);
            chip.setTextSize(13);
            chip.setPadding(34, 12, 34, 12);
            LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
            params.setMarginEnd(10);
            chip.setLayoutParams(params);
            chip.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    if (recording) {
                        return;  // Changing the label mid-take would mislabel the data.
                    }
                    selectedPreset = name;
                    hintView.setText(hint);
                    refreshChips();
                }
            });
            presetRow.addView(chip);
        }
        hintView.setText(PRESETS[0][1]);
        refreshChips();
    }

    private void refreshChips() {
        for (int i = 0; i < presetRow.getChildCount(); i++) {
            Button chip = (Button) presetRow.getChildAt(i);
            boolean on = PRESETS[i][0].equals(selectedPreset);
            chip.setBackgroundResource(on ? R.drawable.chip_on : R.drawable.chip);
            chip.setTextColor(on ? 0xFF14141A : 0xFFECEAF0);
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        startBackgroundThread();
        if (previewView.isAvailable()) {
            openCamera();
        } else {
            previewView.setSurfaceTextureListener(new TextureView.SurfaceTextureListener() {
                @Override
                public void onSurfaceTextureAvailable(SurfaceTexture s, int w, int h) {
                    openCamera();
                }

                @Override
                public void onSurfaceTextureSizeChanged(SurfaceTexture s, int w, int h) {
                    configureTransform(w, h);
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
        closeCamera();
        stopBackgroundThread();
        super.onPause();
    }

    // ---------------------------------------------------------------- orientation

    private int displayRotation() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && getDisplay() != null) {
            return getDisplay().getRotation();
        }
        return getWindowManager().getDefaultDisplay().getRotation();
    }

    private static int rotationDegrees(int rotation) {
        switch (rotation) {
            case Surface.ROTATION_90: return 90;
            case Surface.ROTATION_180: return 180;
            case Surface.ROTATION_270: return 270;
            default: return 0;
        }
    }

    /**
     * Degrees the camera image must be rotated to appear upright on screen.
     *
     * <p>The sensor is mounted at a fixed angle relative to the device's natural orientation --
     * 90 degrees on this phone -- so the buffer arrives rotated by that much regardless of how the
     * device is being held. Subtracting the current display rotation gives what is left to correct.
     */
    private int previewRotation() {
        return (sensorOrientation - rotationDegrees(displayRotation()) + 360) % 360;
    }

    /**
     * Undoes TextureView's stretch, then rotates and centre-crops.
     *
     * <p>TextureView always scales the surface buffer to fill the view's bounds, ignoring aspect
     * ratio, and applies any transform matrix on top of that. Supplying no matrix therefore
     * guarantees a distorted image whenever the view and buffer differ in shape -- which is why
     * the preview appeared stretched. The first step maps the view rect back onto the buffer's
     * true proportions, cancelling that stretch; then we scale up to cover the view and rotate.
     */
    private void configureTransform(int viewWidth, int viewHeight) {
        if (viewWidth == 0 || viewHeight == 0) {
            return;
        }
        int rotate = previewRotation();
        // A quarter turn swaps which buffer dimension maps to which screen axis.
        float bufferWidth = (rotate % 180 == 0) ? previewSize.getWidth() : previewSize.getHeight();
        float bufferHeight = (rotate % 180 == 0) ? previewSize.getHeight() : previewSize.getWidth();

        RectF viewRect = new RectF(0, 0, viewWidth, viewHeight);
        RectF bufferRect = new RectF(0, 0, bufferWidth, bufferHeight);
        float centerX = viewRect.centerX();
        float centerY = viewRect.centerY();
        bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY());

        Matrix matrix = new Matrix();
        matrix.setRectToRect(viewRect, bufferRect, Matrix.ScaleToFit.FILL);
        float scale = Math.max(viewHeight / bufferHeight, viewWidth / bufferWidth);
        matrix.postScale(scale, scale, centerX, centerY);
        matrix.postRotate(rotate, centerX, centerY);
        previewView.setTransform(matrix);
    }

    // ---------------------------------------------------------------- camera

    /**
     * Chooses the main back camera explicitly.
     *
     * <p>Modern phones expose several back-facing physical cameras and the stock app switches
     * between them as you zoom, changing intrinsics partway through a capture. We pick one and stay
     * on it: among back-facing cameras, the one whose focal length is nearest the main lens.
     * Ultra-wide sits near 2mm and telephoto near 7mm or beyond.
     */
    private String selectBackCamera() throws CameraAccessException {
        String best = null;
        float bestScore = Float.MAX_VALUE;
        for (String id : cameraManager.getCameraIdList()) {
            CameraCharacteristics c = cameraManager.getCameraCharacteristics(id);
            Integer facing = c.get(CameraCharacteristics.LENS_FACING);
            if (facing == null || facing != CameraCharacteristics.LENS_FACING_BACK) {
                continue;
            }
            float[] focals = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS);
            if (focals == null || focals.length == 0) {
                continue;
            }
            float score = Math.abs(focals[0] - 5.0f);
            if (score < bestScore) {
                bestScore = score;
                best = id;
            }
        }
        return best;
    }

    /** Nearest available size to the target, preferring an exact match. */
    private Size chooseSize(Size[] choices) {
        if (choices == null || choices.length == 0) {
            return new Size(TARGET_WIDTH, TARGET_HEIGHT);
        }
        Size best = choices[0];
        long bestCost = Long.MAX_VALUE;
        for (Size size : choices) {
            long cost = Math.abs((long) size.getWidth() - TARGET_WIDTH)
                    + Math.abs((long) size.getHeight() - TARGET_HEIGHT);
            if (cost < bestCost) {
                bestCost = cost;
                best = size;
            }
        }
        return best;
    }

    private void openCamera() {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            statusView.setText("camera permission not granted");
            return;
        }
        try {
            cameraId = selectBackCamera();
            if (cameraId == null) {
                statusView.setText("no back camera found");
                return;
            }
            characteristics = cameraManager.getCameraCharacteristics(cameraId);
            Integer orientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
            sensorOrientation = orientation == null ? 90 : orientation;

            StreamConfigurationMap map = characteristics.get(
                    CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            if (map != null) {
                previewSize = chooseSize(map.getOutputSizes(MediaRecorder.class));
            }

            cameraManager.openCamera(cameraId, new CameraDevice.StateCallback() {
                @Override
                public void onOpened(CameraDevice device) {
                    cameraDevice = device;
                    startPreview();
                }

                @Override
                public void onDisconnected(CameraDevice device) {
                    device.close();
                    cameraDevice = null;
                }

                @Override
                public void onError(CameraDevice device, int error) {
                    device.close();
                    cameraDevice = null;
                    setStatus("camera error " + error);
                }
            }, backgroundHandler);
        } catch (CameraAccessException | SecurityException e) {
            statusView.setText("open failed: " + e.getMessage());
        }
    }

    private void closeCamera() {
        if (captureSession != null) {
            captureSession.close();
            captureSession = null;
        }
        if (cameraDevice != null) {
            cameraDevice.close();
            cameraDevice = null;
        }
        if (mediaRecorder != null) {
            mediaRecorder.release();
            mediaRecorder = null;
        }
    }

    private Surface newPreviewSurface() {
        SurfaceTexture texture = previewView.getSurfaceTexture();
        texture.setDefaultBufferSize(previewSize.getWidth(), previewSize.getHeight());
        uiHandler.post(new Runnable() {
            @Override
            public void run() {
                configureTransform(previewView.getWidth(), previewView.getHeight());
            }
        });
        return new Surface(texture);
    }

    /** Preview only, with the auto algorithms running so they converge before we lock them. */
    private void startPreview() {
        try {
            Surface previewSurface = newPreviewSurface();
            requestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            requestBuilder.addTarget(previewSurface);
            applyFixedCaptureSettings(requestBuilder, false);

            List<OutputConfiguration> outputs =
                    Collections.singletonList(new OutputConfiguration(previewSurface));
            cameraDevice.createCaptureSession(new SessionConfiguration(
                    SessionConfiguration.SESSION_REGULAR, outputs, backgroundExecutor,
                    new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured(CameraCaptureSession session) {
                            captureSession = session;
                            try {
                                session.setRepeatingRequest(requestBuilder.build(),
                                        metadataCallback, backgroundHandler);
                                setStatus(describeCamera());
                            } catch (CameraAccessException e) {
                                setStatus("preview failed: " + e.getMessage());
                            }
                        }

                        @Override
                        public void onConfigureFailed(CameraCaptureSession session) {
                            setStatus("preview session config failed");
                        }
                    }));
        } catch (CameraAccessException e) {
            setStatus("preview error: " + e.getMessage());
        }
    }

    /**
     * Applies the settings that make a capture reconstructable.
     *
     * @param lock when true, freeze exposure, white balance and focus for the duration of a take
     */
    private void applyFixedCaptureSettings(CaptureRequest.Builder b, boolean lock) {
        b.set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_OFF);
        b.set(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
                CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_OFF);

        b.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);
        b.set(CaptureRequest.CONTROL_SCENE_MODE, CameraMetadata.CONTROL_SCENE_MODE_DISABLED);
        b.set(CaptureRequest.CONTROL_EFFECT_MODE, CameraMetadata.CONTROL_EFFECT_MODE_OFF);
        b.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, new Range<>(TARGET_FPS, TARGET_FPS));

        // Report geometry as captured, so the distortion coefficients we record describe the
        // pixels we recorded rather than a corrected version of them.
        if (characteristics.get(CameraCharacteristics.DISTORTION_CORRECTION_AVAILABLE_MODES)
                != null) {
            b.set(CaptureRequest.DISTORTION_CORRECTION_MODE,
                    CameraMetadata.DISTORTION_CORRECTION_MODE_OFF);
        }

        if (lock) {
            b.set(CaptureRequest.CONTROL_AE_LOCK, true);
            b.set(CaptureRequest.CONTROL_AWB_LOCK, true);
            b.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_OFF);
            if (lockedFocusDistance != null) {
                b.set(CaptureRequest.LENS_FOCUS_DISTANCE, lockedFocusDistance);
            }
        } else {
            b.set(CaptureRequest.CONTROL_AE_LOCK, false);
            b.set(CaptureRequest.CONTROL_AWB_LOCK, false);
            b.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_VIDEO);
        }
    }

    /** Records what the camera actually did, per frame, rather than what we asked for. */
    private final CameraCaptureSession.CaptureCallback metadataCallback =
            new CameraCaptureSession.CaptureCallback() {
                @Override
                public void onCaptureCompleted(CameraCaptureSession session, CaptureRequest request,
                                               TotalCaptureResult result) {
                    Float focus = result.get(CaptureResult.LENS_FOCUS_DISTANCE);
                    if (focus != null) {
                        lockedFocusDistance = focus;
                    }
                    lastIso = longOrNull(result.get(CaptureResult.SENSOR_SENSITIVITY));
                    lastExposureNanos = result.get(CaptureResult.SENSOR_EXPOSURE_TIME);

                    if (!recording) {
                        return;
                    }
                    try {
                        JSONObject row = new JSONObject();
                        Long timestamp = result.get(CaptureResult.SENSOR_TIMESTAMP);
                        row.put("t_ns", timestamp == null ? JSONObject.NULL : timestamp);
                        row.put("iso", lastIso == null ? JSONObject.NULL : lastIso);
                        row.put("exposure_ns", lastExposureNanos == null
                                ? JSONObject.NULL : lastExposureNanos);
                        row.put("focus_distance", focus == null ? JSONObject.NULL : focus);
                        Float aperture = result.get(CaptureResult.LENS_APERTURE);
                        row.put("aperture", aperture == null ? JSONObject.NULL : aperture);
                        Float focal = result.get(CaptureResult.LENS_FOCAL_LENGTH);
                        row.put("focal_length_mm", focal == null ? JSONObject.NULL : focal);
                        row.put("frame", frameCount);
                        synchronized (MainActivity.this) {
                            if (frameWriter != null) {
                                frameWriter.write(row.toString());
                                frameWriter.write("\n");
                            }
                        }
                        frameCount++;
                    } catch (Exception ignored) {
                        // A dropped metadata row must never interrupt a recording.
                    }
                }
            };

    // ---------------------------------------------------------------- recording

    private void startRecording() {
        if (cameraDevice == null) {
            setStatus("camera not ready");
            return;
        }
        try {
            sessionDir = new File(getExternalFilesDir(null), selectedPreset + "_"
                    + new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date()));
            if (!sessionDir.mkdirs() && !sessionDir.isDirectory()) {
                setStatus("cannot create session dir");
                return;
            }

            frameCount = 0;
            imuSamples = 0;
            imuWriter = new BufferedWriter(new FileWriter(new File(sessionDir, "imu.jsonl")));
            frameWriter = new BufferedWriter(new FileWriter(new File(sessionDir, "frames.jsonl")));

            if (captureSession != null) {
                captureSession.close();
                captureSession = null;
            }

            mediaRecorder = buildRecorder(new File(sessionDir, "video.mp4"));
            mediaRecorder.prepare();

            Surface previewSurface = newPreviewSurface();
            Surface recorderSurface = mediaRecorder.getSurface();

            requestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_RECORD);
            requestBuilder.addTarget(previewSurface);
            requestBuilder.addTarget(recorderSurface);
            // Lock now: the preview has been running, so the auto algorithms have converged and
            // the values they settled on are the ones we freeze.
            applyFixedCaptureSettings(requestBuilder, true);

            List<OutputConfiguration> outputs = Arrays.asList(
                    new OutputConfiguration(previewSurface),
                    new OutputConfiguration(recorderSurface));

            cameraDevice.createCaptureSession(new SessionConfiguration(
                    SessionConfiguration.SESSION_REGULAR, outputs, backgroundExecutor,
                    new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured(CameraCaptureSession session) {
                            captureSession = session;
                            try {
                                session.setRepeatingRequest(requestBuilder.build(),
                                        metadataCallback, backgroundHandler);
                                beginRecordingClockAndSensors();
                            } catch (CameraAccessException e) {
                                setStatus("record request failed: " + e.getMessage());
                            }
                        }

                        @Override
                        public void onConfigureFailed(CameraCaptureSession session) {
                            setStatus("record session config failed");
                        }
                    }));
        } catch (Exception e) {
            setStatus("start failed: " + e.getMessage());
        }
    }

    /** Stamps both clocks at t=0, starts the sensors, then the encoder. */
    private void beginRecordingClockAndSensors() {
        recordingStartRealtimeNanos = SystemClock.elapsedRealtimeNanos();
        recordingStartUptimeNanos = System.nanoTime();
        recordingStartWallMillis = System.currentTimeMillis();

        registerSensors();
        mediaRecorder.start();
        recording = true;

        uiHandler.post(new Runnable() {
            @Override
            public void run() {
                recordButton.setBackgroundResource(R.drawable.btn_stop);
                timerView.setVisibility(View.VISIBLE);
                timerView.setText("00:00");
                hintView.setText("recording " + selectedPreset);
                uiHandler.post(timerTick);
            }
        });
    }

    private void stopRecording() {
        recording = false;
        uiHandler.removeCallbacks(timerTick);
        unregisterSensors();

        boolean tooShort = false;
        try {
            if (mediaRecorder != null) {
                mediaRecorder.stop();
            }
        } catch (RuntimeException e) {
            // stop() throws when no frames were written -- a take shorter than one frame.
            tooShort = true;
        } finally {
            if (mediaRecorder != null) {
                mediaRecorder.reset();
                mediaRecorder.release();
                mediaRecorder = null;
            }
        }

        synchronized (this) {
            closeQuietly(imuWriter);
            closeQuietly(frameWriter);
            imuWriter = null;
            frameWriter = null;
        }

        final String summary;
        if (tooShort) {
            deleteRecursively(sessionDir);
            summary = "take too short, discarded";
        } else {
            writeManifest();
            summary = String.format(Locale.US, "saved %s\n%d frames   %d imu samples",
                    sessionDir.getName(), frameCount, imuSamples);
        }

        uiHandler.post(new Runnable() {
            @Override
            public void run() {
                recordButton.setBackgroundResource(R.drawable.btn_record);
                timerView.setVisibility(View.GONE);
                hintView.setText(summary.split("\n")[0]);
                statusView.setText(summary);
            }
        });

        startPreview();
    }

    private MediaRecorder buildRecorder(File output) {
        MediaRecorder recorder = new MediaRecorder(this);
        recorder.setVideoSource(MediaRecorder.VideoSource.SURFACE);
        recorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
        recorder.setOutputFile(output.getAbsolutePath());
        recorder.setVideoEncodingBitRate(VIDEO_BITRATE);
        recorder.setVideoFrameRate(TARGET_FPS);
        recorder.setVideoSize(previewSize.getWidth(), previewSize.getHeight());
        recorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264);
        // Without this the file plays back rotated by the sensor mounting angle, and every
        // downstream tool inherits the mistake.
        recorder.setOrientationHint(previewRotation());
        return recorder;
    }

    // ---------------------------------------------------------------- sensors

    private void registerSensors() {
        sensors.clear();
        addSensor(Sensor.TYPE_ACCELEROMETER);
        addSensor(Sensor.TYPE_GYROSCOPE);
        addSensor(Sensor.TYPE_ROTATION_VECTOR);
        // The uncalibrated variants expose the raw signal plus the bias the system is subtracting,
        // which is what you need if you ever intend to integrate it yourself.
        addSensor(Sensor.TYPE_GYROSCOPE_UNCALIBRATED);
        addSensor(Sensor.TYPE_ACCELEROMETER_UNCALIBRATED);

        for (Sensor sensor : sensors) {
            sensorManager.registerListener(sensorListener, sensor,
                    SensorManager.SENSOR_DELAY_FASTEST, backgroundHandler);
        }
    }

    private void addSensor(int type) {
        Sensor sensor = sensorManager.getDefaultSensor(type);
        if (sensor != null) {
            sensors.add(sensor);
        }
    }

    private void unregisterSensors() {
        sensorManager.unregisterListener(sensorListener);
        sensors.clear();
    }

    private final SensorEventListener sensorListener = new SensorEventListener() {
        @Override
        public void onSensorChanged(SensorEvent event) {
            if (!recording) {
                return;
            }
            try {
                JSONObject row = new JSONObject();
                row.put("type", event.sensor.getStringType());
                // SensorEvent.timestamp is elapsedRealtimeNanos, the same clock the camera reports
                // when its timestamp source is REALTIME. Both the absolute and the
                // relative-to-start forms are written: the absolute one aligns with frame
                // timestamps, the relative one is what a human can read.
                row.put("t_ns", event.timestamp);
                row.put("t_rel_ns", event.timestamp - recordingStartRealtimeNanos);
                JSONArray values = new JSONArray();
                for (float v : event.values) {
                    values.put((double) v);
                }
                row.put("v", values);
                row.put("accuracy", event.accuracy);
                synchronized (MainActivity.this) {
                    if (imuWriter != null) {
                        imuWriter.write(row.toString());
                        imuWriter.write("\n");
                    }
                }
                imuSamples++;
            } catch (Exception ignored) {
                // A dropped sample must never interrupt a recording.
            }
        }

        @Override
        public void onAccuracyChanged(Sensor sensor, int accuracy) { }
    };

    // ---------------------------------------------------------------- manifest

    /**
     * Writes what the desktop pipeline needs in order to interpret the recording.
     *
     * <p>Most importantly it records the camera's timestamp source. If that is not
     * {@code REALTIME}, video and IMU timestamps live in different clocks and alignment needs the
     * offset also written here. Saying so explicitly is the difference between data that can be
     * used and data that merely looks usable.
     */
    private void writeManifest() {
        try {
            JSONObject manifest = new JSONObject();
            manifest.put("capturepack_version", "0.1");
            manifest.put("session_id", UUID.randomUUID().toString());
            manifest.put("session_name", sessionDir.getName());
            manifest.put("preset", selectedPreset);
            manifest.put("capture_type", "static_scene");
            manifest.put("created_at", isoTime(recordingStartWallMillis));
            manifest.put("app", "gausscapture-android/0.1");

            JSONObject device = new JSONObject();
            device.put("manufacturer", Build.MANUFACTURER);
            device.put("model", Build.MODEL);
            device.put("os", "Android " + Build.VERSION.RELEASE
                    + " (API " + Build.VERSION.SDK_INT + ")");
            device.put("camera_id", cameraId);
            device.put("sensor_orientation", sensorOrientation);
            manifest.put("device", device);

            JSONObject video = new JSONObject();
            video.put("main_file", "video.mp4");
            video.put("width", previewSize.getWidth());
            video.put("height", previewSize.getHeight());
            video.put("fps", TARGET_FPS);
            video.put("codec", "h264");
            video.put("bitrate", VIDEO_BITRATE);
            video.put("orientation_hint", previewRotation());
            video.put("frames_recorded", frameCount);
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
            clocks.put("recording_start_elapsed_realtime_ns", recordingStartRealtimeNanos);
            clocks.put("recording_start_uptime_ns", recordingStartUptimeNanos);
            clocks.put("recording_start_wall_ms", recordingStartWallMillis);
            Integer source = characteristics.get(
                    CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE);
            boolean realtime = source != null
                    && source == CameraMetadata.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME;
            clocks.put("camera_timestamp_source", realtime ? "REALTIME" : "UNKNOWN");
            clocks.put("camera_imu_same_clock", realtime);
            manifest.put("clocks", clocks);

            JSONObject files = new JSONObject();
            files.put("intrinsics", "intrinsics.json");
            files.put("imu", "imu.jsonl");
            files.put("frames", "frames.jsonl");
            files.put("poses", JSONObject.NULL);
            files.put("light", JSONObject.NULL);
            manifest.put("metadata_files", files);
            manifest.put("imu_samples", imuSamples);

            write(new File(sessionDir, "manifest.json"), manifest.toString(2));
            write(new File(sessionDir, "intrinsics.json"), intrinsics().toString(2));
        } catch (Exception e) {
            setStatus("manifest failed: " + e.getMessage());
        }
    }

    /** Camera intrinsics as the device reports them, rather than as SfM would guess them. */
    private JSONObject intrinsics() throws Exception {
        JSONObject out = new JSONObject();
        out.put("source", "android_camera2_characteristics");
        out.put("camera_id", cameraId);

        float[] calibration = characteristics.get(
                CameraCharacteristics.LENS_INTRINSIC_CALIBRATION);
        if (calibration != null && calibration.length >= 5) {
            JSONObject k = new JSONObject();
            k.put("fx", calibration[0]);
            k.put("fy", calibration[1]);
            k.put("cx", calibration[2]);
            k.put("cy", calibration[3]);
            k.put("skew", calibration[4]);
            k.put("note", "In pre-correction active array pixels; rescale to the recorded "
                    + "resolution before use.");
            out.put("intrinsic_calibration", k);
        } else {
            out.put("intrinsic_calibration", JSONObject.NULL);
            out.put("intrinsic_note", "Device does not report LENS_INTRINSIC_CALIBRATION; derive "
                    + "focal length from focal_length_mm and sensor physical size, or let "
                    + "structure-from-motion solve it.");
        }

        float[] distortion = characteristics.get(CameraCharacteristics.LENS_DISTORTION);
        if (distortion != null) {
            JSONArray d = new JSONArray();
            for (float v : distortion) {
                d.put((double) v);
            }
            out.put("distortion_kappa", d);
            out.put("distortion_model", "android [k1,k2,k3,p1,p2]");
        }

        SizeF physical = characteristics.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE);
        if (physical != null) {
            JSONObject s = new JSONObject();
            s.put("width_mm", physical.getWidth());
            s.put("height_mm", physical.getHeight());
            out.put("sensor_physical_size", s);
        }

        android.graphics.Rect active = characteristics.get(
                CameraCharacteristics.SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE);
        if (active != null) {
            JSONObject a = new JSONObject();
            a.put("width", active.width());
            a.put("height", active.height());
            out.put("pre_correction_active_array", a);
        }

        float[] focals = characteristics.get(
                CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS);
        if (focals != null && focals.length > 0) {
            out.put("focal_length_mm", focals[0]);
        }
        out.put("recorded_size", previewSize.getWidth() + "x" + previewSize.getHeight());
        return out;
    }

    // ---------------------------------------------------------------- helpers

    private String describeCamera() {
        float[] focals = characteristics.get(
                CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS);
        Integer source = characteristics.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE);
        boolean realtime = source != null
                && source == CameraMetadata.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME;
        float[] calibration = characteristics.get(
                CameraCharacteristics.LENS_INTRINSIC_CALIBRATION);
        return String.format(Locale.US,
                "cam %s  %.1fmm  %dx%d@%d\nAE/AWB/AF lock  stabilisation off\n"
                        + "clock %s   intrinsics %s",
                cameraId,
                focals == null || focals.length == 0 ? 0f : focals[0],
                previewSize.getWidth(), previewSize.getHeight(), TARGET_FPS,
                realtime ? "REALTIME" : "UNKNOWN",
                calibration != null ? "on-device" : "not reported");
    }

    private void setStatus(final String text) {
        uiHandler.post(new Runnable() {
            @Override
            public void run() {
                statusView.setText(text);
            }
        });
    }

    private static Long longOrNull(Integer value) {
        return value == null ? null : value.longValue();
    }

    private static String isoTime(long millis) {
        SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US);
        format.setTimeZone(java.util.TimeZone.getTimeZone("UTC"));
        return format.format(new Date(millis));
    }

    private static void write(File file, String content) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
            writer.write(content);
        }
    }

    private static void closeQuietly(BufferedWriter writer) {
        if (writer == null) {
            return;
        }
        try {
            writer.flush();
            writer.close();
        } catch (IOException ignored) {
            // Nothing useful to do at this point.
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

    private void startBackgroundThread() {
        backgroundThread = new HandlerThread("GaussCaptureCamera");
        backgroundThread.start();
        backgroundHandler = new Handler(backgroundThread.getLooper());
        backgroundExecutor = new Executor() {
            @Override
            public void execute(Runnable command) {
                backgroundHandler.post(command);
            }
        };
    }

    private void stopBackgroundThread() {
        if (backgroundThread == null) {
            return;
        }
        backgroundThread.quitSafely();
        try {
            backgroundThread.join();
        } catch (InterruptedException ignored) {
            Thread.currentThread().interrupt();
        }
        backgroundThread = null;
        backgroundHandler = null;
    }

    @Override
    public void onRequestPermissionsResult(int code, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(code, permissions, results);
        if (code == PERMISSION_REQUEST && previewView.isAvailable()) {
            openCamera();
        }
    }
}
