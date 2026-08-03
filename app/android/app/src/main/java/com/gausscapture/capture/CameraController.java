package com.gausscapture.capture;

import android.content.Context;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
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
import android.graphics.Rect;
import android.media.MediaRecorder;
import android.os.Build;
import android.util.Range;
import android.util.Size;
import android.util.SizeF;
import android.view.Surface;
import android.view.TextureView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Executor;

/**
 * Owns the camera, the encoder, and the per-frame metadata.
 *
 * <p>Everything here exists to make a capture reconstructable rather than merely pretty:
 *
 * <ul>
 *   <li><b>Exposure, white balance and focus locked</b> for the take. Auto algorithms drift and
 *       break the brightness-constancy assumption reconstruction relies on. Correcting that
 *       afterwards is worth several dB; not breaking it costs nothing.
 *   <li><b>Stabilisation off</b>, optical and digital. Both warp each frame independently, which
 *       invalidates a single shared intrinsic model.
 *   <li><b>One physical lens</b>, chosen once. The stock camera switches between ultra-wide, wide
 *       and telephoto as you zoom, changing intrinsics mid-capture.
 * </ul>
 */
public final class CameraController {

    public interface Listener {
        void onReady(String summary);
        void onError(String message);
        /** Per-frame exposure, so guidance can convert angular rate into blur pixels. */
        void onFrameMetadata(long exposureNanos, int iso);
    }

    private static final int TARGET_WIDTH = 1920;
    private static final int TARGET_HEIGHT = 1080;
    private static final int TARGET_FPS = 30;
    /** Generous: compression artefacts are a confound we would rather not introduce. */
    private static final int VIDEO_BITRATE = 24_000_000;

    /**
     * The longest exposure a take is allowed to use, one 120th of a second.
     *
     * <p>Auto-exposure optimises for a pleasant-looking image, which indoors means roughly 1/30 s.
     * Walking an orbit at a normal pace is about 0.35 radians per second, and at the focal length
     * this phone reports -- near 1350 pixels -- that is {@code 0.35 * 0.033 * 1350}, or about 22
     * pixels of smear. The motion warning would fire on every indoor take and be right to.
     *
     * <p>Capping exposure and letting sensitivity take up the slack trades noise for sharpness,
     * which is the correct direction here: feature matching survives grain far better than it
     * survives smeared corners, and a reconstruction is built out of corners.
     */
    private static final long MAX_EXPOSURE_NS = 8_333_333L;

    /**
     * How far sensitivity may be pushed to pay for the shorter exposure.
     *
     * <p>Past this, noise starts costing more matches than the smear it bought back, so the
     * exposure is allowed to lengthen again rather than producing an unusable image. When that
     * happens the manifest says so instead of implying the cap held.
     */
    private static final int MAX_COMPENSATING_ISO = 3200;

    private final Context context;
    private final TextureView view;
    private final CameraManager manager;
    private final Listener listener;

    private CameraDevice device;
    private CameraCaptureSession session;
    private MediaRecorder recorder;
    private String cameraId;
    private CameraCharacteristics characteristics;
    private int sensorOrientation = 90;
    private Size videoSize = new Size(TARGET_WIDTH, TARGET_HEIGHT);

    private JsonlWriter frameWriter;
    /**
     * SENSOR_TIMESTAMP of the first frame the encoder accepted.
     *
     * <p>For a camera feeding a MediaRecorder surface this value <em>is</em> the presentation
     * timestamp the muxer writes, so recording it anchors frames.jsonl to video.mp4 exactly.
     * Without it the two are matched by ordinal, which drifts by about two frames in an
     * unpredictable direction (audit D03).
     */
    private volatile Long firstEncodedFrameNs;
    private final StringBuilder frameLine = new StringBuilder(256);
    private volatile int frameCount;
    private volatile long lastExposureNanos;
    private volatile int lastIso;
    private Float lockedFocusDistance;
    private volatile boolean recording;

    public CameraController(Context context, TextureView view, CameraManager manager,
                            Listener listener) {
        this.context = context;
        this.view = view;
        this.manager = manager;
        this.listener = listener;
    }

    public int frameCount() {
        return frameCount;
    }

    public long lastExposureNanos() {
        return lastExposureNanos;
    }

    public Size videoSize() {
        return videoSize;
    }

    public String cameraId() {
        return cameraId;
    }

    public int sensorOrientation() {
        return sensorOrientation;
    }

    public CameraCharacteristics characteristics() {
        return characteristics;
    }

    /**
     * Focal length expressed in pixels of the recorded frame.
     *
     * <p>The device reports intrinsics against its full pre-correction active array, but we record
     * a scaled and cropped region of it, so the reported focal length has to be rescaled before it
     * means anything in recorded pixels. Guidance needs this to turn angular rate into blur.
     * Returns 0 when the device does not report intrinsics.
     */
    public float focalPixels() {
        if (characteristics == null) {
            return 0f;
        }
        float[] calibration = characteristics.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION);
        Rect active = characteristics.get(
                CameraCharacteristics.SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE);
        if (calibration == null || calibration.length < 1 || active == null || active.width() == 0) {
            return 0f;
        }
        // Widescreen video crops the sensor vertically, not horizontally, so width scales cleanly.
        return calibration[0] * ((float) videoSize.getWidth() / active.width());
    }

    // ---------------------------------------------------------------- open / close

    public void open(int displayRotation) {
        try {
            cameraId = selectBackCamera();
            if (cameraId == null) {
                listener.onError("No back camera found");
                return;
            }
            characteristics = manager.getCameraCharacteristics(cameraId);
            Integer orientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
            sensorOrientation = orientation == null ? 90 : orientation;

            StreamConfigurationMap map = characteristics.get(
                    CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            if (map != null) {
                videoSize = chooseSize(map.getOutputSizes(MediaRecorder.class));
            }

            manager.openCamera(cameraId, new CameraDevice.StateCallback() {
                @Override
                public void onOpened(CameraDevice camera) {
                    device = camera;
                    startPreview(displayRotation);
                }

                @Override
                public void onDisconnected(CameraDevice camera) {
                    // A take in flight is now unrecoverable, and saying so is the difference
                    // between losing one capture and believing a broken file is fine (audit D02).
                    if (recording && recorderProblem != null) {
                        recorderProblem.onRecorderProblem(
                                "The camera was taken by another app", true);
                    }
                    camera.close();
                    device = null;
                }

                @Override
                public void onError(CameraDevice camera, int error) {
                    if (recording && recorderProblem != null) {
                        recorderProblem.onRecorderProblem("The camera failed (" + error + ")", true);
                    }
                    camera.close();
                    device = null;
                    listener.onError("Camera error " + error);
                }
            }, handler);
        } catch (CameraAccessException | SecurityException e) {
            listener.onError("Cannot open camera: " + e.getMessage());
        }
    }

    public void close() {
        if (session != null) {
            session.close();
            session = null;
        }
        if (device != null) {
            device.close();
            device = null;
        }
        if (recorder != null) {
            recorder.release();
            recorder = null;
        }
    }

    private android.os.Handler handler;

    public void setHandler(android.os.Handler handler) {
        this.handler = handler;
    }

    private Executor executor() {
        return new Executor() {
            @Override
            public void execute(Runnable command) {
                handler.post(command);
            }
        };
    }

    /**
     * Chooses the main back camera and stays on it.
     *
     * <p>Ultra-wide sits near 2mm and telephoto near 7mm or beyond, so the main lens is the one
     * nearest 5mm.
     */
    private String selectBackCamera() throws CameraAccessException {
        String best = null;
        float bestScore = Float.MAX_VALUE;
        for (String id : manager.getCameraIdList()) {
            CameraCharacteristics c = manager.getCameraCharacteristics(id);
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

    // ---------------------------------------------------------------- preview

    public void startPreview(int displayRotation) {
        if (device == null) {
            return;
        }
        try {
            Surface previewSurface = newPreviewSurface(displayRotation);
            if (previewSurface == null) {
                return;
            }
            CaptureRequest.Builder builder =
                    device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            builder.addTarget(previewSurface);
            applyCaptureSettings(builder, false);

            List<OutputConfiguration> outputs =
                    Collections.singletonList(new OutputConfiguration(previewSurface));
            device.createCaptureSession(new SessionConfiguration(
                    SessionConfiguration.SESSION_REGULAR, outputs, executor(),
                    new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured(CameraCaptureSession configured) {
                            session = configured;
                            try {
                                configured.setRepeatingRequest(builder.build(), metadataCallback,
                                        handler);
                                listener.onReady(summary());
                            } catch (CameraAccessException | IllegalStateException e) {
                                listener.onError("Preview failed: " + e.getMessage());
                            }
                        }

                        @Override
                        public void onConfigureFailed(CameraCaptureSession configured) {
                            listener.onError("Preview could not be configured");
                        }
                    }));
        } catch (CameraAccessException e) {
            listener.onError("Preview error: " + e.getMessage());
        }
    }

    /** Surfaces handed to a session, released when it is replaced. Never released before: D19. */
    private final List<Surface> liveSurfaces = new ArrayList<>();

    private void releaseSurfaces() {
        for (Surface surface : liveSurfaces) {
            surface.release();
        }
        liveSurfaces.clear();
    }

    private Surface newPreviewSurface(int displayRotation) {
        SurfaceTexture texture = view.getSurfaceTexture();
        if (texture == null) {
            // onSurfaceTextureDestroyed released it, and the stop-then-restart-preview path can
            // still get here afterwards (audit D18).
            return null;
        }
        texture.setDefaultBufferSize(videoSize.getWidth(), videoSize.getHeight());
        applyTransform(displayRotation);
        Surface surface = new Surface(texture);
        liveSurfaces.add(surface);
        return surface;
    }

    /** Degrees the camera image must be rotated to appear upright on screen. */
    public int previewRotation(int displayRotation) {
        int displayDegrees;
        switch (displayRotation) {
            case Surface.ROTATION_90: displayDegrees = 90; break;
            case Surface.ROTATION_180: displayDegrees = 180; break;
            case Surface.ROTATION_270: displayDegrees = 270; break;
            default: displayDegrees = 0; break;
        }
        return (sensorOrientation - displayDegrees + 360) % 360;
    }

    /**
     * Undoes TextureView's stretch, rotates the image upright, then fits it to the view.
     *
     * <p>TextureView always scales its buffer to the view bounds, ignoring aspect ratio, and
     * applies any transform on top. So the transform has three steps, and both their order and the
     * dimensions used at each matter:
     *
     * <ol>
     *   <li>Map the view rect onto the buffer's <em>true</em> dimensions, cancelling the stretch.
     *       Using post-rotation dimensions here squeezes a 16:9 image into a 9:16 box before
     *       rotating, and no later step can recover the aspect ratio.
     *   <li>Rotate.
     *   <li><em>Now</em> the on-screen extent has swapped for a quarter turn, so the fit scale uses
     *       the swapped dimensions.
     * </ol>
     *
     * <p>It fits rather than fills. Filling would centre-crop and hide part of what is being
     * recorded, and the entire job of a capture preview is to show the operator the framing they
     * are committing to.
     */
    public void applyTransform(int displayRotation) {
        int viewWidth = view.getWidth();
        int viewHeight = view.getHeight();
        if (viewWidth == 0 || viewHeight == 0) {
            return;
        }
        int rotate = previewRotation(displayRotation);
        RectF viewRect = new RectF(0, 0, viewWidth, viewHeight);
        float centerX = viewRect.centerX();
        float centerY = viewRect.centerY();

        RectF bufferRect = new RectF(0, 0, videoSize.getWidth(), videoSize.getHeight());
        bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY());

        Matrix matrix = new Matrix();
        matrix.setRectToRect(viewRect, bufferRect, Matrix.ScaleToFit.FILL);
        matrix.postRotate(rotate, centerX, centerY);

        float shownWidth = (rotate % 180 == 0) ? videoSize.getWidth() : videoSize.getHeight();
        float shownHeight = (rotate % 180 == 0) ? videoSize.getHeight() : videoSize.getWidth();
        float scale = Math.min(viewWidth / shownWidth, viewHeight / shownHeight);
        matrix.postScale(scale, scale, centerX, centerY);

        view.setTransform(matrix);
    }

    private void applyCaptureSettings(CaptureRequest.Builder b, boolean lock) {
        b.set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_OFF);
        b.set(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
                CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_OFF);
        b.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);
        b.set(CaptureRequest.CONTROL_SCENE_MODE, CameraMetadata.CONTROL_SCENE_MODE_DISABLED);
        b.set(CaptureRequest.CONTROL_EFFECT_MODE, CameraMetadata.CONTROL_EFFECT_MODE_OFF);
        b.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, new Range<>(TARGET_FPS, TARGET_FPS));

        // Report geometry as captured, so recorded distortion coefficients describe the recorded
        // pixels rather than a corrected version of them.
        if (characteristics.get(CameraCharacteristics.DISTORTION_CORRECTION_AVAILABLE_MODES)
                != null) {
            b.set(CaptureRequest.DISTORTION_CORRECTION_MODE,
                    CameraMetadata.DISTORTION_CORRECTION_MODE_OFF);
        }

        if (lock) {
            b.set(CaptureRequest.CONTROL_AWB_LOCK, true);
            b.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_OFF);
            if (lockedFocusDistance != null) {
                b.set(CaptureRequest.LENS_FOCUS_DISTANCE, lockedFocusDistance);
            }
            if (!applyExposureCap(b)) {
                // No manual control, so the best available is to freeze whatever auto-exposure
                // converged on during the preview. Better than letting it drift mid-take, but it
                // does not bound the smear.
                b.set(CaptureRequest.CONTROL_AE_LOCK, true);
            }
        } else {
            b.set(CaptureRequest.CONTROL_AE_LOCK, false);
            b.set(CaptureRequest.CONTROL_AWB_LOCK, false);
            b.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_VIDEO);
        }
    }

    /** What the exposure cap settled on, for the manifest. Null until a take has started. */
    private Long cappedExposureNs;
    private Integer cappedIso;
    private String exposurePolicy = "auto_locked";

    /**
     * Bound motion blur by taking manual control of exposure.
     *
     * <p>The preview has been running with auto-exposure, so its last converged values are a
     * measurement of the actual light. Those are the starting point: exposure is clipped to the
     * cap and sensitivity is scaled by exactly the factor that was removed, which keeps total
     * light the same until sensitivity runs out of room.
     *
     * @return false when this device offers no manual control, so the caller can fall back.
     */
    private boolean applyExposureCap(CaptureRequest.Builder b) {
        int[] capabilities =
                characteristics.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES);
        boolean manual = false;
        if (capabilities != null) {
            for (int capability : capabilities) {
                if (capability
                        == CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR) {
                    manual = true;
                    break;
                }
            }
        }
        Range<Long> exposureRange =
                characteristics.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE);
        Range<Integer> isoRange =
                characteristics.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE);
        if (!manual || exposureRange == null || isoRange == null) {
            exposurePolicy = "auto_locked";
            return false;
        }

        long observed = lastExposureNanos > 0 ? lastExposureNanos : MAX_EXPOSURE_NS;
        int observedIso = lastIso > 0 ? lastIso : isoRange.getLower();

        if (observed <= MAX_EXPOSURE_NS) {
            // Already short enough; freeze it where it is rather than changing anything.
            cappedExposureNs = clamp(observed, exposureRange);
            cappedIso = clamp(observedIso, isoRange);
            exposurePolicy = "manual_unchanged";
        } else {
            double shortenedBy = observed / (double) MAX_EXPOSURE_NS;
            int wanted = (int) Math.round(observedIso * shortenedBy);
            int ceiling = Math.min(MAX_COMPENSATING_ISO, isoRange.getUpper());

            if (wanted <= ceiling) {
                cappedExposureNs = clamp(MAX_EXPOSURE_NS, exposureRange);
                cappedIso = clamp(wanted, isoRange);
                exposurePolicy = "manual_capped";
            } else {
                // Sensitivity cannot pay for the whole reduction. Spend what it can and let the
                // exposure sit at whatever that affords -- a darker frame is recoverable, and an
                // absurdly noisy one is not.
                cappedIso = clamp(ceiling, isoRange);
                long affordable = (long) (observed * (observedIso / (double) ceiling));
                cappedExposureNs = clamp(affordable, exposureRange);
                exposurePolicy = "manual_iso_limited";
            }
        }

        b.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF);
        b.set(CaptureRequest.SENSOR_EXPOSURE_TIME, cappedExposureNs);
        b.set(CaptureRequest.SENSOR_SENSITIVITY, cappedIso);
        // Frame duration has to leave room for the exposure, or the HAL quietly lengthens one of
        // the two and the request no longer describes what is recorded.
        b.set(CaptureRequest.SENSOR_FRAME_DURATION,
                Math.max(1_000_000_000L / TARGET_FPS, cappedExposureNs));
        return true;
    }

    private static long clamp(long value, Range<Long> range) {
        return Math.max(range.getLower(), Math.min(range.getUpper(), value));
    }

    private static int clamp(int value, Range<Integer> range) {
        return Math.max(range.getLower(), Math.min(range.getUpper(), value));
    }

    /** What the exposure policy decided, so the manifest records it rather than assuming it. */
    public JSONObject exposureSettings() throws org.json.JSONException {
        JSONObject json = new JSONObject();
        json.put("policy", exposurePolicy);
        json.put("max_exposure_ns", MAX_EXPOSURE_NS);
        json.put("exposure_ns", cappedExposureNs == null ? JSONObject.NULL : cappedExposureNs);
        json.put("iso", cappedIso == null ? JSONObject.NULL : cappedIso);
        // The smear this policy admits at a normal orbit rate, stated so a capture can be judged
        // without redoing the arithmetic.
        float focal = focalPixels();
        if (focal > 0 && cappedExposureNs != null) {
            json.put("blur_px_at_0p35_rad_s",
                    0.35f * (cappedExposureNs / 1_000_000_000f) * focal);
        }
        return json;
    }

    /** Records what the camera actually did, per frame, rather than what we asked for. */
    private final CameraCaptureSession.CaptureCallback metadataCallback =
            new CameraCaptureSession.CaptureCallback() {
                @Override
                public void onCaptureCompleted(CameraCaptureSession s, CaptureRequest request,
                                               TotalCaptureResult result) {
                    Float focus = result.get(CaptureResult.LENS_FOCUS_DISTANCE);
                    if (focus != null) {
                        lockedFocusDistance = focus;
                    }
                    Long exposure = result.get(CaptureResult.SENSOR_EXPOSURE_TIME);
                    Integer iso = result.get(CaptureResult.SENSOR_SENSITIVITY);
                    if (exposure != null) {
                        lastExposureNanos = exposure;
                    }
                    if (iso != null) {
                        lastIso = iso;
                    }
                    listener.onFrameMetadata(lastExposureNanos, lastIso);

                    if (!recording) {
                        return;
                    }
                    Long timestamp = result.get(CaptureResult.SENSOR_TIMESTAMP);
                    if (firstEncodedFrameNs == null && timestamp != null) {
                        firstEncodedFrameNs = timestamp;
                    }
                    try {
                        JsonlWriter writer = frameWriter;
                        if (writer == null) {
                            return;
                        }
                        frameLine.setLength(0);
                        // Three identifiers, because each answers a question the others cannot.
                        // t_ns is the presentation timestamp the muxer writes, so it locates this
                        // row in the video exactly. capture_frame is the HAL's own counter: gaps
                        // in it are the only way to detect a frame dropped *mid-take*, which an
                        // anchor at the start cannot see. `frame` is the ordinal of rows written
                        // here, kept because the previous format used it.
                        frameLine.append("{\"t_ns\":").append(timestamp)
                                .append(",\"capture_frame\":").append(result.getFrameNumber())
                                .append(",\"frame\":").append(frameCount);
                        appendNumber(frameLine, "iso", iso);
                        appendNumber(frameLine, "exposure_ns", exposure);
                        appendNumber(frameLine, "focus_distance", focus);
                        appendNumber(frameLine, "aperture", result.get(CaptureResult.LENS_APERTURE));
                        appendNumber(frameLine, "focal_length_mm",
                                result.get(CaptureResult.LENS_FOCAL_LENGTH));

                        // Everything below is recorded because the manifest used to *assert* it.
                        // The crop region is the authoritative answer to how the sensor array maps
                        // onto the recorded frame, and it is what makes the focal-length rescale
                        // correct rather than an assumption about which axis was cropped (D23).
                        Rect crop = result.get(CaptureResult.SCALER_CROP_REGION);
                        if (crop != null) {
                            frameLine.append(",\"crop\":[").append(crop.left).append(',')
                                    .append(crop.top).append(',').append(crop.right).append(',')
                                    .append(crop.bottom).append(']');
                        }
                        appendNumber(frameLine, "ae_state", result.get(CaptureResult.CONTROL_AE_STATE));
                        appendNumber(frameLine, "awb_state", result.get(CaptureResult.CONTROL_AWB_STATE));
                        appendNumber(frameLine, "lens_state", result.get(CaptureResult.LENS_STATE));
                        appendNumber(frameLine, "rolling_shutter_skew_ns",
                                result.get(CaptureResult.SENSOR_ROLLING_SHUTTER_SKEW));

                        // On a logical multi-camera -- which camera 0 is on an S22 -- the HAL may
                        // switch physical lenses in low light with no zoom change, silently
                        // invalidating the single intrinsic model the whole capture assumes (D21).
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                            String physical = result.get(
                                    CaptureResult.LOGICAL_MULTI_CAMERA_ACTIVE_PHYSICAL_ID);
                            if (physical != null) {
                                frameLine.append(",\"physical_id\":\"").append(physical).append('"');
                            }
                        }
                        frameLine.append('}');
                        writer.append(frameLine);
                        frameCount++;
                    } catch (RuntimeException ignored) {
                        // A dropped metadata row must never interrupt a recording.
                    }
                }
            };

    /** Append {@code ,"name":value}, or nothing at all when the device did not report it. */
    private static void appendNumber(StringBuilder out, String name, Number value) {
        if (value != null) {
            out.append(",\"").append(name).append("\":").append(value);
        }
    }

    /** True once the encoder has produced at least one frame we can anchor to. */
    public Long firstEncodedFrameNs() {
        return firstEncodedFrameNs;
    }

    public long framesDropped() {
        JsonlWriter writer = frameWriter;
        return writer == null ? 0 : writer.linesDropped();
    }

    // ---------------------------------------------------------------- recording

    public interface RecordingCallback {
        void onStarted();
        void onFailed(String message);
    }

    public void startRecording(File sessionDir, int displayRotation, RecordingCallback callback) {
        if (device == null) {
            callback.onFailed("Camera is not ready");
            return;
        }
        try {
            frameCount = 0;
            firstEncodedFrameNs = null;
            frameWriter = new JsonlWriter(new File(sessionDir, "frames.jsonl"));

            if (session != null) {
                session.close();
                session = null;
            }

            recorder = buildRecorder(new File(sessionDir, "video.mp4"), displayRotation);
            recorder.prepare();

            Surface previewSurface = newPreviewSurface(displayRotation);
            if (previewSurface == null) {
                callback.onFailed("The preview is not available yet");
                return;
            }
            Surface recorderSurface = recorder.getSurface();

            CaptureRequest.Builder builder =
                    device.createCaptureRequest(CameraDevice.TEMPLATE_RECORD);
            builder.addTarget(previewSurface);
            builder.addTarget(recorderSurface);
            // Lock now: the preview has been running, so the auto algorithms have converged and
            // the values they settled on are the ones we freeze.
            applyCaptureSettings(builder, true);

            List<OutputConfiguration> outputs = Arrays.asList(
                    new OutputConfiguration(previewSurface),
                    new OutputConfiguration(recorderSurface));

            device.createCaptureSession(new SessionConfiguration(
                    SessionConfiguration.SESSION_REGULAR, outputs, executor(),
                    new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured(CameraCaptureSession configured) {
                            session = configured;
                            try {
                                configured.setRepeatingRequest(builder.build(), metadataCallback,
                                        handler);
                                recorder.start();
                                recording = true;
                                callback.onStarted();
                            } catch (Exception e) {
                                callback.onFailed("Could not start: " + e.getMessage());
                            }
                        }

                        @Override
                        public void onConfigureFailed(CameraCaptureSession configured) {
                            callback.onFailed("Recording session could not be configured");
                        }
                    }));
        } catch (Exception e) {
            callback.onFailed("Could not start: " + e.getMessage());
        }
    }

    /** @return true when the encoder produced a usable file. */
    public boolean stopRecording() {
        recording = false;
        boolean ok = true;
        try {
            if (recorder != null) {
                recorder.stop();
            }
        } catch (RuntimeException e) {
            ok = false;  // Thrown when the take was shorter than a single frame.
        } finally {
            if (recorder != null) {
                recorder.reset();
                recorder.release();
                recorder = null;
            }
        }
        JsonlWriter writer = frameWriter;
        frameWriter = null;
        if (writer != null) {
            writer.close();
        }
        return ok;
    }

    /**
     * Signalled when the encoder stops for a reason the operator cannot see: a full card, the 4 GB
     * MPEG-4 limit at roughly 22 minutes, or a muxer fault. None of these used to reach anyone --
     * encoding simply halted while the timer kept counting (audit D05).
     */
    public interface RecorderProblem {
        void onRecorderProblem(String message, boolean fatal);
    }

    private RecorderProblem recorderProblem;

    public void setRecorderProblem(RecorderProblem listener) {
        this.recorderProblem = listener;
    }

    @SuppressWarnings("deprecation")
    private MediaRecorder buildRecorder(File output, int displayRotation) {
        // The Context constructor is API 31; minSdk is 28. javac against android.jar does not
        // enforce API levels, so the old unconditional call compiled cleanly and threw
        // NoSuchMethodError on the first tap of record on anything older (audit D01).
        MediaRecorder r = Build.VERSION.SDK_INT >= Build.VERSION_CODES.S
                ? new MediaRecorder(context)
                : new MediaRecorder();
        r.setVideoSource(MediaRecorder.VideoSource.SURFACE);
        r.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
        r.setOutputFile(output.getAbsolutePath());
        r.setVideoEncodingBitRate(VIDEO_BITRATE);
        r.setVideoFrameRate(TARGET_FPS);
        r.setVideoSize(videoSize.getWidth(), videoSize.getHeight());
        r.setVideoEncoder(MediaRecorder.VideoEncoder.H264);
        // Without this the file is born rotated and every downstream tool inherits the mistake.
        r.setOrientationHint(previewRotation(displayRotation));

        r.setOnErrorListener(new MediaRecorder.OnErrorListener() {
            @Override
            public void onError(MediaRecorder mr, int what, int extra) {
                recording = false;
                if (recorderProblem != null) {
                    recorderProblem.onRecorderProblem("The encoder stopped (" + what + ")", true);
                }
            }
        });
        r.setOnInfoListener(new MediaRecorder.OnInfoListener() {
            @Override
            public void onInfo(MediaRecorder mr, int what, int extra) {
                if (what == MediaRecorder.MEDIA_RECORDER_INFO_MAX_FILESIZE_REACHED
                        || what == MediaRecorder.MEDIA_RECORDER_INFO_MAX_DURATION_REACHED) {
                    recording = false;
                    if (recorderProblem != null) {
                        recorderProblem.onRecorderProblem("Recording hit its size limit", true);
                    }
                }
            }
        });
        // Stop just short of where MPEG-4 stops being able to address the file, so the take ends
        // with a valid trailer instead of being truncated mid-write.
        r.setMaxFileSize(3_800_000_000L);
        return r;
    }

    // ---------------------------------------------------------------- metadata

    public String summary() {
        float[] focals = characteristics.get(
                CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS);
        Integer source = characteristics.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE);
        boolean realtime = source != null
                && source == CameraMetadata.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME;
        return String.format(java.util.Locale.US,
                "cam %s · %.1fmm · %dx%d@%d · locked · %s",
                cameraId, focals == null || focals.length == 0 ? 0f : focals[0],
                videoSize.getWidth(), videoSize.getHeight(), TARGET_FPS,
                realtime ? "imu synced" : "imu offset recorded");
    }

    public boolean timestampsAligned() {
        Integer source = characteristics == null ? null
                : characteristics.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE);
        return source != null
                && source == CameraMetadata.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME;
    }

    /** Camera intrinsics as the device reports them, rather than as SfM would guess them. */
    public JSONObject intrinsics() throws Exception {
        JSONObject out = new JSONObject();
        out.put("source", "android_camera2_characteristics");
        out.put("camera_id", cameraId);

        float[] calibration = characteristics.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION);
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
            out.put("focal_pixels_recorded", focalPixels());
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

        Rect active = characteristics.get(
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
        out.put("recorded_size", videoSize.getWidth() + "x" + videoSize.getHeight());
        return out;
    }
}
