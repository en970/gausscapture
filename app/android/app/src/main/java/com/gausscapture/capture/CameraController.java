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
import android.hardware.camera2.params.ColorSpaceTransform;
import android.hardware.camera2.params.OutputConfiguration;
import android.hardware.camera2.params.RggbChannelVector;
import android.hardware.camera2.params.SessionConfiguration;
import android.hardware.camera2.params.StreamConfigurationMap;
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
 *
 * <p><b>One session for the whole take, including a scripted one.</b> The 4D protocol records
 * several phases into a single file, and the photometric lock is applied once, when that file
 * starts, and never touched again. That is the point rather than a shortcut: the phase that
 * contains the geometry and the phase that contains the motion have to share one exposure, one
 * white balance and one focus distance, or the canonical Gaussians are fitted to one appearance
 * and supervised against another. Reconfiguring the session between phases — the obvious way to
 * write this — would silently reintroduce every difference the protocol exists to remove, and
 * would also cost a 100–500 ms hole in the recording at the least forgiving moment.
 */
public final class CameraController {

    public interface Listener {
        void onReady(String summary);
        void onError(String message);
        /** Per-frame exposure, so guidance can convert angular rate into blur pixels. */
        void onFrameMetadata(long exposureNanos, int iso);
    }

    /**
     * Something happened that invalidates an assumption the manifest would otherwise assert.
     *
     * <p>A capture that quietly switched physical lenses at t = 18 s, or whose white balance came
     * unlocked halfway through, is worse than no capture: it looks fine and is not reconstructable
     * under one model. Each kind is reported once — repeating it every frame would drown the file
     * it is written into.
     */
    public interface Anomaly {
        void onAnomaly(String event, String detail, long tNs);
    }

    private Anomaly anomaly;

    public void setAnomalyListener(Anomaly listener) {
        this.anomaly = listener;
    }

    private void report(String event, String detail, long tNs) {
        Anomaly listener = anomaly;
        if (listener != null) {
            listener.onAnomaly(event, detail, tNs);
        }
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

    /**
     * Which lens faces the scene. The 4D protocol films the operator's own face, so it is the only
     * preset that does not use the back camera, and that choice has to be made before the device
     * is opened rather than after.
     */
    private String lensFacing = "back";

    /** The phase every frame from now belongs to. Written into each row of frames.jsonl. */
    private volatile String phase = "free";

    /** What the desktop splits the recording on. Appended on the camera thread, read at stop. */
    private final List<PhaseSpan> phaseSpans = new ArrayList<>();

    /**
     * The frames the camera was measurably at rest for, immediately before the dynamic phase.
     *
     * <p>These are static scene content from exactly the pose the dynamic phase is shot from,
     * which is what makes them the tier-one source for the fixed pose: they can be registered
     * against the rest of the reconstruction without any masking, and the pose they solve to is
     * the pose every dynamic frame is then given.
     */
    private volatile int restWindowFrameFirst = -1;
    private volatile int restWindowFrameLast = -1;
    private volatile long restWindowCaptureFirst = -1;
    private volatile long restWindowCaptureLast = -1;

    /** One contiguous run of frames belonging to one phase. */
    private static final class PhaseSpan {
        final String id;
        final int frameFirst;
        int frameLast;
        final long captureFrameFirst;
        long captureFrameLast;
        final long tNsFirst;
        long tNsLast;

        PhaseSpan(String id, int frame, long captureFrame, long tNs) {
            this.id = id;
            this.frameFirst = frame;
            this.frameLast = frame;
            this.captureFrameFirst = captureFrame;
            this.captureFrameLast = captureFrame;
            this.tNsFirst = tNs;
            this.tNsLast = tNs;
        }
    }

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

    /** Choose the lens before opening. Ignored once the device is open. */
    public void setLensFacing(String facing) {
        this.lensFacing = "front".equals(facing) ? "front" : "back";
    }

    public String lensFacing() {
        return lensFacing;
    }

    /**
     * Tag every subsequent frame with this phase.
     *
     * <p>The tag is written per row rather than derived from a timestamp afterwards because the
     * boundaries matter to a frame: the last frame of the transition was shot while the phone was
     * still being let go of, and the first frame of the dynamic phase was not. A boundary
     * reconstructed on the desktop from wall-clock times would be a frame or two out, in an
     * unknown direction, exactly where it is least affordable.
     */
    public void setPhase(String phase) {
        this.phase = phase;
    }

    public String phase() {
        return phase;
    }

    /**
     * The camera is now measurably at rest at the pose the dynamic phase will be shot from.
     *
     * <p>{@code frameCount} is the index of the <em>next</em> row frames.jsonl will hold, and that
     * frame is still inside the transition phase because the phase has not changed yet -- so it is
     * the first frame of the rest window and this is correct as written.
     */
    public void markRestWindowStart() {
        restWindowFrameFirst = frameCount;
        restWindowCaptureFirst = lastCaptureFrame;
    }

    /**
     * The dynamic phase has begun, so the rest window ended with the frame before it.
     *
     * <p>{@code frameCount - 1}, not {@code frameCount}. Both bounds are inclusive in the schema
     * and both are consumed inclusively -- {@code ingest/phases.py} iterates
     * {@code range(start, end + 1)} -- and the caller is {@code CaptureEngine.onPhaseChanged},
     * which sets the new phase before it marks the window. So {@code frameCount} names a frame
     * that will be tagged with the dynamic phase, and declaring it here handed one frame of the
     * subject moving to the noise-floor measurement and to the fixed-pose solve as if it were
     * evidence about a still scene.
     */
    public void markRestWindowEnd() {
        restWindowFrameLast = Math.max(restWindowFrameFirst, frameCount - 1);
        restWindowCaptureLast = lastCaptureFrame;
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
            cameraId = selectCamera(lensFacing);
            if (cameraId == null) {
                listener.onError("No " + lensFacing + " camera found");
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
     * Chooses one lens and stays on it.
     *
     * <p>On the back, ultra-wide sits near 2 mm and telephoto near 7 mm or beyond, so the main
     * lens is the one nearest 5 mm. On the front there is usually only one camera, and where a
     * phone offers two the second is a wider one meant for group selfies: the longer focal length
     * is the right choice for a face at arm's length, because it distorts it less and because
     * ultra-wides carry far more of the distortion the intrinsic model has to absorb.
     */
    private String selectCamera(String facing) throws CameraAccessException {
        int wanted = "front".equals(facing)
                ? CameraCharacteristics.LENS_FACING_FRONT
                : CameraCharacteristics.LENS_FACING_BACK;
        boolean preferLongest = "front".equals(facing);
        String best = null;
        float bestScore = Float.MAX_VALUE;
        for (String id : manager.getCameraIdList()) {
            CameraCharacteristics c = manager.getCameraCharacteristics(id);
            Integer lens = c.get(CameraCharacteristics.LENS_FACING);
            if (lens == null || lens != wanted) {
                continue;
            }
            float[] focals = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS);
            if (focals == null || focals.length == 0) {
                continue;
            }
            float score = preferLongest ? -focals[0] : Math.abs(focals[0] - 5.0f);
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

        // Antibanding quantises exposure to multiples of the mains period -- 10 ms on a 50 Hz
        // supply -- which alone defeats any short-exposure policy, and it does so by changing the
        // exposure as the scene brightness changes. Off here and off for the whole take.
        b.set(CaptureRequest.CONTROL_AE_ANTIBANDING_MODE,
                CameraMetadata.CONTROL_AE_ANTIBANDING_MODE_OFF);

        // Fixed rather than off: these pipeline stages cannot always be disabled, but they can be
        // stopped from switching mode between the preview and the recording, or between a bright
        // sweep and a dim one. A stage that changes mode mid-take changes the appearance of a
        // static surface without anything in the scene having moved.
        b.set(CaptureRequest.NOISE_REDUCTION_MODE, CameraMetadata.NOISE_REDUCTION_MODE_FAST);
        b.set(CaptureRequest.EDGE_MODE, CameraMetadata.EDGE_MODE_FAST);
        b.set(CaptureRequest.SHADING_MODE, CameraMetadata.SHADING_MODE_FAST);
        // Zoom is the other way the crop region moves under the intrinsics. Never set
        // SCALER_CROP_REGION; state the ratio instead, so the result's crop is the HAL's own.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            b.set(CaptureRequest.CONTROL_ZOOM_RATIO, 1.0f);
        }

        if (lock) {
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
            applyWhiteBalanceLock(b);
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
    private String whiteBalancePolicy = "auto_locked";

    /** What auto white balance converged on during the preview, and what we then hold fixed. */
    private volatile RggbChannelVector lastColorGains;
    private volatile ColorSpaceTransform lastColorTransform;

    /**
     * Hold white balance still for the whole take, by value where the device allows it.
     *
     * <p>{@code CONTROL_AWB_LOCK} freezes the algorithm's <em>decision</em>, which is enough for a
     * single stretch of recording. A scripted take is harder: the phone is carried around the
     * subject and back, so the field of view fills with a window, then a wall, then a face, and a
     * HAL that honours the lock loosely — several do — has every excuse to re-converge. Where
     * {@code MANUAL_POST_PROCESSING} exists, the gains the preview settled on are therefore
     * written into the request as numbers, which nothing can re-converge.
     *
     * <p>Locking by value matters more than it sounds. The canonical Gaussians are fitted to the
     * phase that has parallax and then supervised against the phase that has motion; a white
     * balance shift between the two is indistinguishable, to the fitter, from the subject changing
     * colour.
     */
    private void applyWhiteBalanceLock(CaptureRequest.Builder b) {
        RggbChannelVector gains = lastColorGains;
        ColorSpaceTransform transform = lastColorTransform;
        if (hasCapability(CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_POST_PROCESSING)
                && gains != null && transform != null) {
            b.set(CaptureRequest.CONTROL_AWB_MODE, CameraMetadata.CONTROL_AWB_MODE_OFF);
            b.set(CaptureRequest.COLOR_CORRECTION_MODE,
                    CameraMetadata.COLOR_CORRECTION_MODE_TRANSFORM_MATRIX);
            b.set(CaptureRequest.COLOR_CORRECTION_GAINS, gains);
            b.set(CaptureRequest.COLOR_CORRECTION_TRANSFORM, transform);
            whiteBalancePolicy = "manual_gains";
        } else {
            b.set(CaptureRequest.CONTROL_AWB_LOCK, true);
            whiteBalancePolicy = "auto_locked";
        }
    }

    private boolean hasCapability(int capability) {
        int[] capabilities =
                characteristics.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES);
        if (capabilities == null) {
            return false;
        }
        for (int candidate : capabilities) {
            if (candidate == capability) {
                return true;
            }
        }
        return false;
    }

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
        boolean manual =
                hasCapability(CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR);
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
                    // Read during the preview, written back into the recording request. This is
                    // the only way to lock white balance by value rather than by decision.
                    RggbChannelVector gains = result.get(CaptureResult.COLOR_CORRECTION_GAINS);
                    if (gains != null) {
                        lastColorGains = gains;
                    }
                    ColorSpaceTransform transform =
                            result.get(CaptureResult.COLOR_CORRECTION_TRANSFORM);
                    if (transform != null) {
                        lastColorTransform = transform;
                    }
                    listener.onFrameMetadata(lastExposureNanos, lastIso);

                    if (!recording) {
                        return;
                    }
                    Long timestamp = result.get(CaptureResult.SENSOR_TIMESTAMP);
                    if (firstEncodedFrameNs == null && timestamp != null) {
                        firstEncodedFrameNs = timestamp;
                    }
                    lastCaptureFrame = result.getFrameNumber();
                    verify(result, timestamp == null ? 0L : timestamp);
                    try {
                        JsonlWriter writer = frameWriter;
                        if (writer == null) {
                            return;
                        }
                        String currentPhase = phase;
                        recordPhase(currentPhase, frameCount, result.getFrameNumber(),
                                timestamp == null ? 0L : timestamp);

                        frameLine.setLength(0);
                        // Three identifiers, because each answers a question the others cannot.
                        // t_ns is the presentation timestamp the muxer writes, so it locates this
                        // row in the video exactly. capture_frame is the HAL's own counter: gaps
                        // in it are the only way to detect a frame dropped *mid-take*, which an
                        // anchor at the start cannot see. `frame` is the ordinal of rows written
                        // here, kept because the previous format used it.
                        frameLine.append("{\"t_ns\":").append(timestamp)
                                .append(",\"capture_frame\":").append(result.getFrameNumber())
                                .append(",\"frame\":").append(frameCount)
                                .append(",\"phase\":\"").append(currentPhase).append('"');
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

    /** The HAL's own frame counter for the last frame seen, which is what phase bounds quote. */
    private volatile long lastCaptureFrame = -1;

    private String firstPhysicalId;
    private Rect firstCrop;
    private boolean reportedLensSwitch;
    private boolean reportedExposureDrift;
    private boolean reportedWhiteBalanceDrift;
    private boolean reportedCropChange;

    /**
     * How many recorded frames were checked against the requested exposure, and how many agreed.
     *
     * <p>{@code events.jsonl} records the <em>first</em> frame at which a lock stopped holding,
     * which answers "did it break". It cannot answer "how much of the take was affected", and a
     * HAL that wobbles for three frames out of two thousand and one that abandoned the lock
     * entirely look identical in that file. The ratio is the difference between the two.
     */
    private volatile int locksChecked;
    private volatile int locksMatched;

    /**
     * Check each frame against what the take claims about itself.
     *
     * <p>The manifest used to <em>assert</em> that exposure, white balance and the lens were held
     * for the duration. A Samsung HAL honours some of that and quietly declines the rest. These
     * four checks turn every one of those assertions into either a confirmation or a line in
     * {@code events.jsonl}, and the scripted protocol needs them more than a single-stretch take
     * does: it is carried around a room and back, which is precisely the excursion that talks a
     * 3A algorithm into re-converging.
     */
    private void verify(TotalCaptureResult result, long tNs) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            String physical = result.get(CaptureResult.LOGICAL_MULTI_CAMERA_ACTIVE_PHYSICAL_ID);
            if (physical != null) {
                if (firstPhysicalId == null) {
                    firstPhysicalId = physical;
                } else if (!firstPhysicalId.equals(physical) && !reportedLensSwitch) {
                    reportedLensSwitch = true;
                    report("lens_switch", firstPhysicalId + " -> " + physical, tNs);
                }
            }
        }

        Long exposure = result.get(CaptureResult.SENSOR_EXPOSURE_TIME);
        if (cappedExposureNs != null && exposure != null) {
            locksChecked++;
            if (Math.abs(exposure - cappedExposureNs) > cappedExposureNs / 50) {
                if (!reportedExposureDrift) {
                    reportedExposureDrift = true;
                    report("exposure_drift",
                            "requested " + cappedExposureNs + " ns, got " + exposure, tNs);
                }
            } else {
                locksMatched++;
            }
        }

        Integer awbMode = result.get(CaptureResult.CONTROL_AWB_MODE);
        if ("manual_gains".equals(whiteBalancePolicy) && awbMode != null
                && awbMode != CameraMetadata.CONTROL_AWB_MODE_OFF && !reportedWhiteBalanceDrift) {
            reportedWhiteBalanceDrift = true;
            report("awb_not_manual", "requested OFF, got mode " + awbMode, tNs);
        }

        Rect crop = result.get(CaptureResult.SCALER_CROP_REGION);
        if (crop != null) {
            if (firstCrop == null) {
                firstCrop = crop;
            } else if (!firstCrop.equals(crop) && !reportedCropChange) {
                reportedCropChange = true;
                report("crop_changed", firstCrop.flattenToString() + " -> "
                        + crop.flattenToString(), tNs);
            }
        }
    }

    /** Extend the current phase's run of frames, or open a new one when the phase has changed. */
    private void recordPhase(String id, int frame, long captureFrame, long tNs) {
        synchronized (phaseSpans) {
            PhaseSpan last = phaseSpans.isEmpty() ? null : phaseSpans.get(phaseSpans.size() - 1);
            if (last == null || !last.id.equals(id)) {
                phaseSpans.add(new PhaseSpan(id, frame, captureFrame, tNs));
                return;
            }
            last.frameLast = frame;
            last.captureFrameLast = captureFrame;
            last.tNsLast = tNs;
        }
    }

    /**
     * The frame bounds of each phase, which is what the desktop splits the recording on.
     *
     * <p>Both indices are published because they answer different questions. {@code frame} is the
     * ordinal among rows written here, which is how a row is matched to a video sample;
     * {@code capture_frame} is the HAL's own counter, whose gaps are the only evidence that a
     * frame was dropped mid-take.
     */
    public JSONArray phaseSpansJson() throws org.json.JSONException {
        JSONArray out = new JSONArray();
        synchronized (phaseSpans) {
            for (PhaseSpan span : phaseSpans) {
                JSONObject row = new JSONObject();
                row.put("id", span.id);
                row.put("frame_first", span.frameFirst);
                row.put("frame_last", span.frameLast);
                row.put("capture_frame_first", span.captureFrameFirst);
                row.put("capture_frame_last", span.captureFrameLast);
                row.put("t_ns_first", span.tNsFirst);
                row.put("t_ns_last", span.tNsLast);
                row.put("frames", span.frameLast - span.frameFirst + 1);
                out.put(row);
            }
        }
        return out;
    }

    /** The frames the camera was confirmed at rest for, or null when it never was. */
    public JSONObject restWindowJson() throws org.json.JSONException {
        if (restWindowFrameFirst < 0) {
            return null;
        }
        JSONObject out = new JSONObject();
        out.put("frame_first", restWindowFrameFirst);
        // Inclusive, like `frame_first`: the last frame written, not the next one.
        out.put("frame_last",
                restWindowFrameLast < 0 ? Math.max(restWindowFrameFirst, frameCount - 1)
                                        : restWindowFrameLast);
        out.put("capture_frame_first", restWindowCaptureFirst);
        out.put("capture_frame_last",
                restWindowCaptureLast < 0 ? lastCaptureFrame : restWindowCaptureLast);
        return out;
    }

    /**
     * Everything held fixed for the take, and how firmly.
     *
     * <p>Named for what it is: a photometric contract between the phase that has parallax and the
     * phase that has motion. The values are what was requested; {@code frames.jsonl} carries what
     * each frame actually did, and {@code events.jsonl} carries the first frame at which the two
     * stopped agreeing.
     */
    public JSONObject photometricSettings() throws org.json.JSONException {
        JSONObject out = new JSONObject();
        // Field names are the capturepack schema's, not this class's, because this object is
        // published verbatim as the manifest's `photometric` block and a reader should not have to
        // know which Android class produced it.
        out.put("exposure_time_ns", cappedExposureNs == null ? JSONObject.NULL : cappedExposureNs);
        out.put("sensitivity_iso", cappedIso == null ? JSONObject.NULL : cappedIso);
        out.put("frame_duration_ns", cappedExposureNs == null
                ? JSONObject.NULL
                : Math.max(1_000_000_000L / TARGET_FPS, cappedExposureNs));
        out.put("white_balance_mode", whiteBalancePolicy);
        out.put("focus_distance_dioptres", lockedFocusDistance == null
                ? JSONObject.NULL : (double) lockedFocusDistance);
        RggbChannelVector gains = lastColorGains;
        if (gains == null) {
            out.put("colour_correction_gains", JSONObject.NULL);
        } else {
            JSONArray array = new JSONArray();
            array.put(gains.getRed());
            array.put(gains.getGreenEven());
            array.put(gains.getGreenOdd());
            array.put(gains.getBlue());
            out.put("colour_correction_gains", array);
        }
        out.put("noise_reduction_mode", "fast");
        out.put("edge_mode", "fast");
        out.put("antibanding_mode", "off");
        out.put("locks_verified_ratio",
                locksChecked == 0 ? JSONObject.NULL : locksMatched / (double) locksChecked);

        // Beyond the schema, and kept because they say what the ratio above cannot: which policy
        // produced these numbers, and where to look when it did not hold.
        out.put("exposure_policy", exposurePolicy);
        out.put("frames_lock_checked", locksChecked);
        out.put("shading_mode", "fast");
        out.put("zoom_ratio", 1.0);
        out.put("locked_once_for_all_phases", true);
        out.put("verified_per_frame_in", "frames.jsonl");
        out.put("deviations_in", "events.jsonl");
        return out;
    }

    /** Fraction of checked frames that must agree before a lock counts as held. */
    private static final double LOCKS_VERIFIED_MIN = 0.999;

    /**
     * The lock state under the four names the capturepack schema publishes.
     *
     * <p>These are what the desktop reads. {@code pack/manifest.py} warns when
     * {@code exposure_locked} is not {@code true}, and {@code telemetry/report.py} tells the
     * operator to "lock exposure before starting the capture" on the strength of the same key.
     * The manifest used to carry only {@code exposure_lock_requested} and its three siblings;
     * the schema's {@code additionalProperties: true} meant the file still validated, the keys
     * the readers wanted were simply absent, and so <em>every</em> take from this app was
     * reported as having no locks at all -- with the advice to go and turn on the thing that
     * was already on.
     *
     * <p>Requested and achieved are different questions, so both are written. A value here is
     * what the sensor came back with: {@code null} where the HAL never reported enough to say.
     */
    public JSONObject achievedLocks() throws org.json.JSONException {
        JSONObject out = new JSONObject();
        // Nine hundred and ninety-nine frames in a thousand: a HAL that wobbles for two frames
        // held the lock; one that re-converged halfway through did not, and the ratio is the
        // only thing that tells them apart.
        out.put("exposure_locked", locksChecked == 0
                ? JSONObject.NULL
                : locksMatched >= LOCKS_VERIFIED_MIN * locksChecked);
        // Either policy holds white balance still for the take -- by value where the device has
        // manual post-processing, by CONTROL_AWB_LOCK otherwise. What breaks it is the HAL
        // re-converging anyway, which `verify` records.
        out.put("white_balance_locked", !reportedWhiteBalanceDrift);
        out.put("focus_locked", lockedFocusDistance != null);
        // Requested off in applyCaptureSettings for every take, unconditionally. Android exposes
        // no result key that says the HAL honoured it, so this is the request and says so by
        // sitting beside `stabilisation_disable_requested` rather than replacing it.
        out.put("stabilisation_disabled", true);
        return out;
    }


    /** The physical lens the first recorded frame came from, or null when the HAL never said. */
    public String activePhysicalId() {
        return firstPhysicalId;
    }

    /**
     * What this camera can actually do, dumped once per session.
     *
     * <p>Costs a few kilobytes and permanently removes guesswork about what a given phone
     * populates — which keys exist, which lens is really behind the logical id, and whether the
     * timestamps are on the same clock as the sensors.
     */
    public JSONObject deviceReport() throws org.json.JSONException {
        JSONObject out = new JSONObject();
        out.put("camera_id", cameraId);
        out.put("lens_facing", lensFacing);
        Integer level = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);
        out.put("hardware_level", level == null ? JSONObject.NULL : level);
        Integer source = characteristics.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE);
        out.put("timestamp_source", source != null
                && source == CameraMetadata.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME
                ? "REALTIME" : "UNKNOWN");
        int[] capabilities =
                characteristics.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES);
        JSONArray caps = new JSONArray();
        if (capabilities != null) {
            for (int capability : capabilities) {
                caps.put(capability);
            }
        }
        out.put("capabilities", caps);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            JSONArray physical = new JSONArray();
            for (String id : characteristics.getPhysicalCameraIds()) {
                physical.put(id);
            }
            out.put("physical_ids", physical);
        }
        out.put("recorded_size", videoSize.getWidth() + "x" + videoSize.getHeight());
        out.put("sensor_orientation", sensorOrientation);
        return out;
    }

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

    /**
     * Frame metadata rows the writer had to drop, which is the manifest's `frame_rows_dropped`.
     *
     * <p>Read after the writer has been closed and discarded, so the final count is kept when it is
     * closed rather than fetched from an object that no longer exists. It used to be fetched, from
     * a field {@code stopRecording} had already set to null — so every manifest ever written
     * reported zero dropped rows, including the ones where rows were dropped.
     */
    public long framesDropped() {
        JsonlWriter writer = frameWriter;
        return writer == null ? frameRowsDropped : writer.linesDropped();
    }

    /** Why frame metadata stopped being written, or null. Same reasoning as {@link #framesDropped}. */
    public String frameWriteError() {
        JsonlWriter writer = frameWriter;
        return writer == null ? frameWriteFailure : writer.failure();
    }

    private volatile long frameRowsDropped;
    private volatile String frameWriteFailure;

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
            frameRowsDropped = 0;
            frameWriteFailure = null;
            firstEncodedFrameNs = null;
            lastCaptureFrame = -1;
            firstPhysicalId = null;
            firstCrop = null;
            reportedLensSwitch = false;
            reportedExposureDrift = false;
            reportedWhiteBalanceDrift = false;
            reportedCropChange = false;
            locksChecked = 0;
            locksMatched = 0;
            restWindowFrameFirst = -1;
            restWindowFrameLast = -1;
            restWindowCaptureFirst = -1;
            restWindowCaptureLast = -1;
            synchronized (phaseSpans) {
                phaseSpans.clear();
            }
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
            // Kept before the reference is let go, because the manifest is written after this call
            // and these two numbers are the only account of what the metadata stream actually did.
            frameRowsDropped = writer.linesDropped();
            frameWriteFailure = writer.failure();
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
