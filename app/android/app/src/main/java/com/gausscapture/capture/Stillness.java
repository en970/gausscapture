package com.gausscapture.capture;

/**
 * Whether the phone is actually at rest, measured rather than assumed.
 *
 * <p>The 4D protocol's second half is only worth recording if the camera does not move: every
 * downstream step gives all of its frames one single pose, and a camera that drifted by a
 * millimetre turns into subject motion that never happened. "The operator put the phone down" is
 * not evidence of that. A phone leaning against a mug slides; a table transmits footsteps; letting
 * go of a phone leaves it oscillating for the better part of a second.
 *
 * <p>So rest is defined as two simultaneous conditions holding continuously:
 *
 * <ul>
 *   <li><b>Angular</b> — bias-corrected gyroscope magnitude below {@link #GYRO_REST_RAD_S}. This
 *       is the sensitive one; a gyro sees a rotation of a hundredth of a degree per second.
 *   <li><b>Linear</b> — the deviation of accelerometer magnitude from the gravity it settled on,
 *       below {@link #ACCEL_REST_M_S2}. A pure translation is invisible to a gyroscope, and a
 *       phone sliding down a slope is exactly that.
 * </ul>
 *
 * <p>The gyro bias matters more than it looks. A consumer MEMS gyro reads a few hundredths of a
 * radian per second while sitting perfectly still, and it is a different few hundredths after
 * every temperature change. A fixed threshold against the raw magnitude would either declare a
 * moving phone still or never confirm rest at all, depending on the device and the hour. The bias
 * is therefore estimated from the samples already judged angularly quiet and subtracted.
 *
 * <p>None of these thresholds has been measured on hardware. They are derived from what the
 * sensors can resolve and from the tilt a reconstruction can tolerate, and the first real take is
 * a measurement of them rather than a confirmation. Every take records what the detector actually
 * saw, so the numbers can be revised from data instead of from opinion.
 *
 * <p>Deliberately free of Android imports: samples are fed in, so the whole detector runs in a
 * plain JVM test.
 */
final class Stillness {

    /**
     * Angular rest threshold, in radians per second — about 1.7 degrees per second.
     *
     * <p>Held for the 700 ms rest confirmation this integrates to roughly 1.2 degrees, which at a
     * 500 mm subject distance is 10 mm of apparent motion. That is a ceiling, not a target: a
     * phone genuinely at rest sits an order of magnitude below it, and the margin exists so that
     * gyro noise on a bad device cannot prevent the protocol from ever advancing.
     */
    static final float GYRO_REST_RAD_S = 0.03f;

    /** Linear rest threshold, in m/s². Below the noise of a hand and above that of a table. */
    static final float ACCEL_REST_M_S2 = 0.25f;

    /**
     * How far the gravity direction may drift from the reference before rest is a lie.
     *
     * <p>One degree of tilt at 500 mm moves the image by roughly 9 mm of subject, which is far
     * more than the reconstruction can absorb silently. This is the check that catches a slow
     * slide the rate thresholds would each pass individually.
     */
    static final float MAX_TILT_DEG = 1.0f;

    /** Samples in each rolling window. At 400 Hz this is about 160 ms. */
    private static final int WINDOW = 64;

    /** How quickly the low-passed gravity estimate follows the accelerometer. */
    private static final float GRAVITY_ALPHA = 0.05f;

    /** How quickly the gyro bias estimate follows quiet samples. Deliberately very slow. */
    private static final float BIAS_ALPHA = 0.002f;

    /** Guards the residual integration against a stalled sensor producing one enormous step. */
    private static final long MAX_STEP_NS = 20_000_000L;

    private final float[] gyroWindow = new float[WINDOW];
    private int gyroIndex;
    private int gyroFilled;

    private final float[] accelWindow = new float[WINDOW];
    private int accelIndex;
    private int accelFilled;

    private final float[] bias = new float[3];
    private final float[] gravity = new float[3];
    private boolean gravitySeeded;

    /** The gravity direction rest was confirmed at, or null before {@link #markReference()}. */
    private float[] reference;

    private volatile float gyroRms;
    private volatile float accelRms;
    private volatile float tiltDeg;
    private volatile float maxTiltDeg;
    private volatile boolean atRest;
    private volatile long restSinceNs = -1;
    private volatile long lastNs;
    private volatile boolean disturbed;

    /**
     * Total rotation the gyroscope integrated since the reference was marked, in radians.
     *
     * <p>Distinct from {@link #maxTiltDeg}, and both are recorded because they fail differently.
     * The tilt is measured against gravity, so it is blind to rotation <em>about</em> gravity — a
     * phone that yaws on a smooth surface keeps a perfect tilt reading. The integral sees that, and
     * in exchange it accumulates bias, so it can report a degree of rotation that never happened.
     * Neither is the published answer; the desktop verifies the pose photometrically and publishes
     * a residual in pixels. These two are what say whether it is worth trying.
     */
    private volatile double residualRad;
    private long residualLastNs = -1;
    private boolean integrating;

    /** Forget everything: a new take is a new device state and a new temperature. */
    synchronized void reset() {
        gyroIndex = 0;
        gyroFilled = 0;
        accelIndex = 0;
        accelFilled = 0;
        bias[0] = 0;
        bias[1] = 0;
        bias[2] = 0;
        gravitySeeded = false;
        reference = null;
        gyroRms = 0;
        accelRms = 0;
        tiltDeg = 0;
        maxTiltDeg = 0;
        atRest = false;
        restSinceNs = -1;
        lastNs = 0;
        disturbed = false;
        residualRad = 0;
        residualLastNs = -1;
        integrating = false;
    }

    synchronized void onGyro(long tNs, float x, float y, float z) {
        // Only samples that are already quiet feed the bias estimate. Otherwise a slow deliberate
        // sweep would be absorbed into the bias and then subtracted out of existence.
        //
        // And the estimate stops entirely once the reference is marked, which is the same instant
        // the residual starts integrating. By then the bias is learned, and a bias that keeps
        // adapting through the hold is a bias that absorbs exactly the slow steady drift the
        // residual exists to detect: measured here, a 1.1 deg/s yaw was reported at 0.73 degrees
        // instead of 1.15 because more than a third of it had been relabelled as bias.
        float raw = magnitude(x - bias[0], y - bias[1], z - bias[2]);
        if (!integrating && raw < GYRO_REST_RAD_S * 2f) {
            bias[0] += BIAS_ALPHA * (x - bias[0]);
            bias[1] += BIAS_ALPHA * (y - bias[1]);
            bias[2] += BIAS_ALPHA * (z - bias[2]);
        }
        float corrected = magnitude(x - bias[0], y - bias[1], z - bias[2]);
        if (integrating) {
            long step = residualLastNs < 0 ? 0 : tNs - residualLastNs;
            if (step > 0 && step <= MAX_STEP_NS) {
                residualRad += corrected * (step / 1_000_000_000.0);
            }
            residualLastNs = tNs;
        }
        gyroWindow[gyroIndex] = corrected;
        gyroIndex = (gyroIndex + 1) % WINDOW;
        if (gyroFilled < WINDOW) {
            gyroFilled++;
        }
        gyroRms = rms(gyroWindow, gyroFilled);
        settle(tNs);
    }

    synchronized void onAccel(long tNs, float x, float y, float z) {
        if (!gravitySeeded) {
            gravity[0] = x;
            gravity[1] = y;
            gravity[2] = z;
            gravitySeeded = true;
        } else {
            gravity[0] += GRAVITY_ALPHA * (x - gravity[0]);
            gravity[1] += GRAVITY_ALPHA * (y - gravity[1]);
            gravity[2] += GRAVITY_ALPHA * (z - gravity[2]);
        }
        // Deviation from the low-passed magnitude rather than from 9.81: the point is change, and
        // an uncalibrated accelerometer can be a percent off without anything being wrong.
        accelWindow[accelIndex] = Math.abs(magnitude(x, y, z) - magnitude(gravity));
        accelIndex = (accelIndex + 1) % WINDOW;
        if (accelFilled < WINDOW) {
            accelFilled++;
        }
        accelRms = rms(accelWindow, accelFilled);

        if (reference != null) {
            tiltDeg = angleDegrees(gravity, reference);
            if (tiltDeg > maxTiltDeg) {
                maxTiltDeg = tiltDeg;
            }
            if (tiltDeg > MAX_TILT_DEG) {
                disturbed = true;
            }
        }
        settle(tNs);
    }

    private void settle(long tNs) {
        lastNs = tNs;
        boolean quiet = gyroFilled > 0 && accelFilled > 0
                && gyroRms < GYRO_REST_RAD_S && accelRms < ACCEL_REST_M_S2;
        if (quiet) {
            if (restSinceNs < 0) {
                restSinceNs = tNs;
            }
        } else {
            restSinceNs = -1;
            if (reference != null) {
                // Rest was confirmed and has now been lost. The take can carry on, but it can
                // never again claim the camera did not move.
                disturbed = true;
            }
        }
        atRest = quiet;
    }

    /**
     * Freeze the current gravity direction as the pose the take claims not to have left.
     *
     * <p>Called when rest is confirmed at the end of the transition, so every later tilt is
     * measured against the orientation the dynamic phase was actually shot from.
     */
    synchronized void markReference() {
        if (!gravitySeeded) {
            return;
        }
        reference = new float[]{gravity[0], gravity[1], gravity[2]};
        tiltDeg = 0;
        maxTiltDeg = 0;
        disturbed = false;
        residualRad = 0;
        residualLastNs = -1;
        integrating = true;
    }

    boolean atRest() {
        return atRest;
    }

    /** How long the phone has been continuously at rest, in milliseconds. Zero when it is not. */
    long restForMillis() {
        long since = restSinceNs;
        return since < 0 ? 0 : Math.max(0, (lastNs - since) / 1_000_000L);
    }

    /**
     * True once rest was confirmed and then lost, or the tilt exceeded its budget.
     *
     * <p>A latch rather than a live reading: the interesting question afterwards is whether the
     * camera moved at any point, not whether it is moving now. It is the difference between a
     * take that can be presented as fixed-camera and one that must not be.
     */
    boolean disturbed() {
        return disturbed;
    }

    float gyroRms() {
        return gyroRms;
    }

    float accelRms() {
        return accelRms;
    }

    float tiltDeg() {
        return tiltDeg;
    }

    float maxTiltDeg() {
        return maxTiltDeg;
    }

    /** Integrated gyro rotation since the reference was marked, in degrees. See {@link #residualRad}. */
    float residualDeg() {
        return (float) Math.toDegrees(residualRad);
    }

    boolean hasReference() {
        return reference != null;
    }

    /** One row of {@code stillness.jsonl}: what the detector saw, so its thresholds can be revised. */
    void appendRow(StringBuilder out, long tNs, String phase) {
        out.setLength(0);
        out.append("{\"t_ns\":").append(tNs)
                .append(",\"phase\":\"").append(phase)
                .append("\",\"gyro_rms_rad_s\":").append(gyroRms)
                .append(",\"accel_rms_m_s2\":").append(accelRms)
                .append(",\"tilt_deg\":").append(tiltDeg)
                .append(",\"at_rest\":").append(atRest)
                .append(",\"rest_for_ms\":").append(restForMillis())
                .append('}');
    }

    // ------------------------------------------------------------------ maths

    private static float magnitude(float x, float y, float z) {
        return (float) Math.sqrt(x * x + y * y + z * z);
    }

    private static float magnitude(float[] v) {
        return magnitude(v[0], v[1], v[2]);
    }

    private static float rms(float[] window, int filled) {
        if (filled == 0) {
            return 0f;
        }
        double sum = 0;
        for (int i = 0; i < filled; i++) {
            sum += window[i] * window[i];
        }
        return (float) Math.sqrt(sum / filled);
    }

    /** Angle between two vectors, in degrees. Zero when either has no length. */
    static float angleDegrees(float[] a, float[] b) {
        float na = magnitude(a);
        float nb = magnitude(b);
        if (na <= 0 || nb <= 0) {
            return 0f;
        }
        double cos = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (double) (na * nb);
        cos = Math.max(-1.0, Math.min(1.0, cos));
        return (float) Math.toDegrees(Math.acos(cos));
    }
}
