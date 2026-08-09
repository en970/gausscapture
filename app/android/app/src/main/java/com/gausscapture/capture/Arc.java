package com.gausscapture.capture;

/**
 * How far the sweep actually went, and whether it was a sweep at all.
 *
 * <p>The arc is the only part of a fixed-camera take that contains geometry. Everything the
 * deliverable can honestly show — the width of the view cone, the shape of the head, the fact that
 * there is a head rather than a billboard — comes from those twelve seconds. So the app measures
 * two things about it while it happens, and records both whether or not they were good.
 *
 * <p><b>Swept angle.</b> Yaw about gravity, integrated from the gyroscope. Only the component of
 * angular velocity along the gravity direction counts: a phone tipped up and down sweeps no
 * horizontal baseline, and counting it would let a nod pass as an orbit. The angle reported is the
 * full range covered, not the net rotation, because a sweep out and back covers its range twice
 * and nets zero.
 *
 * <p><b>Orbit versus pan.</b> Rotating on the spot produces zero parallax no matter how far it
 * goes, and structure from motion is undefined there — this project's preset D exists to
 * demonstrate exactly that. Walking a curved path accelerates the phone sideways; pivoting does
 * not. So the discriminator is the component of measured acceleration <em>perpendicular to
 * gravity</em>, and it only works if the gravity direction comes from an orientation estimate.
 * Low-passing the accelerometer to find gravity cannot do it: the centripetal acceleration of an
 * orbit is nearly constant in the device frame, because the camera keeps pointing at the centre,
 * so any filter slow enough to be stable absorbs exactly the signal being looked for. Without a
 * rotation vector the question is therefore reported as unmeasured rather than answered wrongly.
 *
 * <p>Integrating a gyro drifts, and over twelve seconds a typical bias of 0.01 rad/s accumulates
 * about 7 degrees. That is why the swept angle is a capture-time guide and never the published
 * number: the cone the viewer clamps to is measured offline from the solved cameras, and the
 * manifest's job is only to let the two be compared.
 *
 * <p>Deliberately free of Android imports, so the integration runs in a plain JVM test.
 */
final class Arc {

    /**
     * Sign convention, in one place so it can be flipped after one hardware take.
     *
     * <p>Yaw is integrated about the measured gravity direction, which points from the sky towards
     * the ground in device coordinates. A positive integrated angle is a rotation the operator
     * experiences as swinging the phone towards their right. The maths is unambiguous; which way
     * it feels is not, and the only thing that depends on it is which edge of the screen a
     * directional cue lights up.
     */
    private static final float DIRECTION_SIGN = -1f;

    /** Below this angular rate nothing is a sweep, and integrating noise is worse than not. */
    private static final float SWEEP_RAD_S = 0.05f;

    /** Sideways acceleration above which the path is curving rather than pivoting, in m/s². */
    private static final float ORBIT_M_S2 = 0.15f;

    /** Guards against a stalled sensor producing one enormous integration step. */
    private static final long MAX_STEP_NS = 20_000_000L;

    /** Fallback gravity tracker, used only for the yaw axis when no rotation vector exists. */
    private static final float GRAVITY_ALPHA = 0.05f;

    /**
     * How much of the narrower half-sweep is claimed as a view cone.
     *
     * <p>The sweep is measured at the hand; the cone the viewer clamps to is measured offline from
     * the cameras structure-from-motion actually solved, and that set is always a subset — the ends
     * of a sweep are where the operator is turning around, where the blur is worst and where the
     * overlap with the rest of the arc is thinnest. This factor is the phone's own admission of
     * that, so the number it publishes can only ever be narrower than the truth, never wider.
     */
    private static final float CONE_SAFETY = 0.8f;

    /**
     * Bin width and count for the rate histogram, covering 0 to 128 degrees per second.
     *
     * <p>A histogram rather than a sample buffer because the median is wanted at the end of a
     * twelve-second sweep sampled at 400 Hz, which is five thousand floats to keep and sort for
     * one number. Half a degree per second is finer than the quantity is meaningful to.
     */
    private static final float RATE_BIN_DEG_S = 0.5f;
    private static final int RATE_BINS = 256;

    private final float[] lowPassGravity = new float[3];
    private boolean lowPassSeeded;

    /** Unit vector along gravity in device coordinates, from the rotation vector when there is one. */
    private final float[] gravityDir = new float[3];
    private boolean haveDirection;
    private boolean fromRotationVector;

    private long lastGyroNs;
    private double heading;
    private double minHeading;
    private double maxHeading;
    private boolean running;

    private final int[] rateHistogram = new int[RATE_BINS];
    private int rateSamples;
    private long firstSweepNs = -1;
    private long lastSweepNs = -1;

    private volatile int orbitSamples;
    private volatile int panSamples;
    private volatile float headingDeg;
    private volatile float sweptDeg;
    private volatile float leftDeg;
    private volatile float rightDeg;

    /** Begin integrating from zero. Called when the arc phase starts, not when recording does. */
    synchronized void begin() {
        lastGyroNs = 0;
        heading = 0;
        minHeading = 0;
        maxHeading = 0;
        orbitSamples = 0;
        panSamples = 0;
        headingDeg = 0;
        sweptDeg = 0;
        leftDeg = 0;
        rightDeg = 0;
        running = true;
        java.util.Arrays.fill(rateHistogram, 0);
        rateSamples = 0;
        firstSweepNs = -1;
        lastSweepNs = -1;
    }

    /** Stop integrating. The measurements stay readable for the manifest. */
    synchronized void end() {
        running = false;
    }

    /**
     * The platform's own orientation estimate, which is what makes the orbit test possible.
     *
     * <p>Takes the quaternion in Android's {@code TYPE_GAME_ROTATION_VECTOR} layout — three
     * components, with the scalar either supplied fourth or recovered from unit length — and reads
     * the gravity direction straight out of the rotation matrix's third row, which is where the
     * world's up axis lands once expressed in device coordinates.
     */
    synchronized void onRotation(float[] values) {
        if (values == null || values.length < 3) {
            return;
        }
        float x = values[0];
        float y = values[1];
        float z = values[2];
        float w;
        if (values.length >= 4) {
            w = values[3];
        } else {
            float remainder = 1f - (x * x + y * y + z * z);
            w = remainder > 0 ? (float) Math.sqrt(remainder) : 0f;
        }
        // Third row of the device-to-world rotation, negated: world up expressed in device axes,
        // pointing down, which is the direction the accelerometer's gravity component opposes.
        float gx = -(2f * (x * z - w * y));
        float gy = -(2f * (y * z + w * x));
        float gz = -(1f - 2f * (x * x + y * y));
        float norm = magnitude(gx, gy, gz);
        if (norm <= 0) {
            return;
        }
        gravityDir[0] = gx / norm;
        gravityDir[1] = gy / norm;
        gravityDir[2] = gz / norm;
        haveDirection = true;
        fromRotationVector = true;
    }

    synchronized void onAccel(long tNs, float x, float y, float z) {
        if (!lowPassSeeded) {
            lowPassGravity[0] = x;
            lowPassGravity[1] = y;
            lowPassGravity[2] = z;
            lowPassSeeded = true;
        } else {
            lowPassGravity[0] += GRAVITY_ALPHA * (x - lowPassGravity[0]);
            lowPassGravity[1] += GRAVITY_ALPHA * (y - lowPassGravity[1]);
            lowPassGravity[2] += GRAVITY_ALPHA * (z - lowPassGravity[2]);
        }
        if (!fromRotationVector) {
            float norm = magnitude(lowPassGravity);
            if (norm > 0) {
                // Note the sign: the accelerometer reads the reaction to gravity, so at rest it
                // points away from the ground. The yaw axis is the same line either way.
                gravityDir[0] = -lowPassGravity[0] / norm;
                gravityDir[1] = -lowPassGravity[1] / norm;
                gravityDir[2] = -lowPassGravity[2] / norm;
                haveDirection = true;
            }
        }
        if (!running || !fromRotationVector) {
            return;
        }
        // Everything the accelerometer reads across gravity is the path curving. Along gravity is
        // gravity itself plus whatever vertical bob a walk has, and neither says anything useful.
        float along = x * gravityDir[0] + y * gravityDir[1] + z * gravityDir[2];
        float px = x - along * gravityDir[0];
        float py = y - along * gravityDir[1];
        float pz = z - along * gravityDir[2];
        if (magnitude(px, py, pz) > ORBIT_M_S2) {
            orbitSamples++;
        } else {
            panSamples++;
        }
    }

    synchronized void onGyro(long tNs, float x, float y, float z) {
        if (!running || !haveDirection) {
            lastGyroNs = tNs;
            return;
        }
        long previous = lastGyroNs;
        lastGyroNs = tNs;
        if (previous == 0) {
            return;
        }
        long step = tNs - previous;
        if (step <= 0 || step > MAX_STEP_NS) {
            return;
        }
        // Angular velocity projected onto the gravity direction: yaw, and nothing else.
        double yawRate = x * gravityDir[0] + y * gravityDir[1] + z * gravityDir[2];
        if (Math.abs(yawRate) < SWEEP_RAD_S) {
            return;
        }
        // Only sweeping samples are binned. A sweep that pauses halfway is still a sweep, and
        // counting the pause would report a median rate the operator never moved at.
        int bin = (int) (Math.abs(Math.toDegrees(yawRate)) / RATE_BIN_DEG_S);
        rateHistogram[Math.max(0, Math.min(RATE_BINS - 1, bin))]++;
        rateSamples++;
        if (firstSweepNs < 0) {
            firstSweepNs = tNs;
        }
        lastSweepNs = tNs;

        heading += yawRate * (step / 1_000_000_000.0);
        minHeading = Math.min(minHeading, heading);
        maxHeading = Math.max(maxHeading, heading);

        headingDeg = (float) Math.toDegrees(heading) * DIRECTION_SIGN;
        sweptDeg = (float) Math.toDegrees(maxHeading - minHeading);
        float positive = (float) Math.toDegrees(maxHeading);
        float negative = (float) Math.toDegrees(-minHeading);
        rightDeg = DIRECTION_SIGN > 0 ? positive : negative;
        leftDeg = DIRECTION_SIGN > 0 ? negative : positive;
    }

    /** Total range covered, in degrees. The number the arc is judged against. */
    float sweptDeg() {
        return sweptDeg;
    }

    /** Where the phone is pointing now, relative to where the arc began. */
    float headingDeg() {
        return headingDeg;
    }

    float leftDeg() {
        return leftDeg;
    }

    float rightDeg() {
        return rightDeg;
    }

    /**
     * How far the sweep ended from where it started.
     *
     * <p>Large means the phone was not brought back, so the reseat is unlikely to have landed in
     * the original pose — worth recording, because the desktop verifies exactly that.
     */
    float returnErrorDeg() {
        return Math.abs(headingDeg);
    }

    /**
     * True when the path looked like an orbit, and true by default when it cannot be judged.
     *
     * <p>Silence is not evidence of an orbit; it is the absence of evidence of a pivot. Warning an
     * operator who is doing the right thing is worse than staying quiet, and the manifest records
     * {@link #orbitMeasured()} so nobody downstream mistakes a default for a measurement.
     */
    boolean orbitLikely() {
        return !fromRotationVector || orbitSamples > panSamples;
    }

    /** Whether the orbit-versus-pan question was answerable at all on this device. */
    boolean orbitMeasured() {
        return fromRotationVector;
    }

    /**
     * Half-angle the phone believes the sweep earned, on its narrower side.
     *
     * <p>A symmetric cone can only be as wide as its narrower half, and this is a hand measurement
     * discounted by {@link #CONE_SAFETY}. It is never the published cone: the deliverable clamps to
     * a cone measured from the solved cameras, and this number exists so the two can be compared.
     */
    float impliedConeDeg() {
        return Math.min(leftDeg, rightDeg) * CONE_SAFETY;
    }

    /** Typical sweep rate while actually sweeping, in degrees per second. Zero when there was none. */
    float medianRateDegPerSec() {
        if (rateSamples == 0) {
            return 0f;
        }
        int half = rateSamples / 2;
        int seen = 0;
        for (int bin = 0; bin < RATE_BINS; bin++) {
            seen += rateHistogram[bin];
            if (seen > half) {
                return (bin + 0.5f) * RATE_BIN_DEG_S;
            }
        }
        return (RATE_BINS - 0.5f) * RATE_BIN_DEG_S;
    }

    /** Seconds between the first and last sweeping sample. Zero when nothing was swept. */
    float durationSec() {
        if (firstSweepNs < 0 || lastSweepNs <= firstSweepNs) {
            return 0f;
        }
        return (lastSweepNs - firstSweepNs) / 1_000_000_000f;
    }

    int orbitSamples() {
        return orbitSamples;
    }

    int panSamples() {
        return panSamples;
    }

    /**
     * Which way to keep sweeping: −1 left, +1 right, 0 when both sides have their share.
     *
     * <p>The arc wants roughly half its target on each side of the starting pose, because the
     * cone is centred on the pose the deliverable is shot from. A sweep that went 60 degrees one
     * way and none the other covers the same angle and produces a cone the fixed camera sits on
     * the edge of.
     */
    int suggestedDirection(int targetDeg) {
        float half = targetDeg / 2f;
        if (leftDeg < half && leftDeg <= rightDeg) {
            return -1;
        }
        if (rightDeg < half) {
            return 1;
        }
        return 0;
    }

    private static float magnitude(float x, float y, float z) {
        return (float) Math.sqrt(x * x + y * y + z * z);
    }

    private static float magnitude(float[] v) {
        return magnitude(v[0], v[1], v[2]);
    }
}
