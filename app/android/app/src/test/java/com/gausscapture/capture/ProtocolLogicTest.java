package com.gausscapture.capture;

/**
 * The parts of the 4D protocol that are decisions rather than platform calls, tested on a laptop.
 *
 * <p>{@link PhaseMachine}, {@link Stillness} and {@link Arc} deliberately import nothing from
 * Android, which is what makes this possible: the script that decides when the transition is over,
 * the detector that decides whether the phone is actually at rest, and the integration that
 * decides how far the sweep went can all be driven from synthesised samples in a plain JVM. Every
 * one of them is a place where a wrong answer produces a take that looks fine and is not
 * reconstructable, so "it worked when I tried it on the phone" is not good enough evidence.
 *
 * <p>No JUnit, because the Flutter Android module declares no test dependencies and adding one to
 * fetch a runner would make this suite need a network. Run it directly:
 *
 * <pre>
 * cd app/android/app/src
 * javac -d /tmp/gc-test main/java/com/gausscapture/capture/{Preset,PhaseMachine,Stillness,Arc}.java \
 *       test/java/com/gausscapture/capture/ProtocolLogicTest.java
 * java -cp /tmp/gc-test com.gausscapture.capture.ProtocolLogicTest
 * </pre>
 */
public final class ProtocolLogicTest {

    private static int failures;
    private static int checks;

    public static void main(String[] args) {
        presetShape();
        happyPath();
        perchNeverSettles();
        sweepTooNarrowIsExtended();
        pivotIsCalledOut();
        countdownRestartsWhenTouched();
        movementDuringHoldEndsTheTake();
        holdRaisesNoWarnings();

        stillnessConfirmsQuiet();
        stillnessLearnsGyroBias();
        stillnessRejectsRotation();
        stillnessLatchesTilt();
        stillnessIntegratesTheHoldResidual();

        arcMeasuresSweep();
        arcRangeNotNetRotation();
        arcSeparatesOrbitFromPivot();
        arcSuggestsTheEmptierSide();
        arcReportsRateAndCone();

        System.out.println(checks + " checks, " + failures + " failed");
        if (failures > 0) {
            System.exit(1);
        }
    }

    // ------------------------------------------------------------------ preset

    private static void presetShape() {
        Preset f = Preset.byId("F_bullet");
        is("F is a script", true, f.isScripted());
        is("F has four phases", 4, f.phases.length);
        is("F films the operator", true, f.usesFrontCamera());
        is("F declares its capture type", "fixed_camera_4d", f.captureType);
        is("exactly one phase asks the subject to move", "hold", f.dynamicPhase().id);
        is("A is one undivided stretch", false, Preset.byId("A_good").isScripted());
        is("A keeps the back camera", "back", Preset.byId("A_good").lensFacing);
        // The panel is a light source on a face at arm's length, so F must not run it at full.
        is("F dims the screen", true, f.screenBrightness < 0.5f);
    }

    // ------------------------------------------------------------ phase machine

    /**
     * An operator, simulated.
     *
     * <p>Driven by the phase the machine is <em>in</em> rather than by wall-clock time, because
     * that is how a real take works: nobody starts sweeping at 2.000 s, they start sweeping when
     * the phone tells them to. Scripting against absolute time would test the test.
     */
    private interface Operator {
        PhaseMachine.Motion sample(String phase, long inPhase);
    }

    private static PhaseMachine.Motion motion(boolean atRest, long restMs, float swept,
                                              boolean orbit) {
        PhaseMachine.Motion m = new PhaseMachine.Motion();
        m.atRest = atRest;
        m.restForMillis = restMs;
        m.sweptDeg = swept;
        m.orbitLikely = orbit;
        return m;
    }

    /** Runs the script to completion, or to {@code limitMs}, and returns when it stopped. */
    private static long run(PhaseMachine machine, Operator operator, long limitMs) {
        long t = 0;
        for (; t <= limitMs && !machine.finished(); t += 50) {
            machine.update(t, operator.sample(machine.phaseId(), machine.phaseElapsedMs()));
        }
        return t;
    }

    /** The operator who does everything right: sweeps 60 degrees, reseats, lets go after 1 s. */
    private static Operator competent() {
        return (phase, inPhase) -> {
            switch (phase) {
                case "arc":
                    return motion(false, 0, Math.min(60f, inPhase / 150f), true);
                case "reseat":
                    return inPhase < 1000
                            ? motion(false, 0, 60, true)
                            : motion(true, inPhase - 1000, 60, true);
                default:
                    return motion(true, inPhase, 0, true);
            }
        };
    }

    private static void happyPath() {
        PhaseMachine machine = new PhaseMachine(Preset.byId("F_bullet"));
        long stopped = run(machine, competent(), 40000);

        is("the take completes rather than aborting", PhaseMachine.REASON_COMPLETE,
                machine.finishedReason());
        is("completion is not an abort", false, machine.aborted());
        is("it ends in the phase the subject moved in", "hold", machine.phaseId());
        is("every phase was visited exactly once", 3, machine.transitions());
        is("rest was confirmed before the subject was asked to move", true,
                machine.restConfirmed());
        // 2 s perched + 12 s sweeping + 1 s of reaching + 0.7 s of confirmed rest + 1.5 s of
        // countdown + 3 s of subject motion.
        near("auto-stop lands where the script says", 20250, stopped, 250);
    }

    private static void perchNeverSettles() {
        PhaseMachine machine = new PhaseMachine(Preset.byId("F_bullet"));
        final String[] warningAtOneSecond = new String[1];
        long t = 0;
        for (; t <= 12000 && !machine.finished(); t += 50) {
            machine.update(t, motion(false, 0, 0, true));
            if (t == 1000) {
                warningAtOneSecond[0] = machine.warning();
            }
        }
        is("an unsteady perch says so at once", "Hands off", warningAtOneSecond[0]);
        is("a phone that never settles ends the take", true, machine.aborted());
        is("and says why", "The phone never settled", machine.finishedReason());
        near("after the phase's own ceiling, not for ever", 8000, t, 100);
    }

    private static void sweepTooNarrowIsExtended() {
        PhaseMachine machine = new PhaseMachine(Preset.byId("F_bullet"));
        final String[] atNominal = new String[2];
        final String[] pastCeiling = new String[1];
        long t = 0;
        for (; t <= 40000 && !machine.finished(); t += 50) {
            String phase = machine.phaseId();
            long inPhase = machine.phaseElapsedMs();
            PhaseMachine.Motion m;
            if ("arc".equals(phase)) {
                m = motion(false, 0, 25f, true);          // never reaches the 40 degree floor
                if (inPhase == 12050) {
                    atNominal[0] = phase;
                    atNominal[1] = machine.warning();
                }
            } else if ("reseat".equals(phase)) {
                m = motion(true, inPhase, 25f, true);
                if (pastCeiling[0] == null) {
                    pastCeiling[0] = phase;
                }
            } else {
                m = motion(true, inPhase, 25f, true);
            }
            machine.update(t, m);
        }
        is("the sweep runs past its nominal length", "arc", atNominal[0]);
        is("and asks for more rather than accepting a cone too narrow to label", "Sweep wider",
                atNominal[1]);
        is("but not for ever: the ceiling takes whatever was swept", "reseat", pastCeiling[0]);
        is("the take still completes", PhaseMachine.REASON_COMPLETE, machine.finishedReason());
    }

    private static void pivotIsCalledOut() {
        PhaseMachine machine = new PhaseMachine(Preset.byId("F_bullet"));
        final String[] warning = new String[1];
        for (long t = 0; t <= 8000 && !machine.finished(); t += 50) {
            String phase = machine.phaseId();
            long inPhase = machine.phaseElapsedMs();
            machine.update(t, "arc".equals(phase)
                    ? motion(false, 0, 40f, false)        // turning on the spot
                    : motion(true, inPhase, 0, true));
            if ("arc".equals(machine.phaseId()) && machine.phaseElapsedMs() > 3500) {
                warning[0] = machine.warning();
            }
        }
        is("zero parallax is the one thing worth interrupting a sweep for", "Walk around it",
                warning[0]);
    }

    private static void countdownRestartsWhenTouched() {
        PhaseMachine machine = new PhaseMachine(Preset.byId("F_bullet"));
        final long[] countdownBeforeNudge = {-1};
        final long[] countdownAfterNudge = {0};
        final String[] phaseAfterNudge = new String[1];

        Operator clumsy = (phase, inPhase) -> {
            if (!"reseat".equals(phase)) {
                return competent().sample(phase, inPhase);
            }
            if (inPhase < 1000) {
                return motion(false, 0, 60, true);                    // still reaching
            }
            if (inPhase >= 1900 && inPhase < 2400) {
                return motion(false, 0, 60, true);                    // nudged it
            }
            long restedSince = inPhase >= 2400 ? 2400 : 1000;
            return motion(true, inPhase - restedSince, 60, true);
        };

        for (long t = 0; t <= 40000 && !machine.finished(); t += 50) {
            machine.update(t, clumsy.sample(machine.phaseId(), machine.phaseElapsedMs()));
            if ("reseat".equals(machine.phaseId())) {
                long inPhase = machine.phaseElapsedMs();
                if (inPhase == 1850) {
                    countdownBeforeNudge[0] = machine.countdownMillis();
                }
                if (inPhase == 2450) {
                    countdownAfterNudge[0] = machine.countdownMillis();
                    phaseAfterNudge[0] = machine.phaseId();
                }
            }
        }
        is("a countdown had started before the nudge", true, countdownBeforeNudge[0] > 0);
        is("touching the phone cancels it", -1L, countdownAfterNudge[0]);
        is("and leaves the operator in the transition", "reseat", phaseAfterNudge[0]);
        is("the take still completes afterwards", PhaseMachine.REASON_COMPLETE,
                machine.finishedReason());
    }

    private static void movementDuringHoldEndsTheTake() {
        PhaseMachine machine = new PhaseMachine(Preset.byId("F_bullet"));
        run(machine, (phase, inPhase) -> {
            if ("hold".equals(phase) && inPhase >= 1000) {
                return motion(false, 0, 60, true);        // the support slips mid-take
            }
            return competent().sample(phase, inPhase);
        }, 40000);

        is("a camera that moved is not silently kept", true, machine.aborted());
        is("and the reason survives into the manifest", "The phone moved",
                machine.finishedReason());
    }

    private static void holdRaisesNoWarnings() {
        PhaseMachine machine = new PhaseMachine(Preset.byId("F_bullet"));
        final boolean[] warned = {false};
        Operator restless = (phase, inPhase) -> {
            if ("hold".equals(phase)) {
                return motion(true, inPhase, 60, false);  // orbit unlikely, sweep short: irrelevant
            }
            return competent().sample(phase, inPhase);
        };
        for (long t = 0; t <= 40000 && !machine.finished(); t += 50) {
            machine.update(t, restless.sample(machine.phaseId(), machine.phaseElapsedMs()));
            if ("hold".equals(machine.phaseId()) && machine.warning() != null) {
                warned[0] = true;
            }
        }
        // A large amber word in front of a face that is being recorded destroys the recording.
        is("nothing shouts at the subject while they are being filmed", false, warned[0]);
    }

    // ---------------------------------------------------------------- stillness

    /** A phone lying flat and quiet: gravity on +z, no rotation, sampled at 400 Hz. */
    private static long feedQuiet(Stillness stillness, long fromNs, int samples) {
        long t = fromNs;
        for (int i = 0; i < samples; i++) {
            stillness.onGyro(t, 0.001f, -0.002f, 0.0015f);
            stillness.onAccel(t, 0.01f, -0.01f, 9.81f);
            t += 2_500_000L;
        }
        return t;
    }

    private static void stillnessConfirmsQuiet() {
        Stillness stillness = new Stillness();
        long t = feedQuiet(stillness, 1_000_000_000L, 400);
        is("a quiet phone reads as at rest", true, stillness.atRest());
        near("and knows how long it has been", 1000, stillness.restForMillis(), 20);
        is("nothing has gone wrong yet", false, stillness.disturbed());
        // 400 samples at 2.5 ms is one second; the first sample is the start of rest.
        is("rest is a duration, not an instant", true, stillness.restForMillis() > 900);
        is("time advanced", true, t > 1_000_000_000L);
    }

    private static void stillnessLearnsGyroBias() {
        Stillness stillness = new Stillness();
        long t = 0;
        // A gyro reading 0.025 rad/s while the phone sits on a table: bias, not motion. Left
        // uncorrected it is 86 % of the rest threshold and rest would never be confirmed on a
        // cold device.
        for (int i = 0; i < 6000; i++) {
            stillness.onGyro(t, 0.025f, 0f, 0f);
            stillness.onAccel(t, 0f, 0f, 9.81f);
            t += 2_500_000L;
        }
        is("a constant offset is learned rather than reported as motion", true, stillness.atRest());
        is("and the residual is near zero", true, stillness.gyroRms() < 0.005f);
    }

    private static void stillnessRejectsRotation() {
        Stillness stillness = new Stillness();
        long t = feedQuiet(stillness, 0, 400);
        is("at rest before", true, stillness.atRest());
        for (int i = 0; i < 200; i++) {
            stillness.onGyro(t, 0f, 0.5f, 0f);            // a hand picking it up
            stillness.onAccel(t, 0f, 0f, 9.81f);
            t += 2_500_000L;
        }
        is("a real rotation is not absorbed as bias", false, stillness.atRest());
        is("and rest starts counting from zero again", 0L, stillness.restForMillis());
    }

    private static void stillnessLatchesTilt() {
        Stillness stillness = new Stillness();
        long t = feedQuiet(stillness, 0, 400);
        stillness.markReference();
        is("a fresh reference is undisturbed", false, stillness.disturbed());
        // Three degrees of lean, arriving slowly enough that neither rate threshold trips: the
        // exact failure the tilt check exists for.
        double lean = Math.toRadians(3);
        float ax = (float) (9.81 * Math.sin(lean));
        float az = (float) (9.81 * Math.cos(lean));
        for (int i = 0; i < 2000; i++) {
            stillness.onGyro(t, 0.001f, 0f, 0f);
            stillness.onAccel(t, ax, 0f, az);
            t += 2_500_000L;
        }
        is("a slow slide is caught by the gravity direction", true, stillness.disturbed());
        near("and its size is recorded", 3.0, stillness.maxTiltDeg(), 0.3);
    }

    /**
     * The number the manifest publishes as {@code fixed_camera.rest_residual_deg}.
     *
     * <p>It has to see rotation the tilt check cannot: a phone yawing about the gravity axis keeps
     * a perfect gravity reading and has still moved the camera. It also has to not accumulate the
     * gyro's bias into a degree of rotation that never happened, which is why it integrates the
     * bias-corrected magnitude and why the reference is marked only after the bias has settled.
     */
    private static void stillnessIntegratesTheHoldResidual() {
        Stillness stillness = new Stillness();
        long t = feedQuiet(stillness, 0, 4000);            // ten seconds for the bias to settle
        stillness.markReference();
        near("a phone that has not moved has no residual", 0.0, stillness.residualDeg(), 0.05);

        t = feedQuiet(stillness, t, 1200);                 // three seconds of holding still
        is("noise alone stays well inside a tenth of a degree", true, stillness.residualDeg() < 0.1);

        // A yaw about gravity: invisible to the tilt check, which is the whole point.
        double before = stillness.maxTiltDeg();
        for (int i = 0; i < 400; i++) {                    // 0.02 rad/s for one second
            stillness.onGyro(t, 0f, 0f, 0.02f);
            stillness.onAccel(t, 0.01f, -0.01f, 9.81f);
            t += 2_500_000L;
        }
        near("the tilt check cannot see a yaw", before, stillness.maxTiltDeg(), 0.01);
        // 0.02 rad/s is below the rest threshold, so the detector still calls this at rest. That
        // is the residual's whole purpose: it is the only number that sees the drift a threshold
        // was set wide enough to pass.
        is("and the phone still reads as at rest", true, stillness.atRest());
        // Short of the full 1.15 degrees by exactly the frozen z bias (0.0015 rad/s, learned from
        // the quiet samples above), which is the correct answer rather than a tolerance being
        // stretched: the residual reports rotation net of the bias, not raw gyro output.
        near("the integral can", Math.toDegrees(0.02 - 0.0015), stillness.residualDeg(), 0.05);
    }

    // --------------------------------------------------------------------- arc

    /** Identity orientation: the phone flat, device z up, so gravity is along −z. */
    private static final float[] FLAT = {0f, 0f, 0f, 1f};

    private static long feedSweep(Arc arc, long fromNs, double rateRadS, double seconds,
                                  float sidewaysAccel) {
        long t = fromNs;
        int samples = (int) Math.round(seconds * 400);
        for (int i = 0; i < samples; i++) {
            arc.onRotation(FLAT);
            arc.onAccel(t, sidewaysAccel, 0f, 9.81f);
            arc.onGyro(t, 0f, 0f, (float) rateRadS);
            t += 2_500_000L;
        }
        return t;
    }

    private static void arcMeasuresSweep() {
        Arc arc = new Arc();
        arc.begin();
        feedSweep(arc, 0, -0.5, 2.094, 0.4f);        // 0.5 rad/s about gravity for 2.094 s
        near("sixty degrees measured as sixty", 60.0, arc.sweptDeg(), 1.0);
        is("the sweep is one-sided here", true, arc.suggestedDirection(60) != 0);
    }

    private static void arcRangeNotNetRotation() {
        Arc arc = new Arc();
        arc.begin();
        long t = feedSweep(arc, 0, -0.5, 1.047, 0.4f);      // 30 degrees out
        feedSweep(arc, t, 0.5, 1.047, 0.4f);                // and back
        near("out and back is thirty of range, not zero", 30.0, arc.sweptDeg(), 1.5);
        near("and the return error is small", 0.0, arc.returnErrorDeg(), 1.5);
    }

    private static void arcSeparatesOrbitFromPivot() {
        Arc orbit = new Arc();
        orbit.begin();
        feedSweep(orbit, 0, -0.5, 2.0, 0.4f);               // walking a curve
        is("a curved path reads as an orbit", true, orbit.orbitLikely());
        is("and the answer is a measurement", true, orbit.orbitMeasured());

        Arc pivot = new Arc();
        pivot.begin();
        feedSweep(pivot, 0, -0.5, 2.0, 0.0f);               // turning on the spot
        is("turning on the spot does not", false, pivot.orbitLikely());

        Arc blind = new Arc();
        blind.begin();
        long t = 0;
        for (int i = 0; i < 800; i++) {                     // no rotation vector on this device
            blind.onAccel(t, 0f, 0f, 9.81f);
            blind.onGyro(t, 0f, 0f, -0.5f);
            t += 2_500_000L;
        }
        is("a device that cannot judge it still measures the sweep", true, blind.sweptDeg() > 50f);
        is("and does not accuse the operator", true, blind.orbitLikely());
        is("but says the question went unanswered", false, blind.orbitMeasured());
    }

    private static void arcSuggestsTheEmptierSide() {
        Arc arc = new Arc();
        arc.begin();
        long t = feedSweep(arc, 0, -0.5, 1.1, 0.4f);        // 31 degrees, all to one side
        int first = arc.suggestedDirection(60);
        is("a one-sided sweep is sent the other way", true, first != 0);
        feedSweep(arc, t, 0.5, 2.2, 0.4f);                  // 63 back the other way
        is("once both sides have their share, nothing is suggested", 0, arc.suggestedDirection(60));
    }

    /**
     * The two numbers the manifest publishes about the sweep, beyond its extent.
     *
     * <p>The median rate is binned over sweeping samples only. Including the pauses would report a
     * rate the operator never moved at, and the rate is what the blur budget is judged against.
     * The implied cone is the narrower half discounted, and it must be narrower than the half —
     * this is the one number in the pack that could overstate what the recording supports, so its
     * failure direction is pinned here rather than left to a code reading.
     */
    private static void arcReportsRateAndCone() {
        Arc arc = new Arc();
        arc.begin();
        long t = feedSweep(arc, 0, -0.131, 4.0, 0.4f);      // 7.5 deg/s for four seconds
        near("the median rate is the rate that was swept", 7.5, arc.medianRateDegPerSec(), 0.6);
        near("and the duration is the time it took", 4.0, arc.durationSec(), 0.05);

        // Two seconds of holding the phone still in the middle of the sweep.
        for (int i = 0; i < 800; i++) {
            arc.onRotation(FLAT);
            arc.onAccel(t, 0.4f, 0f, 9.81f);
            arc.onGyro(t, 0f, 0f, 0f);
            t += 2_500_000L;
        }
        near("a pause does not drag the median towards zero", 7.5, arc.medianRateDegPerSec(), 0.6);

        // 30 degrees one way, 50 the other: the cone can only be as wide as the narrower side.
        Arc lopsided = new Arc();
        lopsided.begin();
        long u = feedSweep(lopsided, 0, -0.5, 1.047, 0.4f);        // 30 to one side
        feedSweep(lopsided, u, 0.5, 2.79, 0.4f);                   // back through zero, 50 the other
        float narrower = Math.min(lopsided.leftDeg(), lopsided.rightDeg());
        near("the narrower side is the thirty", 30.0, narrower, 2.0);
        is("the published cone never exceeds it", true, lopsided.impliedConeDeg() < narrower);
        is("and it is not zero either", true, lopsided.impliedConeDeg() > narrower * 0.5f);
    }

    // ------------------------------------------------------------------ checks

    private static void is(String what, Object expected, Object actual) {
        checks++;
        if (expected == null ? actual != null : !expected.equals(actual)) {
            failures++;
            System.out.println("FAIL  " + what + ": expected " + expected + ", got " + actual);
        }
    }

    private static void near(String what, double expected, double actual, double tolerance) {
        checks++;
        if (Math.abs(expected - actual) > tolerance) {
            failures++;
            System.out.println("FAIL  " + what + ": expected " + expected + " ± " + tolerance
                    + ", got " + actual);
        }
    }
}
