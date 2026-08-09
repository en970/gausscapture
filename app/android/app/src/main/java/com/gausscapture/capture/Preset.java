package com.gausscapture.capture;

/**
 * One take in the capture protocol.
 *
 * <p>These are one tap each rather than a filename to type. Typing while holding a phone at arm's
 * length is exactly the friction that produces mislabelled data, and mislabelled data is worse
 * than none -- an unlabelled failure is a gap, a mislabelled one is a wrong answer.
 *
 * <p>Two of the first five are expected to fail. That is deliberate: a predictor trained only on
 * successes has never seen the thing it is meant to predict, and "always succeeds" would score
 * perfectly on such a set. Zero parallax and a moving subject are the two failure modes worth
 * having, because they fail for different reasons.
 *
 * <p><b>F is a different kind of preset.</b> A–E are a single stretch of recording with a
 * duration; F is a <em>script</em> — several phases inside one file, each with its own instruction
 * and its own idea of whether the phone should be moving. It exists because a fixed camera pointed
 * at a moving face produces zero parallax, and zero parallax is undefined for structure from
 * motion: the geometry simply is not in the recording. So the take first sweeps the phone around
 * the subject while the subject holds still — which makes the head part of a rigid scene and gives
 * ordinary triangulation a baseline — and only then returns the phone to its support and lets the
 * subject move. One file, because the two halves must share an exposure, a white balance and a
 * clock; splitting them into two recordings would reintroduce every difference the protocol exists
 * to remove.
 */
public final class Preset {

    /** How the phone is expected to behave during a phase. */
    public static final String PHONE_AT_REST = "at_rest";
    public static final String PHONE_IN_HAND = "in_hand";
    /** The phone should be coming back to rest; the phase ends when it does, not on a clock. */
    public static final String PHONE_SETTLING = "settling";

    /**
     * One scripted stretch of a take.
     *
     * <p>The phases are what the desktop side splits a recording on, so their identifiers are part
     * of the data format rather than of the interface: {@code perch}, {@code arc}, {@code reseat},
     * {@code hold} appear in {@code frames.jsonl} and in the manifest, and
     * {@code gausscapture prep4d} partitions frames by them.
     */
    public static final class Phase {

        /** Stable identifier, written into every frame row. */
        public final String id;
        /** The one word shown while this phase runs. Neutral: an instruction, not a fault. */
        public final String cue;
        /** One supporting line, read once at the start of the phase. */
        public final String detail;
        /** Nominal seconds. What the progress ring is drawn against. */
        public final int seconds;
        /** Hard ceiling for a phase that ends on a condition rather than on the clock. */
        public final int maxSeconds;
        /** One of {@link #PHONE_AT_REST}, {@link #PHONE_IN_HAND}, {@link #PHONE_SETTLING}. */
        public final String phone;
        /** Whether the subject is asked to move. Exactly one phase of a 4D take may say yes. */
        public final boolean subjectMoves;

        Phase(String id, String cue, String detail, int seconds, int maxSeconds, String phone,
              boolean subjectMoves) {
            this.id = id;
            this.cue = cue;
            this.detail = detail;
            this.seconds = seconds;
            this.maxSeconds = maxSeconds;
            this.phone = phone;
            this.subjectMoves = subjectMoves;
        }

        public boolean phoneAtRest() {
            return PHONE_AT_REST.equals(phone);
        }

        public boolean phoneInHand() {
            return PHONE_IN_HAND.equals(phone);
        }

        public boolean endsOnRest() {
            return PHONE_SETTLING.equals(phone);
        }
    }

    public final String id;
    public final String title;
    public final String hint;
    /** Seconds of footage this take is aiming for. For a script, the sum of its phases. */
    public final int targetSeconds;
    /** Whether structure-from-motion is expected to fail on this take. */
    public final boolean expectedToFail;
    /** The phases, in order. A–E have exactly one; F has four. */
    public final Phase[] phases;
    /** {@code "front"} or {@code "back"}. The 4D protocol films the operator's own face. */
    public final String lensFacing;
    /** Total sweep the arc is aiming for, in degrees. Zero when the preset has no arc. */
    public final int arcTargetDeg;
    /** Below this the arc is too narrow to label a view cone with, and the pack says so. */
    public final int arcFloorDeg;
    /** What capture_type the manifest declares, which is what the desktop dispatches on. */
    public final String captureType;

    /**
     * Screen brightness held for the whole take, or a negative number to leave it alone.
     *
     * <p>Not a comfort setting. On the back camera the panel faces away from the scene and full
     * brightness is the only thing that makes the preview readable in sunlight. On the front
     * camera at arm's length the panel <em>is</em> a light source on the subject's face, roughly
     * 300 mm away, and during the arc it moves with the camera — so a bright panel writes a
     * moving specular highlight into every frame and breaks the brightness constancy the whole
     * capture depends on. Dim and constant beats bright and moving.
     */
    public final float screenBrightness;

    private Preset(String id, String title, String hint, int targetSeconds, boolean expectedToFail,
                   Phase[] phases, String lensFacing, int arcTargetDeg, int arcFloorDeg,
                   String captureType, float screenBrightness) {
        this.id = id;
        this.title = title;
        this.hint = hint;
        this.targetSeconds = targetSeconds;
        this.expectedToFail = expectedToFail;
        this.phases = phases;
        this.lensFacing = lensFacing;
        this.arcTargetDeg = arcTargetDeg;
        this.arcFloorDeg = arcFloorDeg;
        this.captureType = captureType;
        this.screenBrightness = screenBrightness;
    }

    /** A–E: one undivided stretch of recording, stopped by hand. */
    private static Preset simple(String id, String title, String hint, int seconds,
                                 boolean expectedToFail, String captureType) {
        Phase[] one = {new Phase("free", "", hint, seconds, seconds, PHONE_IN_HAND, false)};
        return new Preset(id, title, hint, seconds, expectedToFail, one, "back", 0, 0,
                captureType, 1.0f);
    }

    /**
     * The arc target, raised from the 8 seconds the geometry needs to 12.
     *
     * <p>The blur budget sets a ceiling on sweep rate, not a floor: at a 10 ms exposure and
     * 1150 px of focal length, 1.5 px of smear allows 7.5 deg/s. Sweeping the same 60 degrees at
     * 5 deg/s costs four extra seconds and buys 1.0 px of smear and about 120 sampled images
     * instead of 80. Nothing about the reconstruction prefers the faster sweep.
     */
    private static final int ARC_SECONDS = 12;

    public static final Preset[] ALL = {
            simple("A_good", "A · Good capture",
                    "Walk a slow full circle around the subject, keeping it centred.", 60, false,
                    "static_scene"),
            simple("B_normal", "B · Normal pace",
                    "One loop at ordinary walking speed.", 30, false, "static_scene"),
            simple("C_fast", "C · Fast and shaky",
                    "Deliberately quick, with sharp turns.", 20, false, "static_scene"),
            simple("D_rotation", "D · Rotation only",
                    "Stand on one spot and turn the phone. Expected to fail.", 20, true,
                    "static_scene"),
            simple("E_subject", "E · Moving subject",
                    "Hold the phone still while something moves. Expected to fail.", 20, true,
                    "dynamic_subject"),

            new Preset("F_bullet", "F · Bullet time",
                    "Sweep around your own face, put the phone back, then move.",
                    2 + ARC_SECONDS + 3 + 3, false,
                    new Phase[]{
                            // The pose the deliverable is shot from, held long enough for the 3A
                            // algorithms to converge on it before anything is locked. These frames
                            // are also static scene content from very near the final pose, which
                            // is what the desktop verifies the fixed pose against.
                            new Phase("perch", "Hold still",
                                    "Phone in its rest, your face framed. Hands off.",
                                    2, 8, PHONE_AT_REST, false),
                            // The only phase that contains geometry. The subject is part of the
                            // rigid scene here, so their head triangulates like the wall behind
                            // them; this is the parallax a fixed camera can never produce.
                            new Phase("arc", "Sweep around",
                                    "Lift the phone and sweep it around your face. Stay still.",
                                    ARC_SECONDS, ARC_SECONDS + 8, PHONE_IN_HAND, false),
                            // Ends when the phone is measurably at rest, never on a clock: the
                            // whole point is that the next phase begins from a camera that is not
                            // moving, and only the sensors know when that is true.
                            new Phase("reseat", "Phone back",
                                    "Put the phone back exactly where it was, and let go.",
                                    3, 12, PHONE_SETTLING, false),
                            // Auto-stopped. Reaching for the phone to press stop moves it, and the
                            // corrupted frames would be the ones closest to the motion we want.
                            new Phase("hold", "Now move",
                                    "Talk, smile, turn your head. The phone stays put.",
                                    3, 3, PHONE_AT_REST, true),
                    },
                    "front", 60, 40, "fixed_camera_4d", 0.35f),
    };

    public static Preset byId(String id) {
        for (Preset preset : ALL) {
            if (preset.id.equals(id)) {
                return preset;
            }
        }
        return ALL[0];
    }

    /** True when this take is a script rather than one stretch of recording. */
    public boolean isScripted() {
        return phases.length > 1;
    }

    /** The phase the subject is asked to move in, or null when there is none. */
    public Phase dynamicPhase() {
        for (Phase phase : phases) {
            if (phase.subjectMoves) {
                return phase;
            }
        }
        return null;
    }

    public boolean usesFrontCamera() {
        return "front".equals(lensFacing);
    }

    /** Short label for the bottom bar, for example "Shot: A". */
    public String shortLabel() {
        return "Shot: " + id.substring(0, 1);
    }

    public static String[] menuLabels() {
        String[] labels = new String[ALL.length];
        for (int i = 0; i < ALL.length; i++) {
            labels[i] = ALL[i].title + "\n" + ALL[i].targetSeconds + "s · " + ALL[i].hint;
        }
        return labels;
    }
}
