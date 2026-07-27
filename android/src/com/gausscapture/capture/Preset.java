package com.gausscapture.capture;

/**
 * One take in the capture protocol.
 *
 * <p>These are one tap each rather than a filename to type. Typing while holding a phone at arm's
 * length is exactly the friction that produces mislabelled data, and mislabelled data is worse
 * than none -- an unlabelled failure is a gap, a mislabelled one is a wrong answer.
 *
 * <p>Two of the five are expected to fail. That is deliberate: a predictor trained only on
 * successes has never seen the thing it is meant to predict, and "always succeeds" would score
 * perfectly on such a set. Zero parallax and a moving subject are the two failure modes worth
 * having, because they fail for different reasons.
 */
public final class Preset {

    public final String id;
    public final String title;
    public final String hint;
    /** Seconds of footage this take is aiming for. */
    public final int targetSeconds;
    /** Whether structure-from-motion is expected to fail on this take. */
    public final boolean expectedToFail;

    private Preset(String id, String title, String hint, int targetSeconds, boolean expectedToFail) {
        this.id = id;
        this.title = title;
        this.hint = hint;
        this.targetSeconds = targetSeconds;
        this.expectedToFail = expectedToFail;
    }

    public static final Preset[] ALL = {
            new Preset("A_good", "A · Good capture",
                    "Walk a slow full circle around the subject, keeping it centred.", 60, false),
            new Preset("B_normal", "B · Normal pace",
                    "One loop at ordinary walking speed.", 30, false),
            new Preset("C_fast", "C · Fast and shaky",
                    "Deliberately quick, with sharp turns.", 20, false),
            new Preset("D_rotation", "D · Rotation only",
                    "Stand on one spot and turn the phone. Expected to fail.", 20, true),
            new Preset("E_subject", "E · Moving subject",
                    "Hold the phone still while something moves. Expected to fail.", 20, true),
    };

    public static Preset byId(String id) {
        for (Preset preset : ALL) {
            if (preset.id.equals(id)) {
                return preset;
            }
        }
        return ALL[0];
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
