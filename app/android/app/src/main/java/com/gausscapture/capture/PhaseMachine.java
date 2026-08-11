package com.gausscapture.capture;

/**
 * The script that runs inside one scripted take: which phase is current, what to say, when to stop.
 *
 * <p>It holds no Android state at all — it is fed an elapsed time and a description of what the
 * sensors currently believe, and it answers with a phase, a cue, at most one warning, a countdown
 * and, eventually, a reason to stop. That is not tidiness for its own sake: this is the one piece
 * of the 4D protocol whose behaviour is a sequence of decisions rather than a call into the
 * platform, and it is therefore the one piece that can be tested exhaustively without a phone.
 *
 * <p>Three of its rules are load-bearing rather than cosmetic.
 *
 * <p><b>The transition ends on a measurement, never on a clock.</b> The phase after it gives every
 * one of its frames the same camera pose. If it began while the phone was still settling, that
 * assumption is false for the first half second and nothing downstream would ever say so.
 *
 * <p><b>The dynamic phase auto-stops.</b> Reaching for the phone to press stop moves it, and the
 * frames that would corrupt are the last ones — the ones closest to the motion the whole take
 * exists to record.
 *
 * <p><b>The dynamic phase raises no warnings.</b> The subject is being asked to move and to look
 * natural while doing it; a large amber word appearing in front of them is both useless (there is
 * nothing they can usefully do about the phone now) and destructive of the thing being recorded.
 * If something goes wrong there, the take ends and says why afterwards.
 */
final class PhaseMachine {

    /** Continuous rest required before the transition is allowed to complete. */
    static final long REST_CONFIRM_MS = 700;

    /**
     * Hands-off pause between rest being confirmed and the dynamic phase starting.
     *
     * <p>Long enough for the ripple of letting go to die out and for the subject to be ready, short
     * enough that nobody fills it by reaching back towards the phone.
     */
    static final long COUNTDOWN_MS = 1500;

    /** How long a hand-held phase runs before it is fair to complain about the sweep. */
    private static final long SWEEP_GRACE_MS = 3000;

    /** Below this, nothing has been swept at all and the operator has misread the instruction. */
    private static final float SWEEP_STARTED_DEG = 6f;

    /** What the sensors currently believe. Everything the machine is allowed to know. */
    static final class Motion {
        boolean atRest;
        long restForMillis;
        float sweptDeg;
        boolean orbitLikely;
    }

    /** Why a scripted take ended. */
    static final String REASON_COMPLETE = "complete";

    private final Preset preset;
    private final Preset.Phase[] phases;

    private int index;
    private long phaseStartMs;
    private long countdownStartMs = -1;
    private long lastElapsedMs;
    private int transitions;

    private String cue = "";
    private String warning;
    private long countdownMs = -1;
    private boolean restConfirmed;
    private long restConfirmedAtMs = -1;
    private String finishedReason;
    private boolean aborted;

    PhaseMachine(Preset preset) {
        this.preset = preset;
        this.phases = preset.phases;
        this.cue = phases[0].cue;
    }

    /** Advance the script to {@code elapsedMs} since the recording started. */
    void update(long elapsedMs, Motion motion) {
        lastElapsedMs = elapsedMs;
        if (finishedReason != null) {
            return;
        }
        Preset.Phase phase = phases[index];
        long inPhase = elapsedMs - phaseStartMs;
        long nominal = phase.seconds * 1000L;
        long ceiling = phase.maxSeconds * 1000L;

        cue = phase.cue;
        warning = null;
        countdownMs = -1;

        if (phase.subjectMoves) {
            updateDynamic(phase, inPhase, motion);
            return;
        }
        if (phase.endsOnRest()) {
            updateSettling(inPhase, ceiling, motion);
            return;
        }
        if (phase.phoneInHand()) {
            updateSweep(inPhase, nominal, ceiling, motion);
            return;
        }
        updateAtRest(inPhase, nominal, ceiling, motion);
    }

    /**
     * The opening phase: the phone is already where the take will be shot from.
     *
     * <p>It is not padding. These frames are static scene content from the pose the deliverable is
     * shot from, and they are what the desktop's fixed-pose check has to verify against. They are
     * also where the 3A algorithms converge before anything is locked, which is why the phase has
     * a floor as well as a target.
     */
    private void updateAtRest(long inPhase, long nominal, long ceiling, Motion motion) {
        if (!motion.atRest) {
            warning = "Hands off";
        }
        if (inPhase >= nominal && motion.restForMillis >= REST_CONFIRM_MS) {
            advance();
        } else if (inPhase >= ceiling) {
            abort("The phone never settled");
        }
    }

    private void updateSweep(long inPhase, long nominal, long ceiling, Motion motion) {
        if (inPhase >= SWEEP_GRACE_MS) {
            if (motion.sweptDeg < SWEEP_STARTED_DEG) {
                warning = "Move around";
            } else if (!motion.orbitLikely) {
                // Turning on the spot produces no parallax however far it goes. This is the one
                // failure that cannot be recovered afterwards by any amount of processing.
                warning = "Walk around it";
            }
        }
        if (inPhase >= nominal) {
            if (motion.sweptDeg >= preset.arcFloorDeg) {
                advance();
            } else if (inPhase >= ceiling) {
                // Ending the take rather than carrying on without the sweep. This used to
                // advance at the ceiling whatever the angle, and the result was a take that
                // looked complete and could not be reconstructed: the arc is where the
                // geometry comes from, so a capture without it is not a weaker capture, it
                // is an unusable one. A real take measured 0.34 degrees here, ran to the end
                // and was only found to be worthless on the desktop half an hour later.
                abort("The sweep never got wide enough — "
                        + Math.round(motion.sweptDeg) + "° of " + preset.arcFloorDeg + "°. "
                        + "Carry the phone around your face, keeping your head still.");
            } else {
                // Short of the floor: keep going rather than accept a cone too narrow to label.
                warning = "Sweep wider";
            }
        }
    }

    private void updateSettling(long inPhase, long ceiling, Motion motion) {
        if (motion.atRest && motion.restForMillis >= REST_CONFIRM_MS) {
            if (countdownStartMs < 0) {
                countdownStartMs = lastElapsedMs;
                restConfirmed = true;
                restConfirmedAtMs = lastElapsedMs;
            }
        } else {
            // Picking the phone back up, or a table still ringing, restarts the countdown. A
            // countdown that survives the phone being touched is a countdown that means nothing.
            countdownStartMs = -1;
            restConfirmed = false;
        }

        if (countdownStartMs >= 0) {
            countdownMs = COUNTDOWN_MS - (lastElapsedMs - countdownStartMs);
            if (countdownMs <= 0) {
                countdownMs = -1;
                advance();
            }
        } else if (inPhase >= ceiling) {
            abort("The phone never settled");
        }
    }

    private void updateDynamic(Preset.Phase phase, long inPhase, Motion motion) {
        if (!motion.atRest) {
            // Not a warning. There is nothing useful to do about it now, and the recording is of a
            // face that would react to being shouted at. The take ends and the manifest says why.
            abort("The phone moved");
            return;
        }
        if (inPhase >= phase.seconds * 1000L) {
            finishedReason = REASON_COMPLETE;
        }
    }

    private void advance() {
        index++;
        phaseStartMs = lastElapsedMs;
        countdownStartMs = -1;
        transitions++;
        if (index >= phases.length) {
            index = phases.length - 1;
            finishedReason = REASON_COMPLETE;
            return;
        }
        cue = phases[index].cue;
        warning = null;
    }

    private void abort(String reason) {
        finishedReason = reason;
        aborted = true;
        warning = reason;
    }

    // ------------------------------------------------------------------ state

    Preset.Phase phase() {
        return phases[index];
    }

    String phaseId() {
        return phases[index].id;
    }

    int phaseIndex() {
        return index;
    }

    int phaseCount() {
        return phases.length;
    }

    long phaseElapsedMs() {
        return lastElapsedMs - phaseStartMs;
    }

    /** Nominal length of the current phase, for a progress ring. */
    long phaseTargetMs() {
        return phases[index].seconds * 1000L;
    }

    /** The neutral instruction for the current phase. Never a fault. */
    String cue() {
        return cue;
    }

    /** The one thing that is wrong, or null. Never populated during the dynamic phase. */
    String warning() {
        return warning;
    }

    /** Milliseconds until the dynamic phase begins, or −1 when nothing is counting down. */
    long countdownMillis() {
        return countdownMs;
    }

    /** Number of phase changes so far, so the interface can buzz exactly once on each. */
    int transitions() {
        return transitions;
    }

    boolean finished() {
        return finishedReason != null;
    }

    /** {@link #REASON_COMPLETE}, or a sentence saying what went wrong. Null while running. */
    String finishedReason() {
        return finishedReason;
    }

    boolean aborted() {
        return aborted;
    }

    /** True once the phone was measured at rest for long enough to start the dynamic phase. */
    boolean restConfirmed() {
        return restConfirmed;
    }

    /** When that happened, in milliseconds since the recording started, or −1. */
    long restConfirmedAtMs() {
        return restConfirmedAtMs;
    }
}
