"""Where the phases of a scripted capture begin and end.

Preset F records four phases into a single video -- ``perch``, ``arc``,
``reseat``, ``hold`` -- because stopping and restarting the recorder between
them would move the phone, and the entire premise of the protocol is that the
phone does *not* move between the reseat and the hold. The phases are therefore
boundaries inside one file, and this module is where the rest of the pipeline
learns where they are.

There are two sources, in order of preference:

* **Declared.** A capturepack 0.3 manifest carries ``protocol.phases``: the
  capture app advanced its own script and knows the frame numbers exactly.
* **Inferred.** Footage shot with any other camera app carries no markers. The
  fallback measures, per frame, the *fraction of the frame that changed* since
  the previous frame. That one number separates the cases we care about with no
  learned model at all, because of the geometry of the protocol rather than any
  property of the subject:

  - camera moving (the arc): almost every pixel changes;
  - camera at rest, subject moving (the hold): a minority of pixels change;
  - camera at rest, nothing moving (the perch, and the tail of the reseat):
    only sensor noise changes.

  Three regimes, two thresholds, and the second threshold is measured from the
  footage's own noise floor rather than assumed.

The one thing inference genuinely cannot recover is the arc/reseat boundary:
both are "camera moving", and nothing in the pixels says which hand movement was
a deliberate sweep and which was putting the phone down. An inferred split says
so in :attr:`PhaseSplit.warnings` and folds the reseat into the arc, rather than
inventing a boundary that would be wrong by a variable amount.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from gausscapture.errors import CaptureFormatError
from gausscapture.pack.manifest import find_main_video, read_manifest

#: The scripted phases, in the order preset F records them.
PHASE_ORDER = ("perch", "arc", "reseat", "hold")

#: ``capture_type`` that promises a stationary camera over a moving subject.
FIXED_CAMERA_CAPTURE_TYPE = "fixed_camera_4d"

#: Long side the change profile is measured at. Motion detection does not need
#: resolution -- it needs to not spend a minute decoding 1080p to answer a
#: question about whole-frame statistics.
ANALYSIS_LONG_SIDE = 192

#: A pixel counts as changed when it moves by more than this many grey levels.
#: Below roughly six levels a JPEG/H.264 frame differs from itself for reasons
#: that have nothing to do with the scene.
CHANGE_LEVELS = 8.0

#: Fraction of the frame that must change before we call the *camera* moving.
#: A head and shoulders at conversational distance occupies well under a third
#: of the frame, so subject motion cannot reach this; camera motion sails past
#: it because parallax moves the background too.
MOVING_FRACTION = 0.30

#: Floor under the measured subject-motion threshold, as a fraction of the
#: frame. Without it, a perfectly clean sensor would make every stray compressed
#: macroblock the start of the hold.
MIN_SUBJECT_FRACTION = 0.004

MIN_ARC_FRAMES = 8
MIN_HOLD_FRAMES = 4
MIN_REST_FRAMES = 3

#: Longest rest window we will derive, in seconds. The rest window exists to
#: measure a noise floor; more than half a second of it buys nothing and starts
#: to risk catching the subject's first move.
MAX_REST_SECONDS = 0.5


@dataclass
class Phase:
    """One contiguous span of frames, both bounds inclusive."""

    name: str
    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if self.name not in PHASE_ORDER:
            raise CaptureFormatError(
                f"Unknown capture phase {self.name!r}. Known phases are "
                f"{', '.join(PHASE_ORDER)}."
            )
        if self.end_frame < self.start_frame:
            raise CaptureFormatError(
                f"Phase {self.name!r} ends at frame {self.end_frame} but starts at "
                f"{self.start_frame}. Phase bounds are inclusive and must not run backwards."
            )
        if self.start_frame < 0:
            raise CaptureFormatError(
                f"Phase {self.name!r} starts at frame {self.start_frame}; frame numbers "
                "are zero-based and never negative."
            )

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1

    def contains(self, frame_no: int) -> bool:
        return self.start_frame <= frame_no <= self.end_frame

    def frame_numbers(self) -> list[int]:
        return list(range(self.start_frame, self.end_frame + 1))

    def sample(self, source_fps: float, target_fps: float) -> list[int]:
        """Frame numbers thinned to roughly ``target_fps``.

        Always keeps the first and last frame of the phase: the ends of the arc
        are the widest baselines in the capture and are exactly what the cone
        is measured from, so a naive stride that drops them narrows the
        published cone for no reason.
        """
        if source_fps <= 0 or target_fps <= 0 or target_fps >= source_fps:
            return self.frame_numbers()
        stride = max(1, int(round(source_fps / target_fps)))
        chosen = list(range(self.start_frame, self.end_frame + 1, stride))
        if chosen[-1] != self.end_frame:
            chosen.append(self.end_frame)
        return chosen

    def evenly_spaced(self, count: int) -> list[int]:
        """``count`` frame numbers spread across the phase, ends included."""
        count = max(1, min(int(count), self.frame_count))
        if count == 1:
            return [self.start_frame]
        positions = np.linspace(self.start_frame, self.end_frame, count)
        return sorted({int(round(p)) for p in positions})

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "frame_count": self.frame_count,
        }


@dataclass
class PhaseSplit:
    """Which frames belong to which phase, and where they came from."""

    phases: dict[str, Phase]
    #: Frames with the camera at rest *and* nothing in the scene moving. This is
    #: the only span in the whole recording that measures the sensor's noise
    #: floor rather than the scene, which is what makes the dynamic-mask
    #: threshold a measurement instead of a guess.
    rest_window: tuple[int, int] | None = None
    source: str = "declared"
    rest_window_source: str = "declared"
    fps: float = 0.0
    total_frames: int = 0
    warnings: list[str] = field(default_factory=list)

    def require(self, name: str) -> Phase:
        phase = self.phases.get(name)
        if phase is None:
            raise CaptureFormatError(
                f"This capture has no {name!r} phase, and the step you asked for cannot "
                f"run without one. Phases present: {', '.join(self.phases) or 'none'}."
            )
        return phase

    def phase_of(self, frame_no: int) -> str | None:
        for name in PHASE_ORDER:
            phase = self.phases.get(name)
            if phase is not None and phase.contains(frame_no):
                return name
        return None

    def rest_frames(self) -> list[int]:
        if self.rest_window is None:
            return []
        return list(range(self.rest_window[0], self.rest_window[1] + 1))

    def duration_of(self, name: str) -> float:
        """Phase length in seconds, or 0.0 when the frame rate is unknown."""
        if self.fps <= 0:
            return 0.0
        return self.require(name).frame_count / self.fps

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "phases": [self.phases[n].to_dict() for n in PHASE_ORDER if n in self.phases],
            "rest_window": list(self.rest_window) if self.rest_window else None,
            "rest_window_source": self.rest_window_source,
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------
# measuring the video
# --------------------------------------------------------------------------


@dataclass
class ChangeProfile:
    """Per-frame whole-frame change, the signal every inference here reads.

    ``changed_fraction[i]`` is the fraction of pixels that differ from frame
    ``i - 1`` by more than :data:`CHANGE_LEVELS`; entry 0 is zero by definition.
    """

    fps: float
    changed_fraction: np.ndarray
    mean_abs_diff: np.ndarray

    @property
    def count(self) -> int:
        return int(self.changed_fraction.size)


def measure_change_profile(
    video_path: Path,
    long_side: int = ANALYSIS_LONG_SIDE,
    level_threshold: float = CHANGE_LEVELS,
) -> ChangeProfile:
    """Decode a video once and record how much each frame differs from the last."""
    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise CaptureFormatError(
            f"Video could not be opened while looking for capture phases: {video_path}"
        )
    changed: list[float] = []
    mean_diff: list[float] = []
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        previous: np.ndarray | None = None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            small = _downscale_gray(frame, long_side)
            if previous is None:
                changed.append(0.0)
                mean_diff.append(0.0)
            else:
                diff = cv2.absdiff(small, previous)
                changed.append(float(np.count_nonzero(diff > level_threshold)) / diff.size)
                mean_diff.append(float(diff.mean()))
            previous = small
    finally:
        capture.release()

    if not changed:
        raise CaptureFormatError(
            f"Video decoded to zero frames while looking for capture phases: {video_path}"
        )
    return ChangeProfile(
        fps=fps,
        changed_fraction=np.asarray(changed, dtype=np.float64),
        mean_abs_diff=np.asarray(mean_diff, dtype=np.float64),
    )


def iter_video_frames(
    video_path: Path, frame_numbers: Sequence[int]
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_number, bgr)`` for the requested frames, in order.

    Decoding is sequential rather than seeking per frame: phone video is
    long-GOP, so ``CAP_PROP_POS_FRAMES`` costs a keyframe-to-target decode every
    time and, on several containers, lands on the wrong frame. One linear pass
    is both faster and correct.
    """
    wanted = sorted({int(n) for n in frame_numbers if n >= 0})
    if not wanted:
        return
    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise CaptureFormatError(f"Video could not be opened for frame extraction: {video_path}")
    try:
        pending = iter(wanted)
        target = next(pending)
        frame_no = -1
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_no += 1
            while target is not None and frame_no == target:
                yield frame_no, frame
                target = next(pending, None)
            if target is None:
                break
    finally:
        capture.release()


def _downscale_gray(frame: np.ndarray, long_side: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    height, width = gray.shape[:2]
    longest = max(height, width)
    if longest > long_side:
        scale = long_side / longest
        gray = cv2.resize(
            gray,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    # A light blur before differencing: without it, sensor noise on a static
    # frame trips the per-pixel threshold often enough to look like motion.
    return cv2.GaussianBlur(gray, (3, 3), 0).astype(np.uint8)


# --------------------------------------------------------------------------
# declared phases
# --------------------------------------------------------------------------


def read_declared_split(manifest: dict[str, Any], total_frames: int = 0) -> PhaseSplit:
    """Build a split from a capturepack ``protocol`` block.

    Raises :class:`CaptureFormatError` naming the field at fault, because the
    only person who can fix a malformed protocol block is whoever wrote it and
    "invalid manifest" tells them nothing.
    """
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise CaptureFormatError(
            "This CapturePack declares no `protocol` block, so its phase boundaries are "
            "unknown. Record with preset F on a capturepack 0.3 capture app, or run the "
            "4D preparation with phase inference enabled."
        )
    entries = protocol.get("phases")
    if not isinstance(entries, list) or not entries:
        raise CaptureFormatError(
            "`protocol.phases` is missing or empty. It must be a list of "
            "{name, start_frame, end_frame} objects, one per recorded phase."
        )

    phases: dict[str, Phase] = {}
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise CaptureFormatError(
                f"`protocol.phases[{position}]` is not an object; expected "
                "{name, start_frame, end_frame}."
            )
        name = entry.get("name")
        if not isinstance(name, str):
            raise CaptureFormatError(f"`protocol.phases[{position}]` has no string `name`.")
        try:
            start = int(entry["start_frame"])
            end = int(entry["end_frame"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CaptureFormatError(
                f"`protocol.phases[{position}]` ({name}) needs integer `start_frame` and "
                f"`end_frame` fields: {exc}"
            ) from exc
        if name in phases:
            raise CaptureFormatError(
                f"Phase {name!r} is declared twice. Each phase occupies one contiguous span."
            )
        phases[name] = Phase(name=name, start_frame=start, end_frame=end)

    ordered = [phases[n] for n in PHASE_ORDER if n in phases]
    for earlier, later in zip(ordered, ordered[1:], strict=False):
        if later.start_frame <= earlier.end_frame:
            raise CaptureFormatError(
                f"Phases {earlier.name!r} (ends at frame {earlier.end_frame}) and "
                f"{later.name!r} (starts at frame {later.start_frame}) overlap. The four "
                "phases are recorded back to back and cannot share a frame."
            )

    capture_type = manifest.get("capture_type")
    if capture_type == FIXED_CAMERA_CAPTURE_TYPE and "hold" not in phases:
        raise CaptureFormatError(
            f"This CapturePack declares capture_type {FIXED_CAMERA_CAPTURE_TYPE!r} but has no "
            "`hold` phase, so it contains no footage of a moving subject in front of a "
            "stationary camera -- which is the only thing that capture type means. Re-record "
            "with preset F and let the take auto-stop rather than stopping it by hand."
        )

    warnings: list[str] = []
    if capture_type != FIXED_CAMERA_CAPTURE_TYPE and "hold" in phases:
        warnings.append(
            f"A `hold` phase is declared but capture_type is {capture_type!r} rather than "
            f"{FIXED_CAMERA_CAPTURE_TYPE!r}; treating the capture as fixed-camera anyway."
        )
    for name in PHASE_ORDER:
        if name not in phases:
            warnings.append(f"No {name!r} phase was declared.")

    rest = _read_declared_rest_window(protocol, phases)
    return PhaseSplit(
        phases=phases,
        rest_window=rest,
        source="declared",
        rest_window_source="declared" if rest else "missing",
        fps=float(protocol.get("fps") or manifest.get("video", {}).get("fps") or 0.0),
        total_frames=int(total_frames or ordered[-1].end_frame + 1),
        warnings=warnings,
    )


def _read_declared_rest_window(
    protocol: dict[str, Any], phases: dict[str, Phase]
) -> tuple[int, int] | None:
    window = protocol.get("rest_window")
    if not isinstance(window, dict):
        return None
    try:
        start = int(window["start_frame"])
        end = int(window["end_frame"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CaptureFormatError(
            f"`protocol.rest_window` needs integer `start_frame` and `end_frame`: {exc}"
        ) from exc
    if end < start:
        raise CaptureFormatError(
            f"`protocol.rest_window` ends at frame {end} but starts at {start}."
        )
    covering = {name for name, phase in phases.items() if phase.contains(start)}
    if covering and not covering & {"reseat", "hold"}:
        raise CaptureFormatError(
            "`protocol.rest_window` starts inside the "
            f"{sorted(covering)[0]!r} phase. The rest window is the span after the phone is "
            "back in its support and before the subject moves, so it belongs to `reseat` or "
            "to the start of `hold`."
        )
    return (start, end)


# --------------------------------------------------------------------------
# inferred phases
# --------------------------------------------------------------------------


def infer_phase_split(
    profile: ChangeProfile,
    moving_fraction: float = MOVING_FRACTION,
    min_arc_frames: int = MIN_ARC_FRAMES,
    min_hold_frames: int = MIN_HOLD_FRAMES,
) -> PhaseSplit:
    """Recover phase boundaries from a change profile alone."""
    count = profile.count
    if count < min_arc_frames + min_hold_frames:
        raise CaptureFormatError(
            f"This video is {count} frames long, which is too short to contain both a camera "
            "sweep and a fixed-camera hold. Record at least a few seconds of each."
        )

    moving = _smooth_flags(profile.changed_fraction > moving_fraction)
    runs = _runs(moving)
    sweeps = [(a, b) for value, a, b in runs if value and (b - a + 1) >= min_arc_frames]
    if not sweeps:
        raise CaptureFormatError(
            "No camera sweep was found in this recording: no run of at least "
            f"{min_arc_frames} frames changes more than {moving_fraction:.0%} of the image. "
            "A fixed-camera 4D capture needs a hand-held arc around the subject before the "
            "phone is set down -- without it there is no parallax anywhere in the clip and "
            "structure-from-motion has nothing to solve."
        )
    arc_start, arc_end = max(sweeps, key=lambda run: run[1] - run[0])

    if arc_end >= count - min_hold_frames:
        raise CaptureFormatError(
            "The recording ends while the camera is still moving. The hold -- the phone at "
            "rest while the subject moves -- is the footage the 4D scene is trained on, and "
            "there is none here. Let the take auto-stop instead of stopping it by hand."
        )

    warnings = [
        "Phases were inferred from image motion because the CapturePack declares no protocol "
        "block. The reseat cannot be separated from the arc without markers, so the frames "
        "where the phone was being replaced are counted as arc."
    ]

    phases: dict[str, Phase] = {}
    if arc_start > 0:
        phases["perch"] = Phase("perch", 0, arc_start - 1)
    else:
        warnings.append(
            "The recording starts with the camera already moving, so there is no perch phase "
            "and no frames shot from the pose the deliverable is rendered from."
        )
    phases["arc"] = Phase("arc", arc_start, arc_end)

    hold_start = _first_subject_motion(profile, arc_end + 1, count - 1)
    if hold_start is None or hold_start > count - min_hold_frames:
        # Nothing in the tail rises above the noise floor, or it rises too late
        # to leave a usable hold. Everything after the sweep is the hold; a
        # subject who never moved is a training problem, not a parsing one.
        hold_start = arc_end + 1
        warnings.append(
            "No subject motion was detected after the camera came to rest. The whole tail of "
            "the recording is treated as the hold, but a 4D scene trained on a subject who "
            "did not move is a static scene."
        )
    elif hold_start > arc_end + 1:
        phases["reseat"] = Phase("reseat", arc_end + 1, hold_start - 1)
    phases["hold"] = Phase("hold", hold_start, count - 1)

    split = PhaseSplit(
        phases=phases,
        source="inferred",
        fps=profile.fps,
        total_frames=count,
        warnings=warnings,
    )
    split.rest_window, split.rest_window_source = derive_rest_window(profile, split)
    return split


def derive_rest_window(profile: ChangeProfile, split: PhaseSplit) -> tuple[tuple[int, int] | None, str]:
    """Find the frames with the camera at rest and nothing moving.

    Preferred source is the tail of the reseat: the phone is in its support,
    hands are off, and the subject has not been cued yet. When there is no
    reseat -- an inferred split, or a capture that went straight from the sweep
    to the subject moving -- the head of the hold is used instead and labelled
    as such, because those frames are inside the dynamic phase and a caller that
    masks the hold has to know not to mask these.
    """
    max_frames = max(MIN_REST_FRAMES, int(round((profile.fps or 30.0) * MAX_REST_SECONDS)))
    quiet = profile.changed_fraction <= _subject_threshold(profile, split)

    reseat = split.phases.get("reseat")
    if reseat is not None:
        run = _trailing_run(quiet, reseat.start_frame, reseat.end_frame)
        if run is not None and run[1] - run[0] + 1 >= MIN_REST_FRAMES:
            start = max(run[0], run[1] - max_frames + 1)
            return (start, run[1]), "reseat_tail"

    hold = split.phases.get("hold")
    if hold is not None and hold.frame_count >= MIN_REST_FRAMES:
        end = min(hold.end_frame, hold.start_frame + max_frames - 1)
        return (hold.start_frame, end), "hold_head"
    return None, "missing"


def _subject_threshold(profile: ChangeProfile, split: PhaseSplit) -> float:
    """Change fraction above which something in a static frame really moved.

    Measured from the quiet part of the recording rather than assumed: take the
    frames after the sweep, and set the threshold four robust deviations above
    their median. A noisy sensor raises its own threshold, which is the whole
    point -- a constant would either miss the subject on a clean sensor or fire
    on grain on a dirty one.
    """
    arc = split.phases.get("arc")
    first = (arc.end_frame + 1) if arc is not None else 0
    tail = profile.changed_fraction[first:]
    if tail.size == 0:
        return MIN_SUBJECT_FRACTION
    median = float(np.median(tail))
    mad = float(np.median(np.abs(tail - median)))
    return max(median + 4.0 * 1.4826 * mad, MIN_SUBJECT_FRACTION)


def _first_subject_motion(profile: ChangeProfile, first: int, last: int) -> int | None:
    """First frame in ``[first, last]`` where motion is sustained for two frames."""
    if first > last:
        return None
    window = profile.changed_fraction[first : last + 1]
    median = float(np.median(window))
    mad = float(np.median(np.abs(window - median)))
    threshold = max(median + 4.0 * 1.4826 * mad, MIN_SUBJECT_FRACTION)
    active = window > threshold
    for i in range(active.size - 1):
        if active[i] and active[i + 1]:
            return first + i
    return None


def _smooth_flags(flags: np.ndarray, width: int = 5) -> np.ndarray:
    """Median-filter a boolean run so one dropped frame does not split a phase."""
    if flags.size < width:
        return flags
    half = width // 2
    padded = np.pad(flags.astype(np.uint8), half, mode="edge")
    stacked = np.stack([padded[i : i + flags.size] for i in range(width)], axis=0)
    return np.median(stacked, axis=0) > 0.5


def _runs(flags: np.ndarray) -> list[tuple[bool, int, int]]:
    runs: list[tuple[bool, int, int]] = []
    if flags.size == 0:
        return runs
    start = 0
    for i in range(1, flags.size):
        if flags[i] != flags[start]:
            runs.append((bool(flags[start]), start, i - 1))
            start = i
    runs.append((bool(flags[start]), start, flags.size - 1))
    return runs


def _trailing_run(flags: np.ndarray, first: int, last: int) -> tuple[int, int] | None:
    last = min(last, flags.size - 1)
    if first > last or not flags[last]:
        return None
    start = last
    while start > first and flags[start - 1]:
        start -= 1
    return start, last


# --------------------------------------------------------------------------
# the entry point the pipeline uses
# --------------------------------------------------------------------------


def phase_split_for_pack(
    pack_dir: Path,
    video_path: Path | None = None,
    allow_inference: bool = True,
) -> PhaseSplit:
    """Read or measure the phase split for a CapturePack.

    A declared split is trusted for its boundaries but still gets its rest
    window measured when the manifest omits one, because a manifest that knows
    the phases but not the noise floor is the common case for anything but the
    newest app build.
    """
    pack_dir = Path(pack_dir)
    manifest = read_manifest(pack_dir)
    video_path = Path(video_path) if video_path else find_main_video(pack_dir)

    try:
        split = read_declared_split(manifest)
    except CaptureFormatError:
        if not allow_inference:
            raise
        profile = measure_change_profile(video_path)
        return infer_phase_split(profile)

    if split.rest_window is None or split.fps <= 0 or split.total_frames <= 0:
        profile = measure_change_profile(video_path)
        if split.fps <= 0:
            split.fps = profile.fps
        split.total_frames = max(split.total_frames, profile.count)
        if split.rest_window is None:
            split.rest_window, split.rest_window_source = derive_rest_window(profile, split)
            if split.rest_window_source == "hold_head":
                split.warnings.append(
                    "The manifest declared no rest window, and there is no quiet tail to the "
                    "reseat, so the noise floor is measured from the first frames of the hold."
                )
    return split
