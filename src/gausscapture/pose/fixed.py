"""One pose for the whole dynamic phase, and how much to believe it.

During the hold the phone does not move, so there is exactly one camera pose for
every frame of it. Solving a pose per frame anyway -- which is what happens if
the hold frames are simply thrown at COLMAP -- does not measure that; it
measures the solver's noise on a degenerate problem. Handing those wobbles to a
deformation trainer is worse than useless: the trainer has no way to tell camera
jitter from subject motion, so it fits the jitter as motion the subject never
made, in the one part of the scene we have least evidence about.

So the pose is derived once, from the frames where the camera is provably at
rest, and stamped onto every hold frame. The derivation runs in three tiers, and
which tier produced the answer is published rather than hidden:

``inherited``
    The rotational chordal-L1 median and the translational median of the
    rest-window cameras. These registered unmasked against the full scene, so
    they are the best-conditioned views of the fixed pose in the capture.

``verified``
    As above, and the masked hold keyframes also registered, and reprojecting
    the sparse points through them lands within ``residual_tolerance_px`` of
    where the inherited pose puts them. The pose is confirmed by frames from
    inside the dynamic phase itself.

``asserted``
    The hold keyframes did not register at all -- a face against a blank wall
    leaves the masked frames with almost no background to match on. The
    inherited pose is used, and nothing downstream may present it as verified.

There is a fourth outcome the original three-tier sketch did not have a name
for, and it needs one: the hold keyframes register and *disagree*. That is not a
weak pose, it is evidence that the phone moved between the reseat and the hold,
and calling it ``asserted`` would file a contradiction under "unconfirmed". It
is reported as ``contradicted``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gausscapture.errors import CaptureFormatError
from gausscapture.pose.model import Camera, Image, SparseModel, quaternion_to_rotation

#: Reprojection agreement, in pixels, below which the hold keyframes confirm the
#: inherited pose. Three pixels at 1080p is about a tenth of a degree of yaw at
#: a typical phone focal length -- tight enough that a phone genuinely nudged in
#: its support fails it, loose enough that solver noise on a masked frame does
#: not.
RESIDUAL_TOLERANCE_PX = 3.0

#: Sparse points used for every pixel-space comparison. Reprojection statistics
#: converge long before a full model's worth of points, and the cap keeps the
#: check instant on a 500k-point reconstruction.
MAX_PROBE_POINTS = 2000


@dataclass
class FixedPoseReport:
    """The single camera pose of the dynamic phase, with its provenance."""

    #: COLMAP world-to-camera rotation quaternion, (w, x, y, z).
    qvec: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    #: COLMAP world-to-camera translation.
    tvec: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    #: Camera centre in world coordinates, which is what a viewer wants.
    centre: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    source: str = "asserted"
    derived_from: str = "rest_window"
    rest_cameras: int = 0
    hold_cameras: int = 0
    #: Widest angular departure of any contributing camera from the median.
    spread_deg: float = 0.0
    #: The same spread expressed where it matters, by reprojecting the sparse
    #: points. Degrees are not comparable across focal lengths; pixels are.
    spread_px: float | None = None
    #: Disagreement between the hold keyframes and the derived pose.
    residual_px: float | None = None
    residual_tolerance_px: float = RESIDUAL_TOLERANCE_PX
    warnings: list[str] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return self.source == "verified"

    def rotation(self) -> np.ndarray:
        return quaternion_to_rotation(np.asarray(self.qvec, dtype=float))

    def to_dict(self) -> dict[str, Any]:
        return {
            "qvec": list(self.qvec),
            "tvec": list(self.tvec),
            "centre": list(self.centre),
            "source": self.source,
            "verified": self.verified,
            "derived_from": self.derived_from,
            "rest_cameras": self.rest_cameras,
            "hold_cameras": self.hold_cameras,
            "spread_deg": self.spread_deg,
            "spread_px": self.spread_px,
            "residual_px": self.residual_px,
            "residual_tolerance_px": self.residual_tolerance_px,
            "warnings": list(self.warnings),
        }


def derive_fixed_pose(
    model: SparseModel,
    rest_names: Sequence[str],
    hold_names: Sequence[str] = (),
    points: np.ndarray | None = None,
    residual_tolerance_px: float = RESIDUAL_TOLERANCE_PX,
) -> FixedPoseReport:
    """Derive the one pose the dynamic phase was shot from.

    ``rest_names`` and ``hold_names`` are COLMAP image names; images that did
    not register are simply absent from the model and are counted as such rather
    than treated as an error, because failing to register is the measured
    outcome this pipeline is built to report.
    """
    by_name = {image.name: image for image in model.images.values()}
    rest = [by_name[name] for name in rest_names if name in by_name]
    hold = [by_name[name] for name in hold_names if name in by_name]

    if not rest and not hold:
        raise CaptureFormatError(
            "Neither the rest-window frames nor the hold keyframes registered in the sparse "
            f"model, so there is no evidence anywhere for the fixed camera pose. Of "
            f"{len(rest_names)} rest and {len(hold_names)} hold frames offered, none appear in "
            "the reconstruction. The usual cause is a background with no texture: reshoot with "
            "more of the room visible behind you."
        )

    warnings: list[str] = []
    if rest:
        contributors, derived_from = rest, "rest_window"
    else:
        contributors, derived_from = hold, "hold_keyframes"
        warnings.append(
            "No rest-window frame registered, so the fixed pose is taken from the masked hold "
            "keyframes themselves. Those frames see only the background, so the pose is "
            "solved from less evidence than the protocol intends."
        )

    qvec, centre = _median_pose(contributors)
    rotation = quaternion_to_rotation(qvec)
    tvec = -rotation @ centre

    camera = _camera_for(model, contributors[0])
    probe = _probe_points(points)

    spread_deg = _max_angle_deg(qvec, [image.qvec for image in contributors])
    spread_px = _max_reprojection_px(probe, camera, contributors, qvec, tvec)

    residual_px: float | None = None
    if rest and hold:
        residual_px = _max_reprojection_px(probe, camera, hold, qvec, tvec)
        if residual_px is None:
            source = "asserted"
            warnings.append(
                "The hold keyframes registered but the model has no sparse points to compare "
                "them against, so the pose could not be verified in pixel space."
            )
        elif residual_px <= residual_tolerance_px:
            source = "verified"
        else:
            source = "contradicted"
            warnings.append(
                f"The hold keyframes disagree with the rest-window pose by {residual_px:.1f} px, "
                f"past the {residual_tolerance_px:.1f} px tolerance. The phone moved between "
                "being set down and the subject moving. Every hold frame is still given the "
                "rest-window pose, because a per-frame pose on a stationary camera is noise -- "
                "but this take should be reshot rather than trained."
            )
    elif hold and not rest:
        source = "verified"
    else:
        source = "asserted"
        warnings.append(
            "The masked hold keyframes did not register, so the fixed pose is inherited from "
            "the rest window and never confirmed from inside the dynamic phase. It must not be "
            "presented as verified anywhere downstream."
        )

    return FixedPoseReport(
        qvec=[float(v) for v in qvec],
        tvec=[float(v) for v in tvec],
        centre=[float(v) for v in centre],
        source=source,
        derived_from=derived_from,
        rest_cameras=len(rest),
        hold_cameras=len(hold),
        spread_deg=float(spread_deg),
        spread_px=None if spread_px is None else float(spread_px),
        residual_px=None if residual_px is None else float(residual_px),
        residual_tolerance_px=float(residual_tolerance_px),
        warnings=warnings,
    )


def chordal_l1_median_quaternion(quats: Sequence[np.ndarray], iterations: int = 64) -> np.ndarray:
    """Geometric median of unit quaternions under the chordal metric.

    The median rather than the mean because one badly-solved rest frame should
    not drag the answer: an L2 mean moves proportionally to the outlier, an L1
    median barely moves at all. Weiszfeld's iteration converges in a handful of
    steps for the sub-degree spread a stationary camera actually produces.
    """
    q = np.asarray(quats, dtype=float).reshape(-1, 4)
    if q.shape[0] == 0:
        raise CaptureFormatError("Cannot take the median of zero rotations")
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise CaptureFormatError("Encountered a zero-length rotation quaternion")
    q = q / norms

    # q and -q are the same rotation, so the arithmetic has to be done on one
    # hemisphere or opposite-signed copies of the same pose cancel out.
    reference = q[0]
    signs = np.where(q @ reference < 0, -1.0, 1.0)
    q = q * signs[:, None]

    median = q.mean(axis=0)
    median /= np.linalg.norm(median) or 1.0
    for _ in range(iterations):
        distances = np.linalg.norm(q - median, axis=1)
        weights = 1.0 / np.maximum(distances, 1e-9)
        candidate = (weights[:, None] * q).sum(axis=0)
        norm = np.linalg.norm(candidate)
        if norm == 0:
            break
        candidate /= norm
        if np.linalg.norm(candidate - median) < 1e-12:
            median = candidate
            break
        median = candidate
    return median


def _median_pose(images: Sequence[Image]) -> tuple[np.ndarray, np.ndarray]:
    """Median rotation, and the component-wise median of the camera centres.

    Centres rather than translations: COLMAP's ``t`` is ``-R c``, so a median
    taken on ``t`` mixes the rotation into the position and lands somewhere that
    is not the median of anything physical.
    """
    qvec = chordal_l1_median_quaternion([image.qvec for image in images])
    centres = np.stack(
        [-quaternion_to_rotation(image.qvec).T @ image.tvec for image in images], axis=0
    )
    return qvec, np.median(centres, axis=0)


def _camera_for(model: SparseModel, image: Image) -> Camera | None:
    camera = model.cameras.get(image.camera_id)
    if camera is None and model.cameras:
        camera = next(iter(model.cameras.values()))
    return camera


def _probe_points(points: np.ndarray | None) -> np.ndarray | None:
    if points is None:
        return None
    array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if array.shape[0] == 0:
        return None
    if array.shape[0] <= MAX_PROBE_POINTS:
        return array
    stride = np.linspace(0, array.shape[0] - 1, MAX_PROBE_POINTS).round().astype(int)
    return array[np.unique(stride)]


def _max_angle_deg(reference: np.ndarray, quats: Sequence[np.ndarray]) -> float:
    worst = 0.0
    for raw in quats:
        q = np.asarray(raw, dtype=float)
        q = q / (np.linalg.norm(q) or 1.0)
        dot = float(np.clip(abs(np.dot(q, reference)), -1.0, 1.0))
        worst = max(worst, float(np.degrees(2.0 * np.arccos(dot))))
    return worst


def _max_reprojection_px(
    points: np.ndarray | None,
    camera: Camera | None,
    images: Sequence[Image],
    qvec: np.ndarray,
    tvec: np.ndarray,
) -> float | None:
    """Worst RMS pixel displacement between each image's pose and the reference.

    Distortion is deliberately left out: this is a *difference* of two
    projections of the same point through the same lens, and the distortion
    terms cancel to first order while adding a Newton solve per point.
    """
    if points is None or camera is None or not images:
        return None
    reference = _project(points, quaternion_to_rotation(qvec), np.asarray(tvec, dtype=float), camera)
    worst = 0.0
    for image in images:
        projected = _project(
            points, quaternion_to_rotation(image.qvec), np.asarray(image.tvec, dtype=float), camera
        )
        valid = np.isfinite(reference).all(axis=1) & np.isfinite(projected).all(axis=1)
        if not np.any(valid):
            continue
        delta = np.linalg.norm(projected[valid] - reference[valid], axis=1)
        worst = max(worst, float(np.sqrt(np.mean(delta**2))))
    return worst


def _project(points: np.ndarray, rotation: np.ndarray, tvec: np.ndarray, camera: Camera) -> np.ndarray:
    camera_space = points @ rotation.T + tvec
    depth = camera_space[:, 2]
    # Points behind the camera have no image position; NaN keeps them out of the
    # statistics instead of folding a mirrored projection into them.
    depth = np.where(depth > 1e-6, depth, np.nan)
    return np.stack(
        [
            camera.fx * camera_space[:, 0] / depth + camera.cx,
            camera.fy * camera_space[:, 1] / depth + camera.cy,
        ],
        axis=1,
    )
