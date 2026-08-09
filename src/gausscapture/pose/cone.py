"""How wide the trained view actually is, measured rather than declared.

A fixed-camera 4D scene can only be honestly shown from angles the capture
observed. The viewer therefore clamps the camera to a cone, and the number that
cone is built from has to be a measurement of the recording, not a constant in a
config file and not the phone's own guess.

Once structure-from-motion has solved the arc, that measurement is available
directly: the registered arc cameras have world positions, the subject has a
centroid, and the angular extent of the former about the latter -- in azimuth
and in elevation separately, in the reconstruction's own frame -- is the extent
that was reconstructed. The IMU's swept angle is kept as a cross-check and may
only ever *narrow* the result, because the IMU measures how far the operator's
hand went, not how much of that COLMAP could use.

The cone is reported as a symmetric half-angle per axis, and a symmetric cone
has to fit inside the observed range on **both** sides: sweeping 40 degrees left
and 5 right gives a 5 degree half-angle, not 22. Publishing the average would
authorise the viewer to render 22 degrees to the right, where nothing was ever
seen.

Elevation is almost always the small one and that is geometry, not a bug: the
arc is horizontal, so it contributes essentially no vertical baseline, and a
subject's chin occludes everything below. Publishing a comfortable-looking
vertical number next to an honest horizontal one would defeat the purpose of
measuring either.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gausscapture.errors import CaptureFormatError
from gausscapture.pose.fixed import FixedPoseReport
from gausscapture.pose.model import Camera, SparseModel, quaternion_to_rotation

#: Half-angle below which the cone is too narrow to be worth labelling. The
#: protocol asks for a 40 degree total sweep for this reason.
MIN_USEFUL_HALF_DEG = 20.0

#: Registered arc cameras below which the extent is a guess dressed as a
#: measurement.
MIN_ARC_CAMERAS = 8

#: Percentile trimmed from each end before the extent is read. One camera solved
#: onto a wild pose would otherwise widen the published cone, and the cone is the
#: one number in this pipeline that must never be optimistic.
TRIM_PERCENTILE = 2.0


@dataclass
class ConeReport:
    """The angular extent the reconstruction actually covers."""

    azimuth_half_deg: float = 0.0
    elevation_half_deg: float = 0.0
    azimuth_min_deg: float = 0.0
    azimuth_max_deg: float = 0.0
    elevation_min_deg: float = 0.0
    elevation_max_deg: float = 0.0
    #: Median distance from the arc cameras to the subject centroid, in the
    #: reconstruction's arbitrary scale. The viewer's default orbit radius.
    radius: float = 0.0
    cameras: int = 0
    centroid: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    centroid_source: str = "sparse_points"
    source: str = "measured"
    imu_implied_half_deg: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.azimuth_half_deg >= MIN_USEFUL_HALF_DEG

    def label(self) -> str:
        """The string the viewer prints under the azimuth rail."""
        return (
            f"Trained view · ±{self.azimuth_half_deg:.0f}° × "
            f"±{self.elevation_half_deg:.0f}°"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "azimuth_half_deg": self.azimuth_half_deg,
            "elevation_half_deg": self.elevation_half_deg,
            "azimuth_min_deg": self.azimuth_min_deg,
            "azimuth_max_deg": self.azimuth_max_deg,
            "elevation_min_deg": self.elevation_min_deg,
            "elevation_max_deg": self.elevation_max_deg,
            "radius": self.radius,
            "cameras": self.cameras,
            "centroid": list(self.centroid),
            "centroid_source": self.centroid_source,
            "source": self.source,
            "imu_implied_half_deg": self.imu_implied_half_deg,
            "label": self.label(),
            "warnings": list(self.warnings),
        }


def subject_centroid(
    points: np.ndarray,
    fixed: FixedPoseReport,
    camera: Camera,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Where the subject is, in reconstruction coordinates.

    The cone is an angle *about the subject*, so the vertex matters: measured
    about the centroid of the whole point cloud, a capture with a deep room
    behind the sitter reports a cone several times too wide. When a dynamic mask
    is available the vertex is the median of the sparse points that reproject
    into it -- the points on the person -- and the fallback to the full cloud is
    labelled so nobody reads the two as equivalent.
    """
    array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if array.shape[0] == 0:
        raise CaptureFormatError(
            "The sparse model contains no 3D points, so the subject's position is unknown and "
            "the view cone cannot be measured. Structure-from-motion produced cameras but no "
            "structure, which usually means the scene had too little texture."
        )

    rotation = fixed.rotation()
    tvec = np.asarray(fixed.tvec, dtype=np.float64)
    camera_space = array @ rotation.T + tvec
    depth = camera_space[:, 2]
    in_front = depth > 1e-6

    if mask is not None and np.any(mask):
        u = camera.fx * camera_space[:, 0] / np.where(in_front, depth, 1.0) + camera.cx
        v = camera.fy * camera_space[:, 1] / np.where(in_front, depth, 1.0) + camera.cy
        height, width = np.asarray(mask).shape[:2]
        col = np.clip(np.round(u), 0, width - 1).astype(int)
        row = np.clip(np.round(v), 0, height - 1).astype(int)
        inside = in_front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        selected = inside & np.asarray(mask, dtype=bool)[row, col]
        if np.count_nonzero(selected) >= 8:
            return np.median(array[selected], axis=0), "dynamic_mask"

    if np.any(in_front):
        return np.median(array[in_front], axis=0), "sparse_points"
    return np.median(array, axis=0), "sparse_points_unfiltered"


def measure_cone(
    model: SparseModel,
    arc_names: Sequence[str],
    fixed: FixedPoseReport,
    centroid: np.ndarray,
    centroid_source: str = "sparse_points",
    imu_implied_half_deg: float | None = None,
    trim_percentile: float = TRIM_PERCENTILE,
) -> ConeReport:
    """Measure the azimuth and elevation extent of the solved arc cameras.

    Angles are signed offsets from the fixed camera's own viewing direction, so
    ``(0, 0)`` is the pose the deliverable was shot from and the viewer's rest
    position needs no separate calibration.
    """
    by_name = {image.name: image for image in model.images.values()}
    arc = [by_name[name] for name in arc_names if name in by_name]
    warnings: list[str] = []

    if not arc:
        raise CaptureFormatError(
            f"None of the {len(arc_names)} arc frames registered in the sparse model, so there "
            "is no parallax anywhere in this reconstruction and no view cone to measure. A "
            "fixed-camera clip on its own is undefined for structure-from-motion: the arc is "
            "the only part of the recording that carries baseline. Reshoot the sweep more "
            "slowly, with the subject holding still, in better light."
        )

    centroid = np.asarray(centroid, dtype=np.float64).reshape(3)
    origin = np.asarray(fixed.centre, dtype=np.float64)
    forward, right, up = _reference_frame(fixed, centroid)

    azimuths: list[float] = []
    elevations: list[float] = []
    radii: list[float] = []
    for image in arc:
        centre = -quaternion_to_rotation(image.qvec).T @ np.asarray(image.tvec, dtype=np.float64)
        offset = centroid - centre
        distance = float(np.linalg.norm(offset))
        if distance < 1e-9:
            continue
        view = offset / distance
        radii.append(distance)
        azimuths.append(float(np.degrees(np.arctan2(view @ right, view @ forward))))
        elevations.append(
            float(np.degrees(np.arctan2(view @ up, float(np.hypot(view @ forward, view @ right)))))
        )

    if not azimuths:
        raise CaptureFormatError(
            "Every registered arc camera sits on the subject centroid, which cannot happen in a "
            "real reconstruction. The sparse model is degenerate; rerun structure-from-motion."
        )

    az_min, az_max = _trimmed_extent(np.asarray(azimuths), trim_percentile)
    el_min, el_max = _trimmed_extent(np.asarray(elevations), trim_percentile)
    azimuth_half = max(0.0, min(-az_min, az_max))
    elevation_half = max(0.0, min(-el_min, el_max))

    source = "measured"
    if imu_implied_half_deg is not None and imu_implied_half_deg < azimuth_half:
        azimuth_half = float(imu_implied_half_deg)
        source = "measured_clamped_by_imu"

    if len(arc) < MIN_ARC_CAMERAS:
        warnings.append(
            f"Only {len(arc)} of {len(arc_names)} arc frames registered. The cone is measured "
            "from too few viewpoints to be worth much; treat it as a lower bound."
        )
    if azimuth_half < MIN_USEFUL_HALF_DEG:
        warnings.append(
            f"The reconstructed azimuth cone is only ±{azimuth_half:.1f}°, short of the "
            f"±{MIN_USEFUL_HALF_DEG:.0f}° the protocol asks for. Observed range is "
            f"{az_min:.1f}° to {az_max:.1f}°, so the sweep was either short or lopsided; "
            "a symmetric cone can only be as wide as its narrower side."
        )
    if centroid_source != "dynamic_mask":
        warnings.append(
            "The cone was measured about the centroid of the whole sparse cloud rather than "
            "about the subject, because too few points reprojected into the dynamic mask. A "
            "deep background biases this outward, so the published cone may be optimistic."
        )

    return ConeReport(
        azimuth_half_deg=float(azimuth_half),
        elevation_half_deg=float(elevation_half),
        azimuth_min_deg=float(az_min),
        azimuth_max_deg=float(az_max),
        elevation_min_deg=float(el_min),
        elevation_max_deg=float(el_max),
        radius=float(np.median(radii)) if radii else float(np.linalg.norm(centroid - origin)),
        cameras=len(arc),
        centroid=[float(v) for v in centroid],
        centroid_source=centroid_source,
        source=source,
        imu_implied_half_deg=(
            None if imu_implied_half_deg is None else float(imu_implied_half_deg)
        ),
        warnings=warnings,
    )


def _reference_frame(
    fixed: FixedPoseReport, centroid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-handed (forward, right, up) about the subject, from the fixed pose.

    Forward is the fixed camera's actual line of sight to the subject rather
    than its optical axis; the two differ by however far off centre the sitter
    is, and it is the line of sight that the viewer's rest position means.
    """
    rotation = fixed.rotation()
    origin = np.asarray(fixed.centre, dtype=np.float64)
    forward = centroid - origin
    norm = float(np.linalg.norm(forward))
    if norm < 1e-9:
        raise CaptureFormatError(
            "The fixed camera sits on the subject centroid, which cannot happen in a real "
            "reconstruction. The sparse model is degenerate; rerun structure-from-motion."
        )
    forward = forward / norm

    # COLMAP's camera y axis points down the image, so world up is its negation.
    up = -rotation[1, :].astype(np.float64)
    up = up - float(up @ forward) * forward
    up_norm = float(np.linalg.norm(up))
    if up_norm < 1e-6:
        # The camera is looking straight up or down at the subject; any vector
        # orthogonal to forward will do as an up reference.
        up = np.cross(forward, rotation[0, :].astype(np.float64))
        up_norm = float(np.linalg.norm(up)) or 1.0
    up = up / up_norm
    right = np.cross(up, forward)
    right = right / (float(np.linalg.norm(right)) or 1.0)
    return forward, right, up


def _trimmed_extent(values: np.ndarray, percentile: float) -> tuple[float, float]:
    if values.size < 5 or percentile <= 0:
        return float(values.min()), float(values.max())
    low = float(np.percentile(values, percentile))
    high = float(np.percentile(values, 100.0 - percentile))
    return low, high
