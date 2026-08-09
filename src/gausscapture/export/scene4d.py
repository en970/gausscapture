"""The ``.g4d`` container: a deformable gaussian scene a browser can stream.

A 4D scene is a canonical set of gaussians plus a description of how they move.
The obvious encoding -- bake every gaussian at every frame -- is what makes 4D
splatting unwatchable on the web: 400,000 gaussians at 32 bytes over 90 frames
is 1.15 GB, and baking only the moving foreground still costs 276 MB. This
format instead stores the canonical gaussians once and the motion as a small
scaffold of rigid nodes that each gaussian is skinned to, which is the
representation the trainer fits in the first place.

For a three-second clip at 30 fps with 400,000 gaussians, 150,000 of them
dynamic, and a 256-node scaffold, that is:

==========  ==========================  ==========
chunk       size                        bytes
==========  ==========================  ==========
``GEOM``    400,000 x 16                6,400,000
``COLR``    400,000 x 4                 1,600,000
``SKIN``    150,000 x 8                 1,200,000
``TRAJ``    90 x 256 x 14                 322,560
``NODE``    256 x 12                        3,072
==========  ==========================  ==========

about **9.5 MB** -- smaller than the 21.8 MB static ``.splat`` this project
already ships, and 29 times smaller than baking the foreground alone. The
motion is 3.4 % of the file. That ratio is the entire argument for the format.

Layout
------

A 192-byte header, then a 32-bytes-per-entry chunk directory, then the chunk
payloads, each starting on a 16-byte boundary so that a reader can take a typed
view over a chunk without copying -- which is what lets the viewer hand
``GEOM`` straight to ``texImage2D`` as ``RGBA32UI``, one texel per gaussian.

Chunks are addressed by a four-character tag. **Unknown tags are skipped**, so
a later version can add data without a version bump. Behaviour that a reader
cannot fake is instead announced in the low 16 bits of ``flags``: any bit set
there that this reader does not know is a refusal, not a warning. The high 16
bits are informational and are ignored when unrecognised.

Ordering
--------

Dynamic gaussians come first, so ``dynamic_count`` is one integer rather than a
per-gaussian flag, and the viewer's shader branches on the instance index.
Within each group gaussians are ordered by importance -- projected area times
opacity -- so that a device which cannot afford the whole scene drops the tail
rather than an arbitrary slice.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from gausscapture.errors import CaptureFormatError
from gausscapture.export.quantise import (
    GRID16_LEVELS,
    dequantise_grid,
    dequantise_weights,
    grid_step,
    pack_quaternion_smallest_three,
    pack_quaternions_snorm16,
    quantise_grid,
    quantise_weights,
    unpack_quaternion_smallest_three,
    unpack_quaternions_snorm16,
)

MAGIC = b"GAUSS4D\x00"
VERSION_MAJOR = 1
VERSION_MINOR = 0
HEADER_SIZE = 192
DIRECTORY_ENTRY_SIZE = 32
ALIGNMENT = 16

#: Bytes per gaussian in ``GEOM``.
GEOM_STRIDE = 16
#: Bytes per gaussian in ``COLR``.
COLR_STRIDE = 4
#: Bytes per node per frame in ``TRAJ``.
TRAJ_STRIDE = 14

#: The scaffold is capped so that a node index fits in one byte and the whole
#: per-frame node set fits in a single small uniform upload.
MAX_NODES = 256
#: More than four influences per gaussian buys nothing for a piecewise-rigid
#: head and costs the viewer a second texture fetch per gaussian.
MAX_SKIN_K = 4

# --- flags ------------------------------------------------------------------
# Low 16 bits are breaking: a reader that does not know the bit must refuse.
#: Reserved for a future residual-keyframe profile (architecture note D8). It is
#: deliberately unimplemented; the tag exists so that adding it later is not a
#: version bump.
FLAG_PROFILE_RESIDUAL = 1 << 0
_KNOWN_BREAKING = 0  # none implemented yet, so every low bit is a refusal

# High 16 bits are informational.
#: The fixed camera pose was verified against masked hold keyframes rather than
#: merely asserted from the rest-window cameras.
FLAG_FIXED_POSE_VERIFIED = 1 << 16
#: Playback should ping-pong rather than cut back to the first frame.
FLAG_LOOP_PINGPONG = 1 << 17

_HEADER_STRUCT = struct.Struct(
    "<8s"      # magic
    "HH"       # version major, minor
    "I"        # flags
    "II"       # header size, directory offset
    "I"        # chunk count
    "IIIII"    # gaussian, dynamic, node, frame counts, skin k
    "ff"       # fps, clip seconds
    "3f3f"     # position origin, span
    "ff"       # log-scale min, span
    "3f3f"     # trajectory translation origin, span
    "3f3f3f"   # capture eye, forward, up
    "ff"       # cone azimuth, elevation (half-angles, degrees)
    "3f"       # scene centre
    "f"        # scene radius
)
assert _HEADER_STRUCT.size <= HEADER_SIZE

_ENTRY_STRUCT = struct.Struct("<4sIQQQ")
assert _ENTRY_STRUCT.size == DIRECTORY_ENTRY_SIZE


@dataclass
class Scene4D:
    """A canonical gaussian set plus the scaffold that moves it.

    Positions, scales and quaternions describe the gaussians at rest. ``means``
    is in the same world frame as ``node_rest``; the deformation contract in
    :mod:`gausscapture.export.deform_reference` says exactly how the two
    combine, and both the trainer and the browser are validated against it.
    """

    means: np.ndarray            # (N, 3) float32
    scales: np.ndarray           # (N, 3) float32, linear (already exponentiated)
    quats: np.ndarray            # (N, 4) float32, (w, x, y, z), unit
    colors: np.ndarray           # (N, 4) uint8, RGBA, alpha carries opacity
    node_rest: np.ndarray        # (K, 3) float32
    node_quats: np.ndarray       # (T, K, 4) float32, (w, x, y, z), unit
    node_trans: np.ndarray       # (T, K, 3) float32
    skin_indices: np.ndarray     # (Nd, k) integer, indices into node_rest
    skin_weights: np.ndarray     # (Nd, k) float32, rows sum to one
    dynamic_count: int
    fps: float = 30.0
    cone_azimuth_deg: float = 0.0
    cone_elevation_deg: float = 0.0
    capture_eye: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    capture_forward: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -1.0], dtype=np.float32))
    capture_up: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float32))
    fixed_pose_verified: bool = False
    loop: str = "ping-pong"
    meta: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- shape

    @property
    def gaussian_count(self) -> int:
        return int(len(self.means))

    @property
    def node_count(self) -> int:
        return int(len(self.node_rest))

    @property
    def frame_count(self) -> int:
        return int(len(self.node_quats))

    @property
    def skin_k(self) -> int:
        return int(self.skin_indices.shape[1]) if self.skin_indices.size else 1

    @property
    def clip_seconds(self) -> float:
        """The span the file can actually be sampled over.

        ``frame_count - 1`` intervals, not ``frame_count``. The viewer's time
        cursor is a frame index clamped to ``[0, frame_count - 1]``, so the
        largest instant it can ever display is ``(frame_count - 1) / fps``.
        Reporting ``frame_count / fps`` made the page's readout stop one frame
        period short of the duration printed beside it -- while the scrubber,
        which divides by ``frame_count - 1``, did reach the end. A bar at 100 %
        next to a clock that never arrives reads as a stuck player.
        """
        if self.fps <= 0:
            return 0.0
        return max(self.frame_count - 1, 0) / self.fps

    def validate(self) -> None:
        """Refuse a scene whose arrays do not describe the same thing.

        Every check here is a shape or range mismatch that would otherwise
        write a well-formed file describing a scene nobody meant.
        """
        n = self.gaussian_count
        for name, array, shape in (
            ("means", self.means, (n, 3)),
            ("scales", self.scales, (n, 3)),
            ("quats", self.quats, (n, 4)),
            ("colors", self.colors, (n, 4)),
        ):
            if array.shape != shape:
                raise CaptureFormatError(
                    f"{name} has shape {array.shape}, expected {shape}"
                )
        if n == 0:
            raise CaptureFormatError("A 4D scene needs at least one gaussian")
        if not 0 <= self.dynamic_count <= n:
            raise CaptureFormatError(
                f"dynamic_count {self.dynamic_count} is outside 0..{n}"
            )
        if not 1 <= self.node_count <= MAX_NODES:
            raise CaptureFormatError(
                f"A scaffold must have 1..{MAX_NODES} nodes, got {self.node_count}"
            )
        if self.frame_count < 1:
            raise CaptureFormatError("A 4D scene needs at least one frame")
        if self.node_quats.shape != (self.frame_count, self.node_count, 4):
            raise CaptureFormatError(
                f"node_quats has shape {self.node_quats.shape}, expected "
                f"{(self.frame_count, self.node_count, 4)}"
            )
        if self.node_trans.shape != (self.frame_count, self.node_count, 3):
            raise CaptureFormatError(
                f"node_trans has shape {self.node_trans.shape}, expected "
                f"{(self.frame_count, self.node_count, 3)}"
            )
        if self.skin_indices.shape[0] != self.dynamic_count:
            raise CaptureFormatError(
                f"skin_indices has {self.skin_indices.shape[0]} rows but "
                f"{self.dynamic_count} gaussians are dynamic"
            )
        if self.skin_indices.shape != self.skin_weights.shape:
            raise CaptureFormatError("Skinning indices and weights must have the same shape")
        if self.dynamic_count and not 1 <= self.skin_k <= MAX_SKIN_K:
            raise CaptureFormatError(
                f"skin_k must be 1..{MAX_SKIN_K}, got {self.skin_k}"
            )
        if self.dynamic_count and (
            self.skin_indices.min() < 0 or self.skin_indices.max() >= self.node_count
        ):
            raise CaptureFormatError("A skinning index points outside the scaffold")
        if self.fps <= 0:
            raise CaptureFormatError("fps must be positive")

    # ------------------------------------------------------------- reshaping

    def ordered_by_importance(self) -> Scene4D:
        """Sort each group by projected area times opacity, dynamic group first.

        The order is what makes truncation graceful: a viewer that can only
        afford half the gaussians keeps the half that covers the most screen.
        """
        volume = np.prod(np.maximum(self.scales, 1e-12), axis=1)
        importance = volume * (self.colors[:, 3].astype(np.float64) / 255.0)

        dyn = np.argsort(-importance[: self.dynamic_count])
        sta = np.argsort(-importance[self.dynamic_count:]) + self.dynamic_count
        order = np.concatenate([dyn, sta]).astype(np.int64)

        return Scene4D(
            means=self.means[order],
            scales=self.scales[order],
            quats=self.quats[order],
            colors=self.colors[order],
            node_rest=self.node_rest,
            node_quats=self.node_quats,
            node_trans=self.node_trans,
            skin_indices=self.skin_indices[dyn] if self.dynamic_count else self.skin_indices,
            skin_weights=self.skin_weights[dyn] if self.dynamic_count else self.skin_weights,
            dynamic_count=self.dynamic_count,
            fps=self.fps,
            cone_azimuth_deg=self.cone_azimuth_deg,
            cone_elevation_deg=self.cone_elevation_deg,
            capture_eye=self.capture_eye,
            capture_forward=self.capture_forward,
            capture_up=self.capture_up,
            fixed_pose_verified=self.fixed_pose_verified,
            loop=self.loop,
            meta=dict(self.meta),
        )

    def truncated(self, max_gaussians: int) -> Scene4D:
        """Keep the most important ``max_gaussians``, split across both groups.

        Both groups are cut in proportion rather than the static tail alone:
        dropping only background would leave a subject floating in nothing, and
        dropping only foreground would defeat the point of the recording. The
        caller is expected to record in ``scene.json`` that this happened --
        a silently truncated scene is a quality claim nobody made.
        """
        total = self.gaussian_count
        if max_gaussians >= total:
            return self
        max_gaussians = max(1, int(max_gaussians))
        fraction = max_gaussians / total
        keep_dynamic = min(self.dynamic_count, max(1, int(round(self.dynamic_count * fraction))))
        keep_static = max(0, max_gaussians - keep_dynamic)
        keep_static = min(keep_static, total - self.dynamic_count)

        index = np.concatenate([
            np.arange(keep_dynamic),
            self.dynamic_count + np.arange(keep_static),
        ]).astype(np.int64)

        return Scene4D(
            means=self.means[index],
            scales=self.scales[index],
            quats=self.quats[index],
            colors=self.colors[index],
            node_rest=self.node_rest,
            node_quats=self.node_quats,
            node_trans=self.node_trans,
            skin_indices=self.skin_indices[:keep_dynamic],
            skin_weights=self.skin_weights[:keep_dynamic],
            dynamic_count=keep_dynamic,
            fps=self.fps,
            cone_azimuth_deg=self.cone_azimuth_deg,
            cone_elevation_deg=self.cone_elevation_deg,
            capture_eye=self.capture_eye,
            capture_forward=self.capture_forward,
            capture_up=self.capture_up,
            fixed_pose_verified=self.fixed_pose_verified,
            loop=self.loop,
            meta=dict(self.meta),
        )


def _hemisphere_continuous(node_quats: np.ndarray) -> np.ndarray:
    """Flip trajectory quaternions so consecutive frames take the short arc.

    ``q`` and ``-q`` are the same rotation, but a trainer is free to emit
    either, and the viewer interpolates by normalised linear blend rather than
    by slerp. Blending across a sign flip sweeps the long way round: on a head
    that is a whole-body spin between two adjacent frames. Fixing it once here
    means neither the browser nor the reference implementation has to.
    """
    quats = np.array(node_quats, dtype=np.float64, copy=True)
    for frame in range(1, len(quats)):
        flip = np.sum(quats[frame] * quats[frame - 1], axis=1) < 0.0
        quats[frame][flip] *= -1.0
    return quats


def _check_position_grid(span: np.ndarray, scales: np.ndarray) -> float:
    """Refuse a position grid too coarse to hold the gaussians it describes.

    16 bits over the scene's bounding box is ample for a room, and hopeless for
    a room that happens to contain one stray gaussian a kilometre away -- the
    box stretches, the step grows, and every gaussian in the subject lands on
    the same few grid points. Comparing the step against the *median* gaussian
    size catches exactly that case, because the median is the one statistic an
    outlier cannot move.
    """
    step = float(np.max(grid_step(span)))
    median_scale = float(np.median(scales))
    if median_scale > 0 and step >= median_scale / 4.0:
        raise CaptureFormatError(
            f"The position grid step is {step:.6g} but the median gaussian scale is "
            f"{median_scale:.6g}; quantisation would be visible. The scene almost "
            "certainly contains outlying gaussians stretching its bounding box."
        )
    return step


def _pack_geometry(scene: Scene4D) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the ``GEOM`` chunk: one 16-byte, texel-aligned record per gaussian."""
    pos_codes, pos_origin, pos_span = quantise_grid(scene.means)
    _check_position_grid(pos_span, scene.scales)

    log_scales = np.log(np.maximum(np.asarray(scene.scales, dtype=np.float64), 1e-12))
    lo = float(log_scales.min())
    span = float(max(log_scales.max() - lo, 1e-9))
    scale_codes, _, _ = quantise_grid(
        log_scales, origin=np.full(3, lo), span=np.full(3, span)
    )

    quat_codes = pack_quaternion_smallest_three(scene.quats)

    words = np.zeros((scene.gaussian_count, 4), dtype=np.uint32)
    pos = pos_codes.astype(np.uint32)
    sca = scale_codes.astype(np.uint32)
    words[:, 0] = pos[:, 0] | (pos[:, 1] << np.uint32(16))
    words[:, 1] = pos[:, 2] | (sca[:, 0] << np.uint32(16))
    words[:, 2] = sca[:, 1] | (sca[:, 2] << np.uint32(16))
    words[:, 3] = quat_codes

    grid = {
        "pos_origin": pos_origin,
        "pos_span": pos_span,
        "log_scale_min": lo,
        "log_scale_span": span,
    }
    return words.reshape(-1).view(np.uint8), grid


def _unpack_geometry(
    payload: np.ndarray, count: int, grid: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    words = payload.view(np.uint32).reshape(count, 4)
    pos = np.stack([
        words[:, 0] & np.uint32(0xFFFF),
        (words[:, 0] >> np.uint32(16)) & np.uint32(0xFFFF),
        words[:, 1] & np.uint32(0xFFFF),
    ], axis=1)
    sca = np.stack([
        (words[:, 1] >> np.uint32(16)) & np.uint32(0xFFFF),
        words[:, 2] & np.uint32(0xFFFF),
        (words[:, 2] >> np.uint32(16)) & np.uint32(0xFFFF),
    ], axis=1)

    means = dequantise_grid(pos, grid["pos_origin"], grid["pos_span"])
    log_scales = dequantise_grid(
        sca,
        np.full(3, grid["log_scale_min"]),
        np.full(3, grid["log_scale_span"]),
    )
    quats = unpack_quaternion_smallest_three(np.ascontiguousarray(words[:, 3]))
    return means, np.exp(log_scales).astype(np.float32), quats


def _pack_trajectories(scene: Scene4D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the ``TRAJ`` chunk, frame-major, 14 bytes per node per frame."""
    quats = _hemisphere_continuous(scene.node_quats)
    quat_codes = pack_quaternions_snorm16(quats)

    trans = np.asarray(scene.node_trans, dtype=np.float64).reshape(-1, 3)
    origin = trans.min(axis=0)
    span = np.maximum(trans.max(axis=0) - origin, 1e-6)
    trans_codes, origin, span = quantise_grid(trans, origin=origin, span=span)

    frames, nodes = scene.frame_count, scene.node_count
    payload = np.zeros((frames * nodes, TRAJ_STRIDE), dtype=np.uint8)
    payload[:, 0:8] = quat_codes.reshape(-1, 4).view(np.uint8).reshape(-1, 8)
    payload[:, 8:14] = trans_codes.reshape(-1, 3).view(np.uint8).reshape(-1, 6)
    return payload.reshape(-1), origin, span


def _unpack_trajectories(
    payload: np.ndarray, frames: int, nodes: int, origin: np.ndarray, span: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rows = payload.reshape(frames * nodes, TRAJ_STRIDE)
    quat_codes = np.ascontiguousarray(rows[:, 0:8]).view(np.int16).reshape(-1, 4)
    trans_codes = np.ascontiguousarray(rows[:, 8:14]).view(np.uint16).reshape(-1, 3)
    quats = unpack_quaternions_snorm16(quat_codes).reshape(frames, nodes, 4)
    trans = dequantise_grid(trans_codes, origin, span).reshape(frames, nodes, 3)
    return quats, trans


def _align(offset: int) -> int:
    return (offset + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def write_scene4d(
    scene: Scene4D,
    path: Path,
    order_by_importance: bool = True,
    max_gaussians: int | None = None,
) -> dict[str, Any]:
    """Write ``scene`` as a ``.g4d`` file and report what it cost.

    The returned dictionary is what the viewer's ``scene.json`` sidecar is
    built from, including whether the scene had to be truncated to fit
    ``max_gaussians`` -- which the page must state, because a truncated scene
    is not the scene that was trained.
    """
    scene.validate()
    original_count = scene.gaussian_count
    if order_by_importance:
        scene = scene.ordered_by_importance()
    truncated = max_gaussians is not None and max_gaussians < original_count
    if truncated:
        scene = scene.truncated(int(max_gaussians))
        scene.validate()

    geom, grid = _pack_geometry(scene)
    colr = np.ascontiguousarray(scene.colors, dtype=np.uint8).reshape(-1)
    node = np.ascontiguousarray(scene.node_rest, dtype=np.float32).reshape(-1).view(np.uint8)
    traj, traj_origin, traj_span = _pack_trajectories(scene)

    if scene.dynamic_count:
        k = scene.skin_k
        # Rows are stored in descending weight order, which is a promise the
        # shader relies on: influence 0 is the hemisphere reference for the
        # rotation blend, so the GPU never has to search for the dominant
        # influence. Sorting here costs one argsort at export.
        weights = np.asarray(scene.skin_weights, dtype=np.float64)
        rank = np.argsort(-weights, axis=1, kind="stable")
        indices = np.take_along_axis(
            np.asarray(scene.skin_indices, dtype=np.int64), rank, axis=1
        )
        skin = np.zeros((scene.dynamic_count, 2 * k), dtype=np.uint8)
        skin[:, :k] = indices.astype(np.uint8)
        skin[:, k:] = quantise_weights(np.take_along_axis(weights, rank, axis=1))
        skin = skin.reshape(-1)
    else:
        skin = np.zeros(0, dtype=np.uint8)

    meta = dict(scene.meta)
    meta.setdefault("loop", scene.loop)
    meta_bytes = json.dumps(meta, separators=(",", ":"), sort_keys=True).encode("utf-8")

    chunks: list[tuple[bytes, bytes]] = [
        (b"GEOM", geom.tobytes()),
        (b"COLR", colr.tobytes()),
        (b"NODE", node.tobytes()),
        (b"TRAJ", traj.tobytes()),
        (b"SKIN", skin.tobytes()),
        (b"META", meta_bytes),
    ]

    directory_offset = HEADER_SIZE
    payload_offset = _align(directory_offset + DIRECTORY_ENTRY_SIZE * len(chunks))

    entries: list[bytes] = []
    body = bytearray()
    cursor = payload_offset
    for tag, blob in chunks:
        pad = _align(cursor) - cursor
        body.extend(b"\x00" * pad)
        cursor += pad
        entries.append(_ENTRY_STRUCT.pack(tag, 0, cursor, len(blob), 0))
        body.extend(blob)
        cursor += len(blob)

    flags = 0
    if scene.fixed_pose_verified:
        flags |= FLAG_FIXED_POSE_VERIFIED
    if scene.loop == "ping-pong":
        flags |= FLAG_LOOP_PINGPONG

    header = bytearray(HEADER_SIZE)
    header[: _HEADER_STRUCT.size] = _HEADER_STRUCT.pack(
        MAGIC, VERSION_MAJOR, VERSION_MINOR, flags,
        HEADER_SIZE, directory_offset, len(chunks),
        scene.gaussian_count, scene.dynamic_count, scene.node_count,
        scene.frame_count, scene.skin_k,
        float(scene.fps), float(scene.clip_seconds),
        *[float(v) for v in grid["pos_origin"]],
        *[float(v) for v in grid["pos_span"]],
        float(grid["log_scale_min"]), float(grid["log_scale_span"]),
        *[float(v) for v in traj_origin],
        *[float(v) for v in traj_span],
        *[float(v) for v in np.asarray(scene.capture_eye, dtype=np.float64).reshape(3)],
        *[float(v) for v in np.asarray(scene.capture_forward, dtype=np.float64).reshape(3)],
        *[float(v) for v in np.asarray(scene.capture_up, dtype=np.float64).reshape(3)],
        float(scene.cone_azimuth_deg), float(scene.cone_elevation_deg),
        *[float(v) for v in scene.means.mean(axis=0)],
        float(np.linalg.norm(scene.means.max(axis=0) - scene.means.min(axis=0)) / 2.0),
    )

    padding = b"\x00" * (payload_offset - directory_offset - DIRECTORY_ENTRY_SIZE * len(chunks))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(bytes(header))
        for entry in entries:
            handle.write(entry)
        handle.write(padding)
        handle.write(bytes(body))

    return {
        "file": path.name,
        "bytes": int(path.stat().st_size),
        "chunk_bytes": {tag.decode("ascii"): len(blob) for tag, blob in chunks},
        "gaussian_count": scene.gaussian_count,
        "dynamic_count": scene.dynamic_count,
        "node_count": scene.node_count,
        "frame_count": scene.frame_count,
        "skin_k": scene.skin_k,
        "fps": float(scene.fps),
        "clip_seconds": float(scene.clip_seconds),
        "truncated": bool(truncated),
        "trained_gaussian_count": int(original_count),
        "position_step": float(np.max(grid_step(grid["pos_span"]))),
        "bytes_per_gaussian": float(path.stat().st_size / max(scene.gaussian_count, 1)),
    }


@dataclass
class Scene4DHeader:
    """The header fields a reader needs before it touches a chunk."""

    version: tuple[int, int]
    flags: int
    gaussian_count: int
    dynamic_count: int
    node_count: int
    frame_count: int
    skin_k: int
    fps: float
    clip_seconds: float
    cone_azimuth_deg: float
    cone_elevation_deg: float
    capture_eye: np.ndarray
    capture_forward: np.ndarray
    capture_up: np.ndarray
    scene_centre: np.ndarray
    scene_radius: float
    chunks: dict[str, tuple[int, int]]
    #: The dequantisation constants for GEOM and TRAJ. Kept as a dictionary
    #: rather than as fields because nothing outside the reader needs them.
    grid: dict[str, Any] = field(default_factory=dict)

    @property
    def fixed_pose_verified(self) -> bool:
        return bool(self.flags & FLAG_FIXED_POSE_VERIFIED)

    @property
    def loop(self) -> str:
        return "ping-pong" if self.flags & FLAG_LOOP_PINGPONG else "forward"


def read_scene4d_header(data: bytes | Path) -> Scene4DHeader:
    """Parse the header and chunk directory, without decoding any payload."""
    if isinstance(data, str | Path):
        data = Path(data).read_bytes()
    if len(data) < HEADER_SIZE:
        raise CaptureFormatError("Not a .g4d file: shorter than its own header")

    fields = _HEADER_STRUCT.unpack_from(data, 0)
    if fields[0] != MAGIC:
        raise CaptureFormatError(
            f"Not a .g4d file: magic is {fields[0]!r}, expected {MAGIC!r}"
        )
    (_, major, minor, flags, header_size, directory_offset, chunk_count,
     gaussians, dynamic, nodes, frames, skin_k, fps, clip_seconds) = fields[:14]

    if major != VERSION_MAJOR:
        raise CaptureFormatError(
            f"This .g4d is version {major}.{minor}; this reader understands {VERSION_MAJOR}.x"
        )
    unknown = flags & 0xFFFF & ~_KNOWN_BREAKING
    if unknown:
        raise CaptureFormatError(
            f"This .g4d sets breaking flag bits 0x{unknown:04x} that this reader does not "
            "implement, so it cannot be drawn correctly. Refusing rather than guessing."
        )
    if header_size < HEADER_SIZE:
        raise CaptureFormatError(
            f"Header claims to be {header_size} bytes, minimum is {HEADER_SIZE}"
        )

    # The tail of the header is a run of small fixed-width vectors. Walking it
    # with an explicit cursor keeps the reader in the same order as the writer's
    # format string, which is the only thing stopping the two from drifting.
    cursor = 14

    def take_vector() -> np.ndarray:
        nonlocal cursor
        vector = np.array(fields[cursor:cursor + 3], dtype=np.float32)
        cursor += 3
        return vector

    def take_pair() -> tuple[float, float]:
        nonlocal cursor
        pair = (fields[cursor], fields[cursor + 1])
        cursor += 2
        return pair

    pos_origin = take_vector()
    pos_span = take_vector()
    log_scale_min, log_scale_span = take_pair()
    traj_origin = take_vector()
    traj_span = take_vector()
    capture_eye = take_vector()
    capture_forward = take_vector()
    capture_up = take_vector()
    cone_azimuth, cone_elevation = take_pair()
    scene_centre = take_vector()
    scene_radius = fields[cursor]

    chunks: dict[str, tuple[int, int]] = {}
    for i in range(chunk_count):
        start = directory_offset + i * DIRECTORY_ENTRY_SIZE
        if start + DIRECTORY_ENTRY_SIZE > len(data):
            raise CaptureFormatError("The chunk directory runs past the end of the file")
        tag, _entry_flags, offset, length = _ENTRY_STRUCT.unpack_from(data, start)[:4]
        if offset + length > len(data):
            raise CaptureFormatError(
                f"Chunk {tag.decode('ascii', 'replace')} claims bytes {offset}..{offset + length} "
                f"but the file is {len(data)} bytes; it is truncated."
            )
        if offset % ALIGNMENT:
            raise CaptureFormatError(
                f"Chunk {tag.decode('ascii', 'replace')} starts at {offset}, which is not "
                f"{ALIGNMENT}-byte aligned; a reader cannot take a typed view over it."
            )
        chunks[tag.decode("ascii", "replace")] = (offset, length)

    header = Scene4DHeader(
        version=(major, minor), flags=flags,
        gaussian_count=gaussians, dynamic_count=dynamic, node_count=nodes,
        frame_count=frames, skin_k=skin_k, fps=fps, clip_seconds=clip_seconds,
        cone_azimuth_deg=cone_azimuth, cone_elevation_deg=cone_elevation,
        capture_eye=capture_eye, capture_forward=capture_forward, capture_up=capture_up,
        scene_centre=scene_centre, scene_radius=scene_radius, chunks=chunks,
        grid={
            "pos_origin": pos_origin, "pos_span": pos_span,
            "log_scale_min": log_scale_min, "log_scale_span": log_scale_span,
            "traj_origin": traj_origin, "traj_span": traj_span,
        },
    )
    return header


def read_scene4d(path: Path | bytes) -> Scene4D:
    """Read a ``.g4d`` back into a :class:`Scene4D`, dequantised.

    Used by the tests to prove the round trip, and by anyone who wants to
    inspect a delivered scene without a browser.
    """
    data = Path(path).read_bytes() if isinstance(path, str | Path) else path
    header = read_scene4d_header(data)
    grid = header.grid
    buffer = np.frombuffer(data, dtype=np.uint8)

    def chunk(tag: str, stride: int, expected: int) -> np.ndarray:
        if tag not in header.chunks:
            raise CaptureFormatError(f"This .g4d has no {tag} chunk, so it is not drawable")
        offset, length = header.chunks[tag]
        if length != stride * expected:
            raise CaptureFormatError(
                f"Chunk {tag} is {length} bytes; the header implies {stride * expected}"
            )
        return buffer[offset:offset + length]

    n, k = header.gaussian_count, header.skin_k
    means, scales, quats = _unpack_geometry(
        np.ascontiguousarray(chunk("GEOM", GEOM_STRIDE, n)), n, grid
    )
    colors = chunk("COLR", COLR_STRIDE, n).reshape(n, 4).copy()
    node_rest = np.ascontiguousarray(
        chunk("NODE", 12, header.node_count)
    ).view(np.float32).reshape(header.node_count, 3).copy()
    node_quats, node_trans = _unpack_trajectories(
        np.ascontiguousarray(chunk("TRAJ", TRAJ_STRIDE, header.frame_count * header.node_count)),
        header.frame_count, header.node_count, grid["traj_origin"], grid["traj_span"],
    )

    dynamic = header.dynamic_count
    if dynamic:
        skin = chunk("SKIN", 2 * k, dynamic).reshape(dynamic, 2 * k)
        skin_indices = skin[:, :k].astype(np.int32)
        skin_weights = dequantise_weights(skin[:, k:])
    else:
        skin_indices = np.zeros((0, k), dtype=np.int32)
        skin_weights = np.zeros((0, k), dtype=np.float32)

    meta: dict[str, Any] = {}
    if "META" in header.chunks:
        offset, length = header.chunks["META"]
        meta = json.loads(bytes(buffer[offset:offset + length]).decode("utf-8"))

    return Scene4D(
        means=means, scales=scales, quats=quats, colors=colors,
        node_rest=node_rest, node_quats=node_quats, node_trans=node_trans,
        skin_indices=skin_indices, skin_weights=skin_weights,
        dynamic_count=dynamic, fps=float(header.fps),
        cone_azimuth_deg=float(header.cone_azimuth_deg),
        cone_elevation_deg=float(header.cone_elevation_deg),
        capture_eye=header.capture_eye, capture_forward=header.capture_forward,
        capture_up=header.capture_up,
        fixed_pose_verified=header.fixed_pose_verified,
        loop=header.loop, meta=meta,
    )


__all__ = [
    "COLR_STRIDE",
    "FLAG_FIXED_POSE_VERIFIED",
    "FLAG_LOOP_PINGPONG",
    "FLAG_PROFILE_RESIDUAL",
    "GEOM_STRIDE",
    "GRID16_LEVELS",
    "MAX_NODES",
    "MAX_SKIN_K",
    "TRAJ_STRIDE",
    "Scene4D",
    "Scene4DHeader",
    "read_scene4d",
    "read_scene4d_header",
    "write_scene4d",
]
