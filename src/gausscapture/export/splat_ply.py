"""Reading a trained Gaussian splat from a ``.ply``.

This is the format 3DGS trainers agree on, inherited from the original Inria
implementation and followed by gsplat, Brush, Postshot and the rest. It is a
binary little-endian PLY whose vertices carry, per gaussian:

* ``x y z`` — position
* ``f_dc_0..2`` — the zeroth spherical-harmonic band, which is the base colour
* ``f_rest_*`` — higher SH bands, view-dependent colour; usually 45 floats
* ``opacity`` — logit-encoded, so it needs a sigmoid
* ``scale_0..2`` — log-encoded, so they need an exponential
* ``rot_0..3`` — orientation quaternion

The two encodings are the part worth stating. Opacity and scale are stored
*before* their activation functions, exactly as the optimiser held them, so
reading them as literal values yields opacities outside [0, 1] and negative
scales. Nothing errors; the numbers are simply wrong.

Only the properties that a viewer needs are decoded. Full spherical harmonics
are read as a count, not as values: this project does not implement a splat
renderer, on the deliberate grounds that SuperSplat and Spark already do it
well (see ``docs/RESEARCH.md`` section 7.1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gausscapture.errors import CaptureFormatError

#: The constant relating the zeroth spherical-harmonic coefficient to colour,
#: 0.5/sqrt(pi). Every 3DGS implementation stores base colour as an SH DC term
#: rather than as RGB, so the conversion is not optional.
SH_C0 = 0.28209479177387814

_NUMPY_TYPES = {
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "ushort": "u2", "uint16": "u2", "short": "i2", "int16": "i2",
    "uint": "u4", "uint32": "u4", "int": "i4", "int32": "i4",
}


@dataclass
class GaussianSplat:
    """A trained splat, decoded far enough to be inspected and drawn as points."""

    positions: np.ndarray          # (N, 3) float32
    colors: np.ndarray             # (N, 3) uint8, base colour with SH0 applied
    opacities: np.ndarray          # (N,) float32 in [0, 1]
    scales: np.ndarray             # (N, 3) float32, world units
    #: (N, 4) float32 unit quaternions, (w, x, y, z). With the scales these give
    #: each gaussian its shape and orientation, which is what separates real
    #: splat rendering from drawing the centres as dots.
    rotations: np.ndarray | None = None
    sh_bands: int = 0              # number of higher-order SH coefficients present

    def __len__(self) -> int:
        return len(self.positions)

    def summary(self) -> dict[str, Any]:
        return {
            "gaussians": len(self),
            "sh_coefficients": self.sh_bands,
            "median_scale": float(np.median(self.scales)) if len(self) else 0.0,
            "mean_opacity": float(self.opacities.mean()) if len(self) else 0.0,
        }

    def visible(self, min_opacity: float = 0.02) -> GaussianSplat:
        """Drop gaussians too transparent to contribute.

        Training leaves a long tail of near-zero-opacity gaussians that the
        renderer never shows. Keeping them in a point-cloud preview would make
        the result look far noisier than it is.
        """
        keep = self.opacities >= min_opacity
        return GaussianSplat(
            positions=self.positions[keep],
            colors=self.colors[keep],
            opacities=self.opacities[keep],
            scales=self.scales[keep],
            rotations=None if self.rotations is None else self.rotations[keep],
            sh_bands=self.sh_bands,
        )


def read_splat_ply(path: Path) -> GaussianSplat:
    """Read a Gaussian splat ``.ply``, applying the stored activations."""
    path = Path(path)
    with path.open("rb") as handle:
        header, count, dtype, big_endian = _read_header(handle)
        raw = np.frombuffer(handle.read(count * dtype.itemsize), dtype=dtype, count=count)

    if big_endian:
        raw = raw.byteswap().view(raw.dtype.newbyteorder())

    names = set(raw.dtype.names or ())
    for required in ("x", "y", "z"):
        if required not in names:
            raise CaptureFormatError(
                f"{path.name} has no '{required}' property; this is not a point PLY"
            )
    if "opacity" not in names or "scale_0" not in names:
        raise CaptureFormatError(
            f"{path.name} has positions but no opacity/scale, so it is a plain point cloud "
            "rather than a Gaussian splat"
        )

    positions = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float32)

    # Opacity is stored as a logit and scale as a logarithm -- exactly as the
    # optimiser held them. Reading either literally yields nonsense that does
    # not raise: opacities outside [0, 1], and negative sizes.
    opacities = _sigmoid(raw["opacity"].astype(np.float32))
    scales = np.exp(
        np.stack([raw[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float32)
    )

    if {"f_dc_0", "f_dc_1", "f_dc_2"} <= names:
        dc = np.stack([raw[f"f_dc_{i}"] for i in range(3)], axis=1).astype(np.float32)
        colors = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)
    elif {"red", "green", "blue"} <= names:
        colors = np.stack([raw[c] for c in ("red", "green", "blue")], axis=1) / 255.0
    else:
        colors = np.full((len(positions), 3), 0.5, dtype=np.float32)

    sh_bands = len([n for n in names if n.startswith("f_rest_")])

    rotations = None
    if all(f"rot_{i}" in names for i in range(4)):
        quats = np.stack([raw[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float32)
        # Stored unnormalised -- the optimiser never constrains them -- so a
        # renderer that skips this gets subtly misshapen gaussians.
        norms = np.linalg.norm(quats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        rotations = quats / norms

    return GaussianSplat(
        positions=positions,
        colors=(colors * 255).astype(np.uint8),
        opacities=opacities.astype(np.float32),
        scales=scales,
        rotations=rotations,
        sh_bands=sh_bands,
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Clipped so a saturated logit does not overflow exp().
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _read_header(handle) -> tuple[list[str], int, np.dtype, bool]:
    """Parse a binary PLY header, returning the vertex dtype in file order."""
    magic = handle.readline().strip()
    if magic != b"ply":
        raise CaptureFormatError("Not a PLY file")

    lines: list[str] = []
    count = 0
    fields: list[tuple[str, str]] = []
    big_endian = False
    in_vertex = False

    while True:
        line = handle.readline()
        if not line:
            raise CaptureFormatError("PLY header ended without end_header")
        text = line.decode("ascii", errors="replace").strip()
        lines.append(text)

        if text.startswith("format"):
            if "ascii" in text:
                raise CaptureFormatError(
                    "ASCII PLY is not supported; splat trainers write binary"
                )
            big_endian = "big_endian" in text
        elif text.startswith("element"):
            parts = text.split()
            in_vertex = len(parts) >= 3 and parts[1] == "vertex"
            if in_vertex:
                count = int(parts[2])
        elif text.startswith("property") and in_vertex:
            match = re.match(r"property\s+(\S+)\s+(\S+)", text)
            if match:
                kind, name = match.group(1), match.group(2)
                if kind not in _NUMPY_TYPES:
                    raise CaptureFormatError(f"Unsupported PLY property type: {kind}")
                fields.append((name, _NUMPY_TYPES[kind]))
        elif text == "end_header":
            break

    if not fields or count == 0:
        raise CaptureFormatError("PLY declares no vertices")
    return lines, count, np.dtype(fields), big_endian
