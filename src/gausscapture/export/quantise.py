"""Fixed-point encodings shared by the ``.g4d`` writer, reader and viewer.

These live in their own module because three independent implementations have
to agree on them *bit for bit*: the Python writer, the Python reader that the
tests round-trip through, and the GLSL/JavaScript decoder in
``report/viewer_4d.js``. A rounding rule that differs by one unit in the last
place is invisible in a unit test of either side alone and shows up in the
browser as geometry that is subtly the wrong shape.

Three encodings carry the whole format:

**Grid-quantised vectors.** A value becomes ``origin + code / (2**bits - 1) *
span``. The grid is derived from the data's own bounding box, so precision
scales with the scene rather than with an assumed unit. The writer asserts that
the resulting step is small against the median gaussian size -- position error
that is a noticeable fraction of a gaussian's own extent is the one
quantisation artefact that is actually visible.

**Smallest-three quaternions in 32 bits.** Two bits name the largest-magnitude
component, which is then reconstructed from the unit norm; the other three are
stored at 10 bits each over ``[-1/sqrt(2), 1/sqrt(2)]``, which is the widest
they can be once the largest is known.

The 10-bit grid is *signed and symmetric*: code ``511`` is exactly zero, and a
component is ``(code - 511) / 511 * (1/sqrt(2))``. The obvious alternative --
spreading 1024 codes evenly over the closed interval -- puts zero at code
511.5, which is not a code at all. The identity quaternion then round-trips
into three components of 7e-4 each and comes back 0.137 degrees away from
identity, which is a rest pose that is visibly not at rest. Sacrificing one of
the 1024 codes to make zero exact is the cheapest fix there is, and it makes
every axis-aligned rotation (0, ±1/sqrt(2), ±1) exact as well, which is most of
the rotations a hand-authored test or a rest scaffold ever contains.

Measured over 200,000 random unit quaternions the round trip is 0.084 degrees
mean and 0.240 degrees worst case. That is five times better than the
8-bit-per-component encoding this project already ships in ``.splat`` and is
far below anything visible, but it is *not* the 0.1-degree worst case the
architecture note assumed: 30 bits of mantissa cannot deliver it, because the
derived largest component absorbs the error of all three stored ones.

**Signed-normalised 16-bit quaternions.** Node trajectories are decoded once
per frame on the CPU for at most 256 nodes, so there is no reason to pay
smallest-three's unpack cost there. Four ``int16`` are simpler, are exact about
sign -- which matters because trajectories are stored hemisphere-continuous so
that ``nlerp`` between neighbouring frames takes the short way round -- and
resolve 0.0032 degrees worst case over 5,000 random quaternions, against a
budget of 0.01.

A warning about measuring any of this. The natural yardstick,
``2 * arccos(|dot(a, b)|)``, is worthless at these magnitudes: ``arccos`` has
an infinite derivative at 1, so a dot product that is correct to float64 still
yields an angle correct to only about half its digits, and a dot product
rounded through float32 -- which is what a decoder returns -- yields an angle
wrong by an order of magnitude. It reported 0.0335 degrees for the 16-bit
encoding whose true error is 0.0032, and it returned exactly zero for half the
sample because the dot product clipped to 1. Use the relative quaternion and
``atan2`` instead; :func:`gausscapture.export.quantise.angular_error_degrees`
is here so that the tests and any future caller share one that is stable.
"""

from __future__ import annotations

import numpy as np

from gausscapture.errors import CaptureFormatError

#: 1 / sqrt(2): the largest a non-largest quaternion component can be.
#:
#: Written ``sqrt(0.5)`` rather than ``1 / sqrt(2)`` because the latter rounds
#: twice -- once for the root, once for the reciprocal -- and lands one unit in
#: the last place below the correctly rounded double. JavaScript's
#: ``Math.SQRT1_2`` and the literal in the shader are the correctly rounded one,
#: so the obvious spelling is the one that makes the three implementations
#: disagree in the last bit, which is precisely what this module exists to stop.
INV_SQRT2 = float(np.sqrt(0.5))

#: Half-range of the signed 10-bit smallest-three grid. Codes run 0..1022 with
#: 511 meaning exactly zero; code 1023 is deliberately never emitted, which is
#: the price of putting a code on zero. See the module docstring.
SMALLEST_THREE_HALF = 511

#: Levels available to a 16-bit grid code.
GRID16_LEVELS = 65535

#: The largest an ``int16`` signed-normalised code may be. 32767 rather than
#: 32768 so that +1 and -1 are both exactly representable.
SNORM16_SCALE = 32767.0


def angular_error_degrees(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle in degrees between two rotations, ignoring the ``q``/``-q`` sign.

    Computed from the relative quaternion ``a^-1 b`` as
    ``2 * atan2(|vec|, |w|)`` rather than from ``2 * arccos(|dot|)``. The
    arccos form is catastrophically ill-conditioned exactly where a
    quantisation test lives: near zero angle ``d(arccos)/dx`` diverges, so it
    turns the last few bits of the dot product into the leading digits of the
    answer. Measuring the 16-bit trajectory encoding with it gave 0.0335
    degrees where the truth is 0.0032, and gave a flat zero for the half of
    the sample whose dot product rounded to 1. This form has bounded
    sensitivity everywhere and needs no clipping.
    """
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    # a^-1 = conj(a) for unit a; only the imaginary magnitude and |w| are used.
    rw = aw * bw + ax * bx + ay * by + az * bz
    rx = aw * bx - ax * bw - ay * bz + az * by
    ry = aw * by + ax * bz - ay * bw - az * bx
    rz = aw * bz - ax * by + ay * bx - az * bw
    imaginary = np.sqrt(rx * rx + ry * ry + rz * rz)
    return np.degrees(2.0 * np.arctan2(imaginary, np.abs(rw)))


def quantise_grid(
    values: np.ndarray,
    origin: np.ndarray | None = None,
    span: np.ndarray | None = None,
    levels: int = GRID16_LEVELS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map ``values`` onto an integer grid derived from their bounding box.

    Returns ``(codes, origin, span)``. ``span`` is never zero in any axis: a
    perfectly planar scene would otherwise produce a division by zero rather
    than a plane, and a degenerate axis costs one wasted bit and nothing else.
    """
    values = np.asarray(values, dtype=np.float64)
    if origin is None:
        origin = values.min(axis=0)
    if span is None:
        span = values.max(axis=0) - origin
    origin = np.asarray(origin, dtype=np.float64)
    span = np.maximum(np.asarray(span, dtype=np.float64), 1e-9)

    normalised = (values - origin) / span
    codes = np.rint(np.clip(normalised, 0.0, 1.0) * levels)
    dtype = np.uint16 if levels <= GRID16_LEVELS else np.uint32
    return codes.astype(dtype), origin.astype(np.float32), span.astype(np.float32)


def dequantise_grid(
    codes: np.ndarray,
    origin: np.ndarray,
    span: np.ndarray,
    levels: int = GRID16_LEVELS,
) -> np.ndarray:
    """Invert :func:`quantise_grid`."""
    codes = np.asarray(codes, dtype=np.float64)
    return (np.asarray(origin, dtype=np.float64)
            + codes / levels * np.asarray(span, dtype=np.float64)).astype(np.float32)


def grid_step(span: np.ndarray, levels: int = GRID16_LEVELS) -> np.ndarray:
    """The distance one code unit covers, per axis."""
    return np.asarray(span, dtype=np.float64) / levels


def pack_quaternion_smallest_three(quats: np.ndarray) -> np.ndarray:
    """Encode unit quaternions ``(w, x, y, z)`` as one ``uint32`` each.

    Bits 30-31 hold the index of the largest-magnitude component; bits 20-29,
    10-19 and 0-9 hold the remaining three in ascending component order. The
    quaternion is negated when its largest component is negative, which is free
    because ``q`` and ``-q`` are the same rotation -- and which is what makes
    the dropped component's sign recoverable at all.

    Each stored component is ``round(v / (1/sqrt(2)) * 511) + 511``, so zero is
    code 511 exactly. Once the largest component is factored out no other one
    can exceed ``1/sqrt(2)`` in magnitude, so no clip is ever reached in
    practice; the clip is there for an input that was not quite unit.
    """
    quats = np.atleast_2d(np.asarray(quats, dtype=np.float64))
    if quats.shape[-1] != 4:
        raise CaptureFormatError("Quaternions must have four components (w, x, y, z)")

    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / np.maximum(norms, 1e-12)

    largest = np.argmax(np.abs(quats), axis=1)
    rows = np.arange(len(quats))
    sign = np.where(quats[rows, largest] < 0.0, -1.0, 1.0)
    quats = quats * sign[:, None]

    # The three components that are not the largest, in ascending index order.
    order = np.argsort(np.where(np.arange(4)[None, :] == largest[:, None], 4, np.arange(4)),
                       axis=1)[:, :3]
    kept = np.take_along_axis(quats, order, axis=1)

    codes = np.rint(np.clip(kept / INV_SQRT2, -1.0, 1.0) * SMALLEST_THREE_HALF)
    codes = (codes + SMALLEST_THREE_HALF).astype(np.uint32)

    packed = (largest.astype(np.uint32) << np.uint32(30))
    packed |= codes[:, 0] << np.uint32(20)
    packed |= codes[:, 1] << np.uint32(10)
    packed |= codes[:, 2]
    return packed.astype(np.uint32)


def unpack_quaternion_smallest_three(packed: np.ndarray) -> np.ndarray:
    """Invert :func:`pack_quaternion_smallest_three`, returning unit ``(w,x,y,z)``."""
    packed = np.asarray(packed, dtype=np.uint32).reshape(-1)
    largest = (packed >> np.uint32(30)) & np.uint32(3)
    codes = np.stack([
        (packed >> np.uint32(20)) & np.uint32(1023),
        (packed >> np.uint32(10)) & np.uint32(1023),
        packed & np.uint32(1023),
    ], axis=1).astype(np.float64)

    kept = (codes - SMALLEST_THREE_HALF) / SMALLEST_THREE_HALF * INV_SQRT2
    rest = np.sqrt(np.maximum(0.0, 1.0 - np.sum(kept**2, axis=1)))

    order = np.argsort(np.where(np.arange(4)[None, :] == largest[:, None], 4, np.arange(4)),
                       axis=1)[:, :3]
    out = np.zeros((len(packed), 4), dtype=np.float64)
    np.put_along_axis(out, order, kept, axis=1)
    out[np.arange(len(packed)), largest] = rest
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return out.astype(np.float32)


def pack_quaternions_snorm16(quats: np.ndarray) -> np.ndarray:
    """Encode unit quaternions as four ``int16`` scaled by 32767.

    Unlike :func:`pack_quaternion_smallest_three` this preserves the sign of
    the whole quaternion, which the trajectory chunk depends on.

    The budget: a component is stored to within half a step, ``0.5 / 32767 =
    1.53e-5``, so the error vector has magnitude at most ``2 * 1.53e-5``, and
    the resulting rotation error is at most twice that, ``6.1e-5`` rad =
    0.0035 degrees. Measured worst case over 5,000 random quaternions is
    0.0032 degrees. Both sit comfortably under the 0.01-degree budget the
    format claims, so 16 bits per component is the right width and the field
    does not need widening.
    """
    quats = np.asarray(quats, dtype=np.float64)
    if quats.shape[-1] != 4:
        raise CaptureFormatError("Quaternions must have four components (w, x, y, z)")
    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    quats = quats / np.maximum(norms, 1e-12)
    return np.clip(
        np.rint(quats * SNORM16_SCALE), -SNORM16_SCALE, SNORM16_SCALE
    ).astype(np.int16)


def unpack_quaternions_snorm16(codes: np.ndarray) -> np.ndarray:
    """Invert :func:`pack_quaternions_snorm16`."""
    quats = np.asarray(codes, dtype=np.float64) / SNORM16_SCALE
    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    return (quats / np.maximum(norms, 1e-12)).astype(np.float32)


def quantise_weights(weights: np.ndarray) -> np.ndarray:
    """Quantise skinning weights to ``uint8`` rows that sum to exactly 255.

    Rounding each weight independently gives rows summing to anything from 252
    to 258, and a row that does not sum to 255 is a gaussian that is uniformly
    brighter or darker than its neighbours after the blend -- a visible seam
    along whichever part of the subject the scaffold happens to divide. The
    largest-remainder method fixes the sum exactly: floor everything, then hand
    the leftover units to the entries whose fractional parts were largest.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2:
        raise CaptureFormatError("Skinning weights must be a (gaussians, k) array")
    if weights.shape[1] == 0:
        raise CaptureFormatError("Skinning weights need at least one influence per gaussian")

    totals = weights.sum(axis=1, keepdims=True)
    # A gaussian bound to nothing is given entirely to its first influence
    # rather than silently producing NaNs.
    safe = np.where(totals > 1e-12, weights / np.maximum(totals, 1e-12), 0.0)
    safe[totals[:, 0] <= 1e-12, 0] = 1.0

    scaled = safe * 255.0
    floors = np.floor(scaled)
    remainder = (255 - floors.sum(axis=1)).astype(np.int64)

    # Rank fractional parts descending; ties break on the lower column index so
    # the result does not depend on the sort's stability guarantees.
    fractions = scaled - floors
    rank = np.argsort(-fractions, axis=1, kind="stable")
    bonus = np.zeros_like(floors)
    columns = np.arange(weights.shape[1])[None, :]
    np.put_along_axis(bonus, rank, (columns < remainder[:, None]).astype(np.float64), axis=1)

    return (floors + bonus).astype(np.uint8)


def dequantise_weights(codes: np.ndarray) -> np.ndarray:
    """Invert :func:`quantise_weights` back to weights summing to one."""
    return (np.asarray(codes, dtype=np.float32) / 255.0).astype(np.float32)
