#!/usr/bin/env python3
"""Generate the GaussCapture launcher icon, and every asset Android wants of it.

The mark is the thing this app produces, drawn the way the renderer actually draws it. A Gaussian
splat is a soft anisotropic ellipse -- not a circle, not a sprite with an edge -- and a 4D capture
is that ellipse at several instants. So the icon is one elliptical Gaussian shown three times: the
leading instance at full strength, the two behind it fading, offset across their own long axis. It
says "splats" and "it moves" in one shape and survives being 48 pixels across.

Three decisions in the geometry are not taste.

The cluster is *tilted*. Three horizontal lobes is a hamburger menu, and an icon that has to be
disambiguated from a navigation control at a glance has already lost. Twenty-two degrees is enough
to break that reading and not enough to look accidental.

The lobes are *parallel and identical*, not fanned. A fan reads as an arrowhead -- a direction, a
play button -- and this app does not play anything. Repeating one shape unchanged is what says
"the same thing, later", which is exactly what a 4D capture is.

The cluster is centred on its own coverage centroid, not on its geometry. The leading lobe is
opaque and the trailing ones are not, so a geometrically centred cluster sits visibly high in the
frame; the mass is what the eye centres on.

Every value is evaluated analytically rather than drawn with a blur filter, for the same reason the
viewer does: a Gaussian has no edge, and an ellipse with a feathered outline is a different object
that happens to look similar at one size. Compositing is back-to-front `over`, which is what a
splat rasteriser does.

Colour comes from ``app/lib/design.dart`` and nowhere else: amber ``#FFB020`` on near-black
``#0B0F14``. Amber is the app's single accent, so an icon in any other hue would be an icon for a
different app.

Run it from anywhere; it writes into ``app/android/app/src/main/res``:

    python3 app/tool/make_launcher_icon.py

Needs only Pillow. Re-run it after changing anything here and commit the PNGs alongside; the
generated files are checked in because a build must not depend on this script having been run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

# --------------------------------------------------------------------------- palette
# These four values are Palette.base, Palette.surface, Palette.warn and a lightened core of the
# same amber. Keep them in step with app/lib/design.dart.
BASE = (0x0B, 0x0F, 0x14)
SURFACE = (0x12, 0x18, 0x20)
AMBER = (0xFF, 0xB0, 0x20)
AMBER_CORE = (0xFF, 0xE3, 0xA8)


@dataclass(frozen=True)
class Splat:
    """One anisotropic Gaussian, in a unit design box spanning -0.5 to +0.5 on both axes.

    ``sigma_major``/``sigma_minor`` are the standard deviations along and across the ellipse's own
    axes; ``theta`` rotates it. ``alpha`` is peak opacity, which is what carries time: the leading
    splat is opaque and the ones behind it are the same shape seen a moment earlier.
    """

    cx: float
    cy: float
    sigma_major: float
    sigma_minor: float
    theta_deg: float
    alpha: float


#: Tilt of the whole cluster, and the spacing between instants, in design-box units.
CLUSTER_TILT_DEG = -22.0
CLUSTER_STEP = 0.150

#: Oldest first, so the list reads in the order it is composited and in the order time ran.
#: Each instant is very slightly larger than the one before it, which is the only concession to
#: perspective: the leading splat is the one being looked at.
INSTANTS = (
    (0.132, 0.043, 0.26),
    (0.140, 0.046, 0.54),
    (0.147, 0.049, 1.00),
)


def _cluster() -> tuple[Splat, ...]:
    """The three instants, laid out across the tilt and recentred on their coverage centroid.

    Offsets are perpendicular to each ellipse's own long axis, so the instants slide across
    themselves rather than along -- along would pile them into one longer ellipse and say nothing.

    The centroid is the alpha-weighted first moment of the three Gaussians. Integrating a Gaussian
    gives ``2 pi sigma_major sigma_minor``, so the weight of an instant is that times its peak
    opacity; the rotation drops out because it does not move a Gaussian's centre. This is exact for
    additive compositing and near enough under `over` at these opacities.
    """
    perpendicular = math.radians(CLUSTER_TILT_DEG + 90.0)
    dx, dy = math.cos(perpendicular), math.sin(perpendicular)
    middle = (len(INSTANTS) - 1) / 2.0

    placed = []
    for index, (major, minor, alpha) in enumerate(INSTANTS):
        offset = (middle - index) * CLUSTER_STEP
        placed.append((offset * dx, offset * dy, major, minor, alpha))

    mass = sum(major * minor * alpha for _, _, major, minor, alpha in placed)
    cx = sum(x * major * minor * alpha for x, _, major, minor, alpha in placed) / mass
    cy = sum(y * major * minor * alpha for _, y, major, minor, alpha in placed) / mass

    return tuple(
        Splat(
            cx=x - cx,
            cy=y - cy,
            sigma_major=major,
            sigma_minor=minor,
            theta_deg=CLUSTER_TILT_DEG,
            alpha=alpha,
        )
        for x, y, major, minor, alpha in placed
    )


#: Back to front, which is both the compositing order and the order time ran.
SPLATS = _cluster()

#: Beyond this many standard deviations a Gaussian contributes less than one 8-bit code.
CUTOFF_SIGMAS = 3.2


def _splat_field(size: int, box: float) -> tuple[list[float], list[float]]:
    """Composite coverage and leading-splat weight, one value per pixel, row-major.

    ``box`` is the design box as a fraction of the canvas. Returns ``(alpha, core)``: alpha is the
    over-composited coverage of all three splats, core is the leading splat alone, which is what
    lightens the centre so the mark reads as luminous rather than flat.
    """
    scale = size * box
    centre = size / 2.0

    prepared = []
    for splat in SPLATS:
        angle = math.radians(splat.theta_deg)
        prepared.append(
            (
                centre + splat.cx * scale,
                centre + splat.cy * scale,
                math.cos(angle),
                math.sin(angle),
                # Two divisions per pixel saved by folding the 1/(2 sigma^2) in here.
                1.0 / (2.0 * (splat.sigma_major * scale) ** 2),
                1.0 / (2.0 * (splat.sigma_minor * scale) ** 2),
                splat.alpha,
                CUTOFF_SIGMAS * splat.sigma_major * scale,
            )
        )

    alpha = [0.0] * (size * size)
    core = [0.0] * (size * size)
    for y in range(size):
        row = y * size
        py = y + 0.5
        for x in range(size):
            px = x + 0.5
            transmittance = 1.0
            leading = 0.0
            for cx, cy, cos_t, sin_t, inv_major, inv_minor, peak, reach in prepared:
                dx = px - cx
                dy = py - cy
                if abs(dx) > reach or abs(dy) > reach:
                    continue
                # Into the ellipse's own frame, then a plain separable Gaussian.
                u = dx * cos_t + dy * sin_t
                v = -dx * sin_t + dy * cos_t
                power = u * u * inv_major + v * v * inv_minor
                if power > 12.0:
                    continue
                coverage = peak * math.exp(-power)
                transmittance *= 1.0 - coverage
                leading = coverage / peak
            alpha[row + x] = 1.0 - transmittance
            core[row + x] = leading
    return alpha, core


def _mark(size: int, box: float, tint: tuple[int, int, int] | None) -> Image.Image:
    """The splat cluster on transparency. ``tint`` of None means the amber-with-a-core treatment."""
    alpha, core = _splat_field(size, box)
    pixels = bytearray(size * size * 4)
    for i in range(size * size):
        a = alpha[i]
        if a <= 0.0:
            continue
        if tint is None:
            # Lighten towards the core of the leading splat. Same hue throughout -- only the
            # lightness moves, so the icon still reads as one colour.
            lift = min(1.0, core[i]) ** 1.6 * 0.85
            colour = tuple(
                int(round(AMBER[c] + (AMBER_CORE[c] - AMBER[c]) * lift)) for c in range(3)
            )
        else:
            colour = tint
        j = i * 4
        pixels[j] = colour[0]
        pixels[j + 1] = colour[1]
        pixels[j + 2] = colour[2]
        pixels[j + 3] = int(round(min(1.0, a) * 255))
    return Image.frombytes("RGBA", (size, size), bytes(pixels))


def _background(size: int) -> Image.Image:
    """Near-black with a soft central lift, so the mark sits on depth rather than on a flat field."""
    pixels = bytearray(size * size * 4)
    centre = size / 2.0
    radius = size * 0.62
    for y in range(size):
        for x in range(size):
            d = math.hypot(x + 0.5 - centre, y + 0.5 - centre) / radius
            t = max(0.0, 1.0 - d) ** 2
            j = (y * size + x) * 4
            for c in range(3):
                pixels[j + c] = int(round(BASE[c] + (SURFACE[c] - BASE[c]) * t))
            pixels[j + 3] = 255
    return Image.frombytes("RGBA", (size, size), bytes(pixels))


def _rounded_mask(size: int, radius_fraction: float, supersample: int = 4) -> Image.Image:
    """A rounded-square alpha mask, drawn large and shrunk so its corners are not stepped."""
    big = size * supersample
    radius = big * radius_fraction
    mask = Image.new("L", (big, big), 0)
    pixels = mask.load()
    assert pixels is not None
    for y in range(big):
        for x in range(big):
            dx = max(radius - (x + 0.5), (x + 0.5) - (big - radius), 0.0)
            dy = max(radius - (y + 0.5), (y + 0.5) - (big - radius), 0.0)
            pixels[x, y] = 0 if math.hypot(dx, dy) > radius else 255
    return mask.resize((size, size), Image.LANCZOS)


def _circle_mask(size: int, supersample: int = 4) -> Image.Image:
    return _rounded_mask(size, 0.5, supersample)


# --------------------------------------------------------------------------- assembly

#: Legacy launcher bitmaps, pre-API-26. The whole bitmap is the icon, so it carries its own shape.
LEGACY_SIZES = {"mdpi": 48, "hdpi": 72, "xhdpi": 96, "xxhdpi": 144, "xxxhdpi": 192}

#: Adaptive layers are 108dp square; the guaranteed-visible area is the middle 72dp.
ADAPTIVE_SIZES = {"mdpi": 108, "hdpi": 162, "xhdpi": 216, "xxhdpi": 324, "xxxhdpi": 432}

#: The mark's design box as a fraction of the canvas. Smaller on the adaptive foreground because a
#: launcher may mask it to a circle inscribed in the middle two thirds.
LEGACY_BOX = 0.98
ADAPTIVE_BOX = 0.80

#: Everything is drawn at this multiple of the largest output and resampled down, so the geometry
#: is identical across densities rather than re-evaluated at each one.
MASTER = 512


def _flatten(background: Image.Image, mark: Image.Image) -> Image.Image:
    out = background.copy()
    out.alpha_composite(mark)
    return out


def build(res: Path) -> list[tuple[Path, int]]:
    written: list[tuple[Path, int]] = []

    def save(path: Path, image: Image.Image) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, "PNG", optimize=True)
        written.append((path, image.width))

    # --- legacy, both square and round -----------------------------------
    master_bg = _background(MASTER)
    master_mark = _mark(MASTER, LEGACY_BOX, None)
    master_flat = _flatten(master_bg, master_mark)
    square_master = master_flat.copy()
    square_master.putalpha(_rounded_mask(MASTER, 0.22))
    round_master = master_flat.copy()
    round_master.putalpha(_circle_mask(MASTER))

    for bucket, size in LEGACY_SIZES.items():
        save(
            res / f"mipmap-{bucket}/ic_launcher.png",
            square_master.resize((size, size), Image.LANCZOS),
        )
        save(
            res / f"mipmap-{bucket}/ic_launcher_round.png",
            round_master.resize((size, size), Image.LANCZOS),
        )

    # --- adaptive layers --------------------------------------------------
    adaptive_bg = _background(MASTER)
    adaptive_fg = _mark(MASTER, ADAPTIVE_BOX, None)
    # The themed icon is tinted by the launcher, so it is drawn in flat white and keeps only the
    # silhouette. The core lift would be invisible after tinting and is left out.
    adaptive_mono = _mark(MASTER, ADAPTIVE_BOX, (255, 255, 255))

    for bucket, size in ADAPTIVE_SIZES.items():
        save(
            res / f"mipmap-{bucket}/ic_launcher_background.png",
            adaptive_bg.resize((size, size), Image.LANCZOS),
        )
        save(
            res / f"mipmap-{bucket}/ic_launcher_foreground.png",
            adaptive_fg.resize((size, size), Image.LANCZOS),
        )
        save(
            res / f"mipmap-{bucket}/ic_launcher_monochrome.png",
            adaptive_mono.resize((size, size), Image.LANCZOS),
        )

    # --- the adaptive descriptors -----------------------------------------
    adaptive_xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<!--\n"
        "  Generated by app/tool/make_launcher_icon.py. Edit that, not this.\n"
        "\n"
        "  The mark is what the app makes: a Gaussian splat is a soft anisotropic ellipse, and\n"
        "  three of them along a short arc, fading backwards, is that ellipse at three instants.\n"
        "  Amber on near-black, from app/lib/design.dart.\n"
        "-->\n"
        '<adaptive-icon xmlns:android="http://schemas.android.com/apk/res/android">\n'
        '    <background android:drawable="@mipmap/ic_launcher_background" />\n'
        '    <foreground android:drawable="@mipmap/ic_launcher_foreground" />\n'
        '    <monochrome android:drawable="@mipmap/ic_launcher_monochrome" />\n'
        "</adaptive-icon>\n"
    )
    anydpi = res / "mipmap-anydpi-v26"
    anydpi.mkdir(parents=True, exist_ok=True)
    for name in ("ic_launcher.xml", "ic_launcher_round.xml"):
        (anydpi / name).write_text(adaptive_xml, encoding="utf-8")
        written.append((anydpi / name, 0))

    return written


def main() -> None:
    res = Path(__file__).resolve().parents[1] / "android/app/src/main/res"
    for path, size in build(res):
        label = f"{size}x{size}" if size else "xml"
        print(f"{path.relative_to(res.parents[4])}  {label}")


if __name__ == "__main__":
    main()
