"""Tests for the Gaussian splat PLY reader.

Written before any real trainer output existed, against a synthetic file built
to the format's own spec. The encodings are the risk: opacity is stored as a
logit and scale as a logarithm, and reading either literally produces numbers
that are wrong without being obviously wrong -- opacities near 0.5 that should
be near 1, sizes that look plausible until you compare them. So the tests check
the decoded values against hand-computed ones rather than merely checking that
parsing succeeds.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from gausscapture.errors import CaptureFormatError
from gausscapture.export.splat_ply import SH_C0, read_splat_ply


def write_splat_ply(path, gaussians, sh_rest: int = 45, ascii_format: bool = False):
    """Write a splat PLY in the layout every 3DGS trainer produces.

    ``gaussians`` is a list of ``(xyz, dc, opacity_logit, log_scales, quat)``.
    """
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{i}" for i in range(3)]
    names += [f"f_rest_{i}" for i in range(sh_rest)]
    names += ["opacity", "scale_0", "scale_1", "scale_2"]
    names += [f"rot_{i}" for i in range(4)]

    header = ["ply"]
    header.append("format ascii 1.0" if ascii_format else "format binary_little_endian 1.0")
    header.append(f"element vertex {len(gaussians)}")
    header += [f"property float {n}" for n in names]
    header.append("end_header")

    with open(path, "wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        if ascii_format:
            return path
        for xyz, dc, opacity, log_scales, quat in gaussians:
            values = [*xyz, 0.0, 0.0, 0.0, *dc, *([0.0] * sh_rest),
                      opacity, *log_scales, *quat]
            handle.write(struct.pack(f"<{len(values)}f", *values))
    return path


def one(xyz=(0.0, 0.0, 0.0), dc=(0.0, 0.0, 0.0), opacity=0.0,
        log_scales=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)):
    return (xyz, dc, opacity, log_scales, quat)


class TestReading:
    def test_reads_positions_and_count(self, tmp_path):
        path = write_splat_ply(tmp_path / "s.ply", [
            one(xyz=(1.0, 2.0, 3.0)),
            one(xyz=(-4.0, 5.0, -6.0)),
        ])
        splat = read_splat_ply(path)
        assert len(splat) == 2
        assert splat.positions[0].tolist() == [1.0, 2.0, 3.0]
        assert splat.positions[1].tolist() == [-4.0, 5.0, -6.0]

    def test_opacity_is_a_logit_not_a_probability(self, tmp_path):
        """The trap: 0.0 stored means 0.5 opacity, not zero."""
        path = write_splat_ply(tmp_path / "s.ply", [
            one(opacity=0.0),      # sigmoid(0)  = 0.5
            one(opacity=4.0),      # sigmoid(4)  ~ 0.982
            one(opacity=-4.0),     # sigmoid(-4) ~ 0.018
        ])
        splat = read_splat_ply(path)
        assert splat.opacities[0] == pytest.approx(0.5, abs=1e-4)
        assert splat.opacities[1] == pytest.approx(0.9820, abs=1e-3)
        assert splat.opacities[2] == pytest.approx(0.0180, abs=1e-3)
        assert ((splat.opacities >= 0) & (splat.opacities <= 1)).all()

    def test_scale_is_a_logarithm(self, tmp_path):
        """Stored 0.0 means a scale of 1.0, and negatives are valid."""
        path = write_splat_ply(tmp_path / "s.ply", [
            one(log_scales=(0.0, 0.0, 0.0)),
            one(log_scales=(-2.0, -2.0, -2.0)),
        ])
        splat = read_splat_ply(path)
        assert splat.scales[0].tolist() == pytest.approx([1.0, 1.0, 1.0])
        assert splat.scales[1][0] == pytest.approx(np.exp(-2.0), abs=1e-5)
        assert (splat.scales > 0).all()

    def test_colour_comes_from_the_sh_dc_term(self, tmp_path):
        """Base colour is 0.5 + C0 * f_dc, not f_dc itself."""
        path = write_splat_ply(tmp_path / "s.ply", [
            one(dc=(0.0, 0.0, 0.0)),                 # mid grey
            one(dc=(1 / (2 * SH_C0),) * 3),          # saturates to white
        ])
        splat = read_splat_ply(path)
        assert splat.colors[0].tolist() == pytest.approx([127, 127, 127], abs=2)
        assert splat.colors[1].tolist() == [255, 255, 255]

    def test_colour_is_clamped_not_wrapped(self, tmp_path):
        """An out-of-range DC term must clip, not overflow uint8."""
        path = write_splat_ply(tmp_path / "s.ply", [one(dc=(100.0, -100.0, 0.0))])
        splat = read_splat_ply(path)
        assert splat.colors[0][0] == 255
        assert splat.colors[0][1] == 0

    def test_counts_higher_order_sh(self, tmp_path):
        path = write_splat_ply(tmp_path / "s.ply", [one()], sh_rest=45)
        assert read_splat_ply(path).sh_bands == 45

    def test_handles_a_splat_without_higher_order_sh(self, tmp_path):
        path = write_splat_ply(tmp_path / "s.ply", [one()], sh_rest=0)
        splat = read_splat_ply(path)
        assert splat.sh_bands == 0
        assert len(splat) == 1


class TestFiltering:
    def test_drops_near_transparent_gaussians(self, tmp_path):
        """Training leaves a tail the renderer never shows."""
        path = write_splat_ply(tmp_path / "s.ply", [
            one(opacity=6.0),      # ~0.998
            one(opacity=-8.0),     # ~0.0003
            one(opacity=2.0),      # ~0.88
        ])
        visible = read_splat_ply(path).visible(min_opacity=0.02)
        assert len(visible) == 2

    def test_filtering_keeps_arrays_aligned(self, tmp_path):
        path = write_splat_ply(tmp_path / "s.ply", [
            one(xyz=(1.0, 0, 0), opacity=6.0),
            one(xyz=(2.0, 0, 0), opacity=-8.0),
        ])
        visible = read_splat_ply(path).visible()
        assert len(visible.positions) == len(visible.colors) == len(visible.opacities)
        assert visible.positions[0][0] == 1.0


class TestRejections:
    def test_rejects_a_non_ply(self, tmp_path):
        path = tmp_path / "not.ply"
        path.write_bytes(b"this is not a ply at all")
        with pytest.raises(CaptureFormatError, match="Not a PLY"):
            read_splat_ply(path)

    def test_rejects_ascii(self, tmp_path):
        """Trainers write binary; ASCII means something else produced it."""
        path = write_splat_ply(tmp_path / "a.ply", [one()], ascii_format=True)
        with pytest.raises(CaptureFormatError, match="ASCII"):
            read_splat_ply(path)

    def test_rejects_a_plain_point_cloud(self, tmp_path):
        """A COLMAP export has x/y/z but no opacity, and is not a splat.

        Worth a clear message: handing the sparse cloud to a splat importer is
        an easy mistake, and the two files look alike from the outside.
        """
        path = tmp_path / "points.ply"
        header = (
            "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
            "property float x\nproperty float y\nproperty float z\nend_header\n"
        )
        with open(path, "wb") as handle:
            handle.write(header.encode())
            handle.write(struct.pack("<3f", 1.0, 2.0, 3.0))
        with pytest.raises(CaptureFormatError, match="Gaussian splat"):
            read_splat_ply(path)

    def test_summary_reports_what_was_read(self, tmp_path):
        path = write_splat_ply(tmp_path / "s.ply", [
            one(opacity=4.0, log_scales=(-1.0,) * 3),
            one(opacity=4.0, log_scales=(-1.0,) * 3),
        ])
        summary = read_splat_ply(path).summary()
        assert summary["gaussians"] == 2
        assert summary["sh_coefficients"] == 45
        assert summary["median_scale"] == pytest.approx(np.exp(-1.0), abs=1e-5)
        assert summary["mean_opacity"] == pytest.approx(0.982, abs=1e-3)
