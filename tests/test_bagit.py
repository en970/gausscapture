"""Tests for the BagIt profile and the archive round trip."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from gausscapture.errors import CaptureFormatError
from gausscapture.pack import archive, bagit, manifest


@pytest.fixture()
def payload(tmp_path):
    root = tmp_path / "payload"
    (root / "video").mkdir(parents=True)
    (root / "camera").mkdir(parents=True)
    (root / "manifest.json").write_text('{"capturepack_version": "0.1"}', encoding="utf-8")
    (root / "video" / "main_video.mp4").write_bytes(b"not really a video, but bytes are bytes")
    (root / "camera" / "intrinsics.json").write_text("{}", encoding="utf-8")
    return root


class TestBagStructure:
    def test_writes_the_required_tag_files(self, payload, tmp_path):
        bag = bagit.write_bag(payload, tmp_path / "bag")
        for name in ("bagit.txt", "bag-info.txt", "manifest-sha256.txt", "tagmanifest-sha256.txt"):
            assert (bag / name).exists(), name
        assert (bag / "data" / "manifest.json").exists()

    def test_declares_the_bagit_version(self, payload, tmp_path):
        bag = bagit.write_bag(payload, tmp_path / "bag")
        text = (bag / "bagit.txt").read_text(encoding="utf-8")
        assert "BagIt-Version: 1.0" in text
        assert "Tag-File-Character-Encoding: UTF-8" in text

    def test_payload_oxum_counts_bytes_and_files(self, payload, tmp_path):
        bag = bagit.write_bag(payload, tmp_path / "bag")
        info = dict(
            line.split(": ", 1)
            for line in (bag / "bag-info.txt").read_text(encoding="utf-8").splitlines()
            if ": " in line
        )
        octets, count = info["Payload-Oxum"].split(".")
        assert int(count) == 3
        assert int(octets) == sum(
            p.stat().st_size for p in (bag / "data").rglob("*") if p.is_file()
        )

    def test_carries_supplied_metadata(self, payload, tmp_path):
        bag = bagit.write_bag(payload, tmp_path / "bag", info={"External-Identifier": "abc-123"})
        assert "External-Identifier: abc-123" in (bag / "bag-info.txt").read_text(encoding="utf-8")

    def test_manifest_lists_every_payload_file(self, payload, tmp_path):
        bag = bagit.write_bag(payload, tmp_path / "bag")
        listed = {
            line.split("  ", 1)[1]
            for line in (bag / "manifest-sha256.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        assert listed == {
            "data/manifest.json",
            "data/video/main_video.mp4",
            "data/camera/intrinsics.json",
        }

    def test_refuses_an_empty_payload(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(CaptureFormatError, match="Nothing to bag"):
            bagit.write_bag(empty, tmp_path / "bag")


class TestBagValidation:
    def test_a_fresh_bag_validates(self, payload, tmp_path):
        bag = bagit.write_bag(payload, tmp_path / "bag")
        result = bagit.validate_bag(bag)
        assert result["valid"], result["errors"]
        assert result["checked"] == 3

    def test_detects_a_modified_payload_file(self, payload, tmp_path):
        bag = bagit.write_bag(payload, tmp_path / "bag")
        (bag / "data" / "manifest.json").write_text("{}", encoding="utf-8")
        result = bagit.validate_bag(bag)
        assert not result["valid"]
        assert any("Checksum mismatch" in e for e in result["errors"])

    def test_detects_a_deleted_payload_file(self, payload, tmp_path):
        bag = bagit.write_bag(payload, tmp_path / "bag")
        (bag / "data" / "camera" / "intrinsics.json").unlink()
        assert not bagit.validate_bag(bag)["valid"]

    def test_detects_an_unlisted_extra_file(self, payload, tmp_path):
        """Completeness runs both ways: an unmanifested file fails the bag."""
        bag = bagit.write_bag(payload, tmp_path / "bag")
        (bag / "data" / "smuggled.txt").write_text("surprise", encoding="utf-8")
        result = bagit.validate_bag(bag)
        assert not result["valid"]
        assert any("not listed in the manifest" in e for e in result["errors"])

    def test_complete_only_skips_hashing(self, payload, tmp_path):
        bag = bagit.write_bag(payload, tmp_path / "bag")
        (bag / "data" / "manifest.json").write_text("{}", encoding="utf-8")
        result = bagit.validate_bag(bag, complete_only=True)
        assert result["valid"]  # present but unverified
        assert result["checked"] == 0

    def test_rejects_a_non_bag(self, tmp_path):
        assert not bagit.validate_bag(tmp_path)["valid"]

    def test_is_bag_predicate(self, payload, tmp_path):
        assert not bagit.is_bag(payload)
        assert bagit.is_bag(bagit.write_bag(payload, tmp_path / "bag"))


class TestArchiveRoundTrip:
    def test_exported_archive_is_a_bag(self, project_with_video):
        out = archive.export_archive(project_with_video.path)
        with zipfile.ZipFile(out) as zf:
            names = set(zf.namelist())
        assert "bagit.txt" in names
        assert "manifest-sha256.txt" in names
        assert "data/manifest.json" in names

    def test_reimport_restores_a_valid_pack(self, project_with_video, store):
        out = archive.export_archive(project_with_video.path)
        other = store.create("reimported")
        result = archive.import_archive(other.path, out)
        assert result["valid"], result["errors"]
        assert result["bag"]["valid"]
        assert manifest.find_main_video(other.capturepack_dir).exists()

    def test_corrupted_transfer_is_rejected(self, project_with_video, store, tmp_path):
        """A bag whose checksums fail is a broken transfer, not a partial capture."""
        out = archive.export_archive(project_with_video.path)

        tampered = tmp_path / "tampered.capturepack"
        with zipfile.ZipFile(out) as src, zipfile.ZipFile(tampered, "w") as dst:
            for item in src.infolist():
                data = src.read(item.filename)
                if item.filename == "data/manifest.json":
                    data = b'{"capturepack_version": "0.1"}'
                dst.writestr(item, data)

        other = store.create("corrupted")
        result = archive.import_archive(other.path, tampered)
        assert not result["valid"]

    def test_legacy_flat_archive_still_imports(self, project_with_video, store, tmp_path):
        """Packs written before the profile existed must keep working."""
        flat = tmp_path / "legacy.capturepack"
        pack = project_with_video.capturepack_dir
        with zipfile.ZipFile(flat, "w") as zf:
            for file in sorted(pack.rglob("*")):
                if file.is_file():
                    zf.write(file, file.relative_to(pack))

        other = store.create("legacy")
        assert archive.import_archive(other.path, flat)["valid"]

    def test_archive_zipped_with_its_parent_folder_still_imports(
        self, project_with_video, store, tmp_path
    ):
        """The shape produced by right-click-Compress on a directory."""
        nested = tmp_path / "nested.capturepack"
        pack = project_with_video.capturepack_dir
        with zipfile.ZipFile(nested, "w") as zf:
            for file in sorted(pack.rglob("*")):
                if file.is_file():
                    zf.write(file, Path_join("session", file.relative_to(pack)))

        other = store.create("nested")
        assert archive.import_archive(other.path, nested)["valid"]

    def test_rejects_a_non_zip(self, project_with_video, tmp_path):
        junk = tmp_path / "not.capturepack"
        junk.write_text("definitely not a zip", encoding="utf-8")
        with pytest.raises(CaptureFormatError):
            archive.import_archive(project_with_video.path, junk)

    def test_including_a_dataset_requires_frames(self, project_with_video):
        with pytest.raises(CaptureFormatError, match="no extracted frames"):
            archive.export_archive(project_with_video.path, include_dataset=True)


class TestThirdPartyInterop:
    """The point of adopting BagIt is that other people's tools can read it.

    These assert that against `bagit-python`, the reference implementation,
    which has no knowledge of GaussCapture whatsoever.
    """

    @pytest.fixture()
    def extracted(self, project_with_video, tmp_path):
        out = archive.export_archive(project_with_video.path)
        destination = tmp_path / "extracted"
        with zipfile.ZipFile(out) as zf:
            zf.extractall(destination)
        return destination

    def test_reference_implementation_accepts_our_bag(self, extracted):
        reference = pytest.importorskip("bagit")
        bag = reference.Bag(str(extracted))
        bag.validate()  # raises BagValidationError if not
        assert bag.version_info == (1, 0)

    def test_reference_implementation_sees_the_payload(self, extracted):
        reference = pytest.importorskip("bagit")
        bag = reference.Bag(str(extracted))
        payload = {Path(p).name for p in bag.payload_files()}
        assert "manifest.json" in payload

    def test_reference_implementation_catches_tampering(self, extracted):
        reference = pytest.importorskip("bagit")
        (extracted / "data" / "manifest.json").write_text("{}", encoding="utf-8")
        with pytest.raises(reference.BagValidationError):
            reference.Bag(str(extracted)).validate()


def Path_join(prefix: str, relative) -> str:
    return f"{prefix}/{relative.as_posix()}"
