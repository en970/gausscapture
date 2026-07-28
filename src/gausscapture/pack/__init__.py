"""CapturePack: a sensor-rich capture archive.

A ``.capturepack`` is a `BagIt <https://datatracker.ietf.org/doc/html/rfc8493>`_
bag, distributed as a zip, whose payload holds a manifest, the source video,
and whatever sensor metadata the capture device could provide.

It is deliberately *not* proposed as a new standard. Prior art -- Stray
Scanner, Record3D, ARCore's recording API -- covers most of this ground, and a
zip dialect with one implementer is not a format. What GaussCapture contributes
is a *profile*: an existing container (BagIt) wrapping an existing interchange
format (``transforms.json``), so a pack is readable by tools that have never
heard of this project. See ``docs/RESEARCH.md`` section 8 for the argument, and
``docs/CAPTUREPACK_SPEC.md`` for the layout.
"""

from __future__ import annotations

from gausscapture.pack.archive import (
    export_archive,
    import_archive,
    verify_checksums,
    write_checksums,
)
from gausscapture.pack.bagit import is_bag, validate_bag, write_bag
from gausscapture.pack.manifest import (
    CAPTUREPACK_VERSION,
    create_minimal_manifest,
    find_main_video,
    read_manifest,
    validate,
    write_manifest,
)
from gausscapture.pack.transforms import (
    build_transforms,
    colmap_to_opengl,
    validate_transforms,
    write_transforms,
)

__all__ = [
    "CAPTUREPACK_VERSION",
    "build_transforms",
    "colmap_to_opengl",
    "create_minimal_manifest",
    "export_archive",
    "find_main_video",
    "import_archive",
    "is_bag",
    "read_manifest",
    "validate",
    "validate_bag",
    "validate_transforms",
    "verify_checksums",
    "write_bag",
    "write_checksums",
    "write_manifest",
    "write_transforms",
]
