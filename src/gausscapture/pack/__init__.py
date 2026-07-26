"""CapturePack: a sensor-rich capture archive.

``.capturepack`` is a zip with a manifest, the source video, and whatever
sensor metadata the capture device could provide. It is deliberately *not*
proposed as a new standard -- prior art (Stray Scanner, Record3D, ARCore's
recording API) covers most of this ground, and a zip dialect with one
implementer is not a format. It is a convenience profile, and the direction of
travel is toward BagIt layout plus an unmodified nerfstudio ``transforms.json``
so that any pack trains in gsplat with zero conversion. See
``docs/RESEARCH.md`` section 8.
"""

from __future__ import annotations

from gausscapture.pack.archive import (
    export_archive,
    import_archive,
    write_checksums,
)
from gausscapture.pack.manifest import (
    create_minimal_manifest,
    find_main_video,
    read_manifest,
    validate,
    write_manifest,
)

__all__ = [
    "create_minimal_manifest",
    "export_archive",
    "find_main_video",
    "import_archive",
    "read_manifest",
    "validate",
    "write_checksums",
    "write_manifest",
]
