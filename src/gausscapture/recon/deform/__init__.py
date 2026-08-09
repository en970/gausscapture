"""The deformable 4D trainer.

Importing this package must not require ``torch``. The CLI's ``doctor`` command
enumerates optional dependencies, ``cli.py`` imports the whole package tree to
discover subcommands, and the 211 existing tests run on a machine with neither
torch nor gsplat installed -- so a top-level ``import torch`` here would turn an
optional extra into a hard one for everybody.

Every name below is therefore resolved lazily on first attribute access, and
:func:`available` answers the question the CLI actually asks without importing
anything heavy.
"""

from __future__ import annotations

import importlib
from typing import Any

from gausscapture.errors import DependencyMissingError

__all__ = [
    "CanonicalParams",
    "ExplicitTrajectoryField",
    "FixedCameraClip",
    "KPlanesField",
    "ReferenceMcmcStrategy",
    "ReferenceRasterizer",
    "ScaffoldField",
    "ScheduleConfig",
    "SceneInit",
    "Skinning",
    "Train4DConfig",
    "TrainingSchedule",
    "available",
    "build_from_init",
    "build_skinning",
    "deform_gaussians",
    "export_scene4d",
    "load_checkpoint",
    "requires_torch",
    "save_checkpoint",
    "seed_nodes",
    "train",
]

_EXPORTS: dict[str, str] = {
    "CanonicalParams": "gausscapture.recon.deform.params",
    "ExplicitTrajectoryField": "gausscapture.recon.deform.field",
    "FixedCameraClip": "gausscapture.recon.deform.train4d",
    "KPlanesField": "gausscapture.recon.deform.field",
    "ReferenceMcmcStrategy": "gausscapture.recon.deform.schedule",
    "ReferenceRasterizer": "gausscapture.recon.deform.raster",
    "ScaffoldField": "gausscapture.recon.deform.field",
    "ScheduleConfig": "gausscapture.recon.deform.schedule",
    "SceneInit": "gausscapture.recon.deform.init4d",
    "Skinning": "gausscapture.recon.deform.field",
    "Train4DConfig": "gausscapture.recon.deform.train4d",
    "TrainingSchedule": "gausscapture.recon.deform.schedule",
    "build_from_init": "gausscapture.recon.deform.bundle",
    "build_skinning": "gausscapture.recon.deform.field",
    "deform_gaussians": "gausscapture.recon.deform.field",
    "export_scene4d": "gausscapture.recon.deform.train4d",
    "load_checkpoint": "gausscapture.recon.deform.train4d",
    "save_checkpoint": "gausscapture.recon.deform.train4d",
    "seed_nodes": "gausscapture.recon.deform.field",
    "train": "gausscapture.recon.deform.train4d",
}


def available() -> bool:
    """Whether the ``train4d`` extra is importable on this machine."""
    return importlib.util.find_spec("torch") is not None


def requires_torch() -> None:
    """Raise the package's own error, with the install line, if torch is absent."""
    if not available():
        raise DependencyMissingError(
            "PyTorch is required for the 4D deformation trainer. "
            "Install the training extra: pip install 'gausscapture[train4d]'"
        )


#: Names that carry no tensors and therefore no torch. ``SceneInit`` is the
#: schema of the file *preparation* writes, on a laptop that has no reason to
#: install a tensor library -- demanding one to name the class would put the
#: extra back where the split was made to take it out of.
_TORCH_FREE = frozenset({"SceneInit"})


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name not in _TORCH_FREE:
        requires_torch()
    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(__all__)
