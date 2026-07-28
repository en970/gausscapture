"""Reconstruction: turning posed images into a Gaussian splat.

GaussCapture does not implement a rasterizer. It drives external trainers
across a subprocess boundary, which is both the pragmatic choice for a small
project and the licence-safe one: the reference 3DGS implementation is
non-commercial, and several apparently permissive forks vendor its rasterizer.
The trainer we target is `gsplat <https://github.com/nerfstudio-project/gsplat>`_
(Apache-2.0). See ``docs/DEPENDENCIES.md``.
"""

from __future__ import annotations

from gausscapture.recon.base import TRAINING_PRESETS, Trainer
from gausscapture.recon.dataset import build_dataset
from gausscapture.recon.external import ExternalTrainer, trainer_status

__all__ = [
    "TRAINING_PRESETS",
    "ExternalTrainer",
    "Trainer",
    "build_dataset",
    "trainer_status",
]
