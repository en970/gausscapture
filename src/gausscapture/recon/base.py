from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from gausscapture.progress import Progress

#: Iteration and resolution budgets. The numbers follow the published gsplat
#: baselines: 7k iterations reaches roughly 27.6 PSNR on Mip-NeRF 360 and 30k
#: reaches roughly 29.2, at 4.5 GB and 6.3 GB peak memory respectively.
TRAINING_PRESETS: dict[str, dict[str, Any]] = {
    "draft": {"iterations": 5_000, "resolution": 1024},
    "balanced": {"iterations": 15_000, "resolution": 1600},
    "high": {"iterations": 30_000, "resolution": 2048},
}


@runtime_checkable
class Trainer(Protocol):
    """Fits a Gaussian splat to a prepared dataset directory."""

    name: str

    def available(self) -> bool:
        """Whether this trainer is configured and present on this machine."""

    def fit(
        self,
        dataset_dir: Path,
        output_dir: Path,
        preset: str = "draft",
        progress: Progress | None = None,
    ) -> dict[str, Any]:
        """Train, returning a summary including the produced model paths."""
