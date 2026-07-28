"""Batch evaluation: the harness the research question needs.

The project asks whether capture-time telemetry predicts downstream
reconstruction quality. Answering that is not a matter of opinion about which
signals sound important -- it requires running many captures through the same
pipeline with pinned settings and measuring what actually happened.

This package provides the two halves of that:

* :mod:`gausscapture.eval.harness` runs a directory of capture packs through
  the deterministic stages unattended and records one row per capture.
* :mod:`gausscapture.eval.stats` computes the statistics the pre-registered
  hypothesis is stated in, with no dependency beyond NumPy.

**Two endpoints, only one of which needs a GPU.** Final reconstruction quality
requires training. *Structure-from-motion registration outcome* does not -- it
falls out of COLMAP alone. The second is therefore measurable on a laptop, and
it is a real endpoint: a capture that fails to register produces nothing at
all, which is the failure users actually hit.
"""

from __future__ import annotations

from gausscapture.eval.analysis import StudyReport, analyse
from gausscapture.eval.harness import CaptureRecord, load_results, run_batch, run_one, write_csv
from gausscapture.eval.stats import (
    CorrelationResult,
    LogisticResult,
    auc,
    fit_logistic,
    spearman,
)

__all__ = [
    "CaptureRecord",
    "CorrelationResult",
    "LogisticResult",
    "StudyReport",
    "analyse",
    "auc",
    "fit_logistic",
    "load_results",
    "run_batch",
    "run_one",
    "spearman",
    "write_csv",
]
