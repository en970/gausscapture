"""Turning a batch of runs into the study's stated result.

The pre-registered hypothesis (``docs/RESEARCH.md`` section 9) has two parts.
Only the first is answerable without a GPU, and this module is careful to say
which is which rather than letting a reader assume both were tested:

1. **Registration outcome.** Does capture telemetry predict whether
   structure-from-motion succeeds at all? Target: AUC >= 0.80. Measurable with
   COLMAP alone, on a laptop.
2. **Reconstruction quality.** Does it predict final PSNR? Target:
   ``|rho| >= 0.40`` for ``vol_p10``. Requires training, so it requires a GPU
   and is not attempted here.

Every reported number carries its sample size, and small samples are labelled
as such. With a dozen captures a correlation is a hint, not a finding, and the
output says so instead of printing three decimal places and leaving the reader
to infer confidence that is not there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gausscapture.eval.stats import CorrelationResult, LogisticResult, fit_logistic, spearman

#: Pre-registered thresholds. Named here so a run either meets them or does
#: not, rather than being read post hoc.
TARGET_ABS_RHO = 0.40
TARGET_AUC = 0.80

#: Below this many captures, report but do not conclude.
MIN_SAMPLES_FOR_A_CLAIM = 30

#: Outcomes that only exist once pose estimation has run.
POSE_DERIVED_OUTCOMES = frozenset(
    {"registered", "registered_ratio", "images_registered", "sparse_points", "seconds_pose"}
)

#: Signals fed to the registration-failure model. Deliberately few and
#: mechanistically motivated -- blur, exposure, motion, redundancy -- rather
#: than everything available, because with tens of samples a wide model fits
#: noise.
DEFAULT_PREDICTORS = ("vol_p10", "blurry_frame_ratio", "too_fast_ratio", "duplicate_ratio")


@dataclass
class StudyReport:
    n: int
    registered: int
    failed: int
    correlations: list[CorrelationResult] = field(default_factory=list)
    logistic: LogisticResult | None = None
    outcome: str = "registered_ratio"
    caveats: list[str] = field(default_factory=list)
    meets_auc_target: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "registered": self.registered,
            "failed": self.failed,
            "outcome": self.outcome,
            "correlations": [c.to_dict() for c in self.correlations],
            "logistic": self.logistic.to_dict() if self.logistic else None,
            "meets_auc_target": self.meets_auc_target,
            "caveats": self.caveats,
        }

    def render(self) -> str:
        lines = [
            f"Captures: {self.n}   registered: {self.registered}   failed: {self.failed}",
            "",
            f"Rank correlation of each signal with {self.outcome}:",
        ]
        for correlation in self.correlations:
            marker = "  *" if abs(correlation.rho) >= TARGET_ABS_RHO else "   "
            lines.append(f"  {correlation}{marker}")

        if self.logistic:
            lines += ["", "Predicting registration failure:"]
            if np.isnan(self.logistic.auc):
                lines.append("  AUC undefined -- every capture had the same outcome.")
            else:
                verdict = "meets" if self.meets_auc_target else "below"
                lines.append(
                    f"  AUC {self.logistic.auc:.3f} ({verdict} the {TARGET_AUC} target), "
                    f"accuracy {self.logistic.accuracy:.2f}, n={self.logistic.n}"
                )
                for name, coefficient in zip(
                    self.logistic.features, self.logistic.coefficients, strict=True
                ):
                    lines.append(f"    {name:<24} {coefficient:+.5g}")
            for note in self.logistic.notes:
                lines.append(f"  note: {note}")

        if self.caveats:
            lines += ["", "Caveats:"]
            lines += [f"  - {caveat}" for caveat in self.caveats]
        return "\n".join(lines)


def analyse(
    rows: list[dict[str, Any]],
    outcome: str = "registered_ratio",
    predictors: tuple[str, ...] = DEFAULT_PREDICTORS,
    permutations: int = 10_000,
) -> StudyReport:
    """Compute correlations and the registration-failure model from run rows."""
    caveats: list[str] = []
    no_pose = {"skipped", "colmap_missing", "not_run"}

    # Only pose-derived outcomes need the pose stage to have run. Correlating
    # telemetry against, say, how many frames survived filtering is a valid
    # question that COLMAP has nothing to do with, and excluding those rows
    # would silently produce an empty analysis.
    if outcome in POSE_DERIVED_OUTCOMES:
        usable = [row for row in rows if row.get("pose_status") not in no_pose]
        if len(usable) < len(rows):
            caveats.append(
                f"{len(rows) - len(usable)} capture(s) had no pose stage and were excluded, "
                f"because '{outcome}' comes from pose estimation."
            )
    else:
        usable = [row for row in rows if np.isfinite(_get(row, outcome))]
        if len(usable) < len(rows):
            caveats.append(
                f"{len(rows) - len(usable)} capture(s) had no value for '{outcome}'."
            )

    n = len(usable)
    posed = [row for row in usable if row.get("pose_status") not in no_pose]
    registered = sum(1 for row in posed if row.get("registered"))
    report = StudyReport(
        n=n, registered=registered, failed=len(posed) - registered, outcome=outcome
    )

    if n < 3:
        report.caveats = [*caveats, f"Only {n} usable capture(s); no statistics computed."]
        return report

    outcome_values = np.array([_get(row, outcome) for row in usable], dtype=float)

    signal_names = sorted(
        {key[len("signal_") :] for row in usable for key in row if key.startswith("signal_")}
    )
    for name in signal_names:
        values = np.array([_get(row, f"signal_{name}") for row in usable], dtype=float)
        if np.all(np.isnan(values)) or np.nanstd(values) == 0:
            continue  # A constant signal cannot correlate with anything.
        report.correlations.append(
            spearman(
                values,
                outcome_values,
                permutations=permutations,
                signal_name=name,
                outcome_name=outcome,
            )
        )
    report.correlations.sort(key=lambda c: abs(c.rho), reverse=True)

    has_pose = [row for row in usable if row.get("pose_status") not in no_pose]
    labels = np.array([0 if row.get("registered") else 1 for row in has_pose], dtype=int)
    available = [p for p in predictors if any(f"signal_{p}" in row for row in has_pose)]
    if not has_pose:
        caveats.append("No capture reached pose estimation, so registration cannot be modelled.")
    elif available and len(np.unique(labels)) == 2:
        features = np.array(
            [[_get(row, f"signal_{p}") for p in available] for row in has_pose], dtype=float
        )
        report.logistic = fit_logistic(features, labels, feature_names=list(available))
        report.meets_auc_target = bool(
            not np.isnan(report.logistic.auc) and report.logistic.auc >= TARGET_AUC
        )
    elif len(np.unique(labels)) < 2:
        caveats.append(
            "Every capture had the same registration outcome, so failure cannot be modelled. "
            "Include captures designed to fail -- pure rotation from a fixed point is the "
            "reliable way to produce one."
        )

    if n < MIN_SAMPLES_FOR_A_CLAIM:
        caveats.append(
            f"n={n} is far below the {MIN_SAMPLES_FOR_A_CLAIM} captures this study "
            "pre-registered. These numbers indicate direction only; they do not test "
            "the hypothesis."
        )

    report.caveats = caveats
    return report


def _get(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None or isinstance(value, bool):
        return float(value) if isinstance(value, bool) else float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
