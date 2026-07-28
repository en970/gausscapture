"""Statistics for the capture-quality study.

Implemented on NumPy alone rather than pulling in SciPy or scikit-learn. Two
reasons, in order of importance:

1. The methods needed here are small and their assumptions matter to the
   result, so they should be legible in the repository rather than hidden
   behind an import. A reviewer asking "how did you get that p-value" should be
   able to read the answer.
2. It keeps the dependency surface -- and therefore the licence surface --
   minimal, which is a stated project constraint.

p-values come from **permutation tests** rather than asymptotic
approximations. With the sample sizes a solo researcher can actually collect
(tens of captures, not thousands), a permutation test is both more defensible
and easier to justify than a t-distribution approximation whose assumptions go
unchecked.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class CorrelationResult:
    """Rank correlation between one signal and one outcome."""

    signal: str
    outcome: str
    rho: float
    p_value: float
    n: int
    permutations: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"{self.signal:<24} rho={self.rho:+.3f}  p={self.p_value:.4f}  n={self.n}"
        )


@dataclass
class LogisticResult:
    """Logistic regression predicting a binary outcome from signals."""

    features: list[str]
    coefficients: list[float]
    intercept: float
    auc: float
    accuracy: float
    n: int
    positives: int
    converged: bool
    separable: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rankdata(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, matching the usual Spearman convention."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)

    # Average the ranks within each group of equal values. Without this, ties
    # get arbitrary distinct ranks and the correlation depends on input order.
    sorted_values = values[order]
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or sorted_values[i] != sorted_values[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.sqrt((x * x).sum() * (y * y).sum())
    if denominator == 0:
        return 0.0  # A constant signal carries no information about anything.
    return float((x * y).sum() / denominator)


def spearman(
    signal: np.ndarray | list[float],
    outcome: np.ndarray | list[float],
    permutations: int = 10_000,
    seed: int = 0,
    signal_name: str = "signal",
    outcome_name: str = "outcome",
) -> CorrelationResult:
    """Spearman rank correlation with a two-sided permutation p-value.

    The permutation null is "this signal is unrelated to this outcome": shuffle
    the outcomes, recompute, and count how often chance produces a correlation
    at least as extreme. ``seed`` is fixed so a reported p-value is
    reproducible.
    """
    x = np.asarray(signal, dtype=float)
    y = np.asarray(outcome, dtype=float)
    if x.shape != y.shape:
        raise ValueError("signal and outcome must have the same length")

    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    n = len(x)
    if n < 3:
        return CorrelationResult(signal_name, outcome_name, 0.0, 1.0, n, 0)

    rank_x, rank_y = rankdata(x), rankdata(y)
    rho = _pearson(rank_x, rank_y)

    rng = np.random.default_rng(seed)
    shuffled = rank_y.copy()
    at_least_as_extreme = 0
    for _ in range(permutations):
        rng.shuffle(shuffled)
        if abs(_pearson(rank_x, shuffled)) >= abs(rho) - 1e-12:
            at_least_as_extreme += 1

    # The +1 in numerator and denominator is the standard correction that keeps
    # the p-value from ever being exactly zero, which it cannot legitimately be
    # with a finite number of permutations.
    p_value = (at_least_as_extreme + 1) / (permutations + 1)
    return CorrelationResult(signal_name, outcome_name, rho, p_value, n, permutations)


def auc(scores: np.ndarray | list[float], labels: np.ndarray | list[int]) -> float:
    """Area under the ROC curve, via the Mann-Whitney rank identity.

    Equals the probability that a randomly chosen positive scores above a
    randomly chosen negative. 0.5 is chance; below 0.5 means the score is
    predictive in the opposite direction.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    positives = labels == 1
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")  # Undefined with only one class present.
    ranks = rankdata(scores)
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def standardise(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre and scale columns, leaving constant columns untouched."""
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale == 0] = 1.0
    return (matrix - mean) / scale, mean, scale


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray | list[int],
    feature_names: list[str] | None = None,
    l2: float = 1.0,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
) -> LogisticResult:
    """Fit a logistic model by iteratively reweighted least squares.

    An L2 penalty is applied by default and is not optional in practice: with a
    few dozen captures and several correlated signals, the data are frequently
    separable, and an unpenalised fit responds by sending coefficients to
    infinity. The penalty keeps the fit finite and the reported AUC meaningful.
    Separation is detected and reported rather than hidden.
    """
    X = np.asarray(features, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(labels, dtype=float)
    names = feature_names or [f"x{i}" for i in range(X.shape[1])]

    finite = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[finite], y[finite]
    n, k = X.shape
    notes: list[str] = []

    if n == 0 or len(np.unique(y)) < 2:
        return LogisticResult(
            features=names,
            coefficients=[0.0] * k,
            intercept=0.0,
            auc=float("nan"),
            accuracy=float("nan"),
            n=n,
            positives=int(y.sum()) if n else 0,
            converged=False,
            notes=["Need both outcome classes present to fit a model"],
        )

    Xs, mean, scale = standardise(X)
    design = np.hstack([np.ones((n, 1)), Xs])
    beta = np.zeros(k + 1)

    converged = False
    for _ in range(max_iterations):
        eta = design @ beta
        # Clipping keeps exp() from overflowing on separable data, where eta
        # grows without bound.
        probability = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        weights = np.clip(probability * (1 - probability), 1e-9, None)
        gradient = design.T @ (y - probability) - l2 * np.r_[0.0, beta[1:]]
        hessian = design.T @ (design * weights[:, None]) + l2 * np.diag(np.r_[0.0, np.ones(k)])
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            notes.append("Hessian was singular; stopped early")
            break
        beta = beta + step
        if np.max(np.abs(step)) < tolerance:
            converged = True
            break

    eta = design @ beta
    probability = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
    predicted = (probability >= 0.5).astype(float)

    # Detect separability from the *ranking*, not from how extreme the fitted
    # probabilities are: the L2 penalty deliberately keeps them away from 0 and
    # 1, so extremeness would never trigger. An AUC of exactly 1.0 means some
    # threshold classifies every training point correctly, which is what
    # "separable" means and what makes the number uninformative.
    fitted_auc = auc(probability, y.astype(int))
    separable = bool(not np.isnan(fitted_auc) and fitted_auc >= 1.0 - 1e-12)
    if separable:
        notes.append(
            "The classes are perfectly separable at this sample size, so the AUC of 1.0 "
            "reflects the fit rather than out-of-sample skill. Collect more captures, "
            "or hold some out."
        )

    # Undo standardisation so coefficients are in the signals' own units.
    raw_coefficients = beta[1:] / scale
    raw_intercept = float(beta[0] - float(np.sum(beta[1:] * mean / scale)))

    return LogisticResult(
        features=names,
        coefficients=[float(c) for c in raw_coefficients],
        intercept=raw_intercept,
        auc=fitted_auc,
        accuracy=float((predicted == y).mean()),
        n=n,
        positives=int(y.sum()),
        converged=converged,
        separable=separable,
        notes=notes,
    )
