"""Tests for the evaluation harness and its statistics.

The statistics are implemented here rather than imported from SciPy, so they
carry the burden of proof. These tests check them against cases whose answers
are known analytically -- perfect correlation, perfect separation, pure ties --
rather than against another implementation's output.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from gausscapture.eval.analysis import TARGET_AUC, analyse
from gausscapture.eval.harness import CaptureRecord, load_results, write_csv
from gausscapture.eval.stats import auc, fit_logistic, rankdata, spearman

# Permutation tests are the slow part; 400 is plenty to distinguish "clearly
# significant" from "clearly not" in a unit test.
FAST = 400


class TestRankdata:
    def test_simple_ordering(self):
        assert list(rankdata(np.array([10.0, 30.0, 20.0]))) == [1.0, 3.0, 2.0]

    def test_ties_get_the_average_rank(self):
        # Values 2 and 3 tie for ranks 2 and 3, so both become 2.5.
        assert list(rankdata(np.array([1.0, 5.0, 5.0, 9.0]))) == [1.0, 2.5, 2.5, 4.0]

    def test_all_equal(self):
        assert list(rankdata(np.array([7.0, 7.0, 7.0]))) == [2.0, 2.0, 2.0]


class TestSpearman:
    def test_perfect_positive(self):
        result = spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50], permutations=FAST)
        assert result.rho == pytest.approx(1.0)
        assert result.p_value < 0.05

    def test_perfect_negative(self):
        result = spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10], permutations=FAST)
        assert result.rho == pytest.approx(-1.0)

    def test_monotone_but_nonlinear_still_scores_one(self):
        """The reason to use rank correlation rather than Pearson."""
        x = [1, 2, 3, 4, 5, 6]
        y = [v**3 for v in x]
        assert spearman(x, y, permutations=FAST).rho == pytest.approx(1.0)

    def test_constant_signal_gives_zero(self):
        assert spearman([2, 2, 2, 2], [1, 5, 3, 9], permutations=FAST).rho == 0.0

    def test_unrelated_data_is_not_significant(self):
        rng = np.random.default_rng(7)
        result = spearman(rng.normal(size=40), rng.normal(size=40), permutations=2000)
        assert result.p_value > 0.05

    def test_p_value_is_never_zero(self):
        """A finite permutation count cannot legitimately produce p = 0."""
        result = spearman(list(range(20)), list(range(20)), permutations=FAST)
        assert result.p_value > 0
        assert result.p_value == pytest.approx(1 / (FAST + 1))

    def test_nan_pairs_are_dropped(self):
        result = spearman([1, 2, float("nan"), 4], [1, 2, 3, 4], permutations=FAST)
        assert result.n == 3

    def test_too_few_samples_returns_null_result(self):
        result = spearman([1, 2], [3, 4], permutations=FAST)
        assert result.rho == 0.0 and result.p_value == 1.0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            spearman([1, 2, 3], [1, 2])

    def test_is_reproducible(self):
        a = spearman([3, 1, 4, 1, 5, 9, 2, 6], [2, 7, 1, 8, 2, 8, 1, 8], permutations=FAST)
        b = spearman([3, 1, 4, 1, 5, 9, 2, 6], [2, 7, 1, 8, 2, 8, 1, 8], permutations=FAST)
        assert a.p_value == b.p_value


class TestAUC:
    def test_perfect_separation(self):
        assert auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0

    def test_perfectly_wrong(self):
        assert auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0

    def test_all_scores_equal_is_chance(self):
        assert auc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]) == 0.5

    def test_single_class_is_undefined(self):
        assert np.isnan(auc([0.1, 0.5, 0.9], [1, 1, 1]))


class TestLogistic:
    def test_recovers_a_known_direction(self):
        """Higher feature values should push the fitted probability upward."""
        rng = np.random.default_rng(3)
        x = np.concatenate([rng.normal(0, 1, 60), rng.normal(3, 1, 60)])
        y = np.array([0] * 60 + [1] * 60)
        result = fit_logistic(x, y, feature_names=["x"])
        assert result.coefficients[0] > 0
        assert result.auc > 0.9
        assert result.converged

    def test_reports_separability_instead_of_hiding_it(self):
        """Perfectly separable data yields AUC 1.0 that means nothing."""
        x = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
        y = np.array([0, 0, 0, 1, 1, 1])
        result = fit_logistic(x, y, feature_names=["x"])
        assert result.auc == pytest.approx(1.0)
        assert result.separable
        assert any("separable" in note for note in result.notes)

    def test_single_class_does_not_crash(self):
        result = fit_logistic(np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1]))
        assert not result.converged
        assert np.isnan(result.auc)
        assert result.notes

    def test_coefficients_are_in_the_features_own_units(self):
        """Standardisation is internal; a caller reading a coefficient should
        not have to know the training set's mean and variance."""
        rng = np.random.default_rng(11)
        small = rng.normal(0, 1, 80)
        large = small * 1000.0  # same information, different units
        labels = (small > 0).astype(int)

        a = fit_logistic(small, labels, feature_names=["s"])
        b = fit_logistic(large, labels, feature_names=["s"])
        assert b.coefficients[0] == pytest.approx(a.coefficients[0] / 1000.0, rel=0.05)

    def test_multiple_features(self):
        rng = np.random.default_rng(5)
        signal = rng.normal(size=100)
        noise = rng.normal(size=100)
        labels = (signal > 0).astype(int)
        result = fit_logistic(
            np.column_stack([signal, noise]), labels, feature_names=["signal", "noise"]
        )
        assert abs(result.coefficients[0]) > abs(result.coefficients[1])


def _row(name: str, vol_p10: float, blurry: float, fast: float, registered: bool) -> dict:
    return {
        "name": name,
        "pose_status": "good" if registered else "bad",
        "registered": registered,
        "registered_ratio": 0.95 if registered else 0.05,
        "signal_vol_p10": vol_p10,
        "signal_blurry_frame_ratio": blurry,
        "signal_too_fast_ratio": fast,
        "signal_duplicate_ratio": 0.05,
    }


class TestAnalysis:
    def test_finds_a_planted_relationship(self):
        rows = [
            _row(f"good{i}", 900 - i * 10, 0.02, 0.02, True) for i in range(12)
        ] + [_row(f"bad{i}", 100 + i * 10, 0.55, 0.60, False) for i in range(12)]
        report = analyse(rows, permutations=FAST)

        assert report.n == 24
        assert report.registered == 12 and report.failed == 12
        assert report.logistic is not None
        assert report.logistic.auc >= TARGET_AUC
        by_name = {c.signal: c for c in report.correlations}
        assert abs(by_name["blurry_frame_ratio"].rho) > 0.8

    def test_warns_when_the_sample_is_too_small_to_conclude(self):
        rows = [_row("a", 900, 0.01, 0.01, True), _row("b", 100, 0.6, 0.6, False)] * 3
        report = analyse(rows, permutations=FAST)
        assert any("pre-registered" in caveat for caveat in report.caveats)

    def test_single_outcome_class_is_explained_not_crashed(self):
        rows = [_row(f"g{i}", 900 - i, 0.02, 0.02, True) for i in range(6)]
        report = analyse(rows, permutations=FAST)
        assert report.logistic is None
        assert any("designed to fail" in caveat for caveat in report.caveats)

    def test_skipped_captures_are_excluded_and_reported(self):
        rows = [_row(f"g{i}", 900 - i, 0.02, 0.02, True) for i in range(4)]
        rows.append({**_row("skipped", 500, 0.1, 0.1, False), "pose_status": "colmap_missing"})
        report = analyse(rows, permutations=FAST)
        assert report.n == 4
        assert any("no pose stage" in caveat for caveat in report.caveats)

    def test_renders_without_raising(self):
        rows = [_row(f"g{i}", 900 - i * 9, 0.02, 0.02, True) for i in range(8)] + [
            _row(f"b{i}", 120 + i * 9, 0.5, 0.5, False) for i in range(8)
        ]
        text = analyse(rows, permutations=FAST).render()
        assert "Rank correlation" in text
        assert "registration failure" in text


class TestHarnessIO:
    def test_record_flattens_signals(self):
        record = CaptureRecord(name="x", signals={"vol_p10": 3.0})
        data = record.to_dict()
        assert data["signal_vol_p10"] == 3.0
        assert "signals" not in data

    def test_csv_round_trip(self, tmp_path):
        records = [
            CaptureRecord(name="a", signals={"vol_p10": 1.0}, registered=True),
            CaptureRecord(name="b", signals={"vol_p10": 2.0}, registered=False),
        ]
        path = write_csv(records, tmp_path / "results.csv")
        text = path.read_text(encoding="utf-8")
        assert "signal_vol_p10" in text.splitlines()[0]
        assert len(text.strip().splitlines()) == 3

    def test_jsonl_round_trip(self, tmp_path):
        path = tmp_path / "results.jsonl"
        rows = [_row("a", 900, 0.01, 0.01, True), _row("b", 100, 0.6, 0.6, False)]
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        assert len(load_results(path)) == 2

    def test_missing_results_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_results(tmp_path / "nope.jsonl")
