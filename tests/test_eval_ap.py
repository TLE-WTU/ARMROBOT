"""
Unit tests for benchmark/eval_ap.py – GraspNet-style AP metric computation.

Test strategy:
- All-positive candidates  → AP should be 100%
- All-negative candidates  → AP should be 0%
- Mixed candidates ordered by score → PR curve interpolation verified
- Per-split (seen / novel) breakdown
- compute_ap_score() bridge from metrics.py
"""

import math

import numpy as np
import pytest

from benchmark.eval_ap import (
    APResult,
    GraspCandidate,
    _area_under_pr,
    _label_candidates,
    _precision_recall_curve,
    compute_ap,
    compute_ap_per_object,
)
from benchmark.metrics import compute_ap_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _perfect_antipodal(score: float, obj: str = "duck", split: str = "seen") -> GraspCandidate:
    """Contact geometry that always passes force-closure at any realistic mu."""
    p1 = np.array([0.0, -0.02, 0.0])
    p2 = np.array([0.0,  0.02, 0.0])
    n1 = np.array([0.0,  1.0, 0.0])
    n2 = np.array([0.0, -1.0, 0.0])
    return GraspCandidate(p1=p1, p2=p2, n1=n1, n2=n2, score=score, object_id=obj, split=split)


def _bad_grasp(score: float, obj: str = "duck", split: str = "seen") -> GraspCandidate:
    """Parallel normals → force-closure never passes."""
    p1 = np.array([0.0, -0.02, 0.0])
    p2 = np.array([0.0,  0.02, 0.0])
    n1 = np.array([0.0, 0.0, 1.0])
    n2 = np.array([0.0, 0.0, 1.0])
    return GraspCandidate(p1=p1, p2=p2, n1=n1, n2=n2, score=score, object_id=obj, split=split)


# ---------------------------------------------------------------------------
# Tests – _label_candidates
# ---------------------------------------------------------------------------

class TestLabelCandidates:
    def test_all_positive(self):
        cands = [_perfect_antipodal(0.9), _perfect_antipodal(0.8), _perfect_antipodal(0.7)]
        _, labels = _label_candidates(cands, friction_coeff=0.4)
        assert labels.sum() == 3

    def test_all_negative(self):
        cands = [_bad_grasp(0.9), _bad_grasp(0.5)]
        _, labels = _label_candidates(cands, friction_coeff=0.8)
        assert labels.sum() == 0

    def test_scores_sorted_descending(self):
        cands = [_perfect_antipodal(0.3), _perfect_antipodal(0.9), _perfect_antipodal(0.6)]
        scores, _ = _label_candidates(cands, friction_coeff=0.4)
        assert list(scores) == sorted(scores.tolist(), reverse=True)

    def test_empty(self):
        scores, labels = _label_candidates([], friction_coeff=0.8)
        assert len(scores) == 0 and len(labels) == 0


# ---------------------------------------------------------------------------
# Tests – _precision_recall_curve
# ---------------------------------------------------------------------------

class TestPrecisionRecallCurve:
    def test_all_positive_gives_pr_one(self):
        labels = np.array([1, 1, 1])
        prec, rec = _precision_recall_curve(labels)
        assert all(p == pytest.approx(1.0) for p in prec)
        assert rec[-1] == pytest.approx(1.0)

    def test_no_positives_returns_stub(self):
        labels = np.array([0, 0, 0])
        prec, rec = _precision_recall_curve(labels)
        assert prec == [1.0] and rec == [0.0]

    def test_recall_monotone(self):
        labels = np.array([1, 0, 1, 0, 1])
        _, rec = _precision_recall_curve(labels)
        for a, b in zip(rec, rec[1:]):
            assert b >= a


# ---------------------------------------------------------------------------
# Tests – _area_under_pr
# ---------------------------------------------------------------------------

class TestAreaUnderPR:
    def test_perfect_pr(self):
        # Precision stays 1.0 all the way, recall goes 0→1 → AP = 100%
        prec = [1.0, 1.0, 1.0]
        rec  = [0.0, 0.5, 1.0]
        ap = _area_under_pr(prec, rec)
        assert ap == pytest.approx(100.0, abs=0.1)

    def test_empty(self):
        assert _area_under_pr([], []) == 0.0

    def test_ap_between_zero_and_hundred(self):
        prec = [1.0, 0.5, 0.33]
        rec  = [0.33, 0.67, 1.0]
        ap = _area_under_pr(prec, rec)
        assert 0.0 <= ap <= 100.0


# ---------------------------------------------------------------------------
# Tests – compute_ap (top-level API)
# ---------------------------------------------------------------------------

class TestComputeAP:
    def test_all_perfect_grasps(self):
        cands = [_perfect_antipodal(score=float(i) / 10) for i in range(10, 0, -1)]
        result = compute_ap(cands)
        assert result.ap_04 == pytest.approx(100.0, abs=0.5)
        assert result.ap_08 == pytest.approx(100.0, abs=0.5)
        assert result.ap == pytest.approx(100.0, abs=0.5)
        assert result.n_positive_04 == 10
        assert result.n_positive_08 == 10

    def test_all_bad_grasps(self):
        cands = [_bad_grasp(score=float(i) / 10) for i in range(10, 0, -1)]
        result = compute_ap(cands)
        assert result.ap_04 == pytest.approx(0.0, abs=0.1)
        assert result.ap_08 == pytest.approx(0.0, abs=0.1)
        assert result.n_positive_04 == 0
        assert result.n_positive_08 == 0

    def test_empty_candidates(self):
        result = compute_ap([])
        assert result.ap == 0.0
        assert result.n_candidates == 0

    def test_ap04_le_ap08(self):
        # mu=0.4 is stricter, so AP_0.4 <= AP_0.8 in general
        cands = (
            [_perfect_antipodal(score=0.9 - i * 0.05) for i in range(5)]
            + [_bad_grasp(score=0.6 - i * 0.05) for i in range(5)]
        )
        result = compute_ap(cands)
        # Both should be in [0, 100]
        assert 0.0 <= result.ap_04 <= 100.0
        assert 0.0 <= result.ap_08 <= 100.0

    def test_n_candidates_correct(self):
        cands = [_perfect_antipodal(0.5)] * 7 + [_bad_grasp(0.3)] * 3
        result = compute_ap(cands)
        assert result.n_candidates == 10

    def test_summary_string(self):
        cands = [_perfect_antipodal(0.9), _bad_grasp(0.5)]
        result = compute_ap(cands)
        s = result.summary()
        assert "AP_0.4" in s
        assert "AP_0.8" in s


# ---------------------------------------------------------------------------
# Tests – per-split (seen / novel)
# ---------------------------------------------------------------------------

class TestPerSplitAP:
    def test_seen_only(self):
        cands = [_perfect_antipodal(0.9, split="seen") for _ in range(5)]
        result = compute_ap(cands)
        assert result.ap_seen == pytest.approx(100.0, abs=0.5)
        assert math.isnan(result.ap_novel)

    def test_novel_only(self):
        cands = [_perfect_antipodal(0.9, split="novel") for _ in range(5)]
        result = compute_ap(cands)
        assert math.isnan(result.ap_seen)
        assert result.ap_novel == pytest.approx(100.0, abs=0.5)

    def test_mixed_splits(self):
        cands = (
            [_perfect_antipodal(0.9, split="seen")  for _ in range(4)]
            + [_bad_grasp(0.8, split="novel") for _ in range(4)]
        )
        result = compute_ap(cands)
        assert result.ap_seen == pytest.approx(100.0, abs=0.5)
        assert result.ap_novel == pytest.approx(0.0, abs=0.1)


# ---------------------------------------------------------------------------
# Tests – compute_ap_per_object
# ---------------------------------------------------------------------------

class TestComputeAPPerObject:
    def test_two_objects(self):
        cands = (
            [_perfect_antipodal(0.9, obj="duck")  for _ in range(3)]
            + [_bad_grasp(0.9, obj="lego")        for _ in range(3)]
        )
        per_obj = compute_ap_per_object(cands, mu=0.8)
        assert "duck" in per_obj and "lego" in per_obj
        assert per_obj["duck"] == pytest.approx(100.0, abs=0.5)
        assert per_obj["lego"] == pytest.approx(0.0, abs=0.1)

    def test_empty(self):
        assert compute_ap_per_object([], mu=0.8) == {}


# ---------------------------------------------------------------------------
# Tests – metrics.py bridge
# ---------------------------------------------------------------------------

class TestComputeAPScoreBridge:
    def test_bridge_returns_ap_result(self):
        cands = [_perfect_antipodal(0.9) for _ in range(5)]
        result = compute_ap_score(cands)
        assert isinstance(result, APResult)
        assert result.ap == pytest.approx(100.0, abs=0.5)

    def test_bridge_custom_mu(self):
        cands = [_perfect_antipodal(0.9) for _ in range(3)]
        result = compute_ap_score(cands, mu_values=(0.2, 0.6))
        assert result.ap_04 == pytest.approx(100.0, abs=0.5)  # ap_04 stores mu_low result
