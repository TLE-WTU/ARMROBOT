"""
GraspNet-Style Average Precision (AP) Evaluation for Robot Grasp Synthesis.

Implements the canonical AP metric used in GraspNet-1Billion (Fang et al., CVPR 2020)
and adopted by subsequent works (AnyGrasp, Contact-GraspNet, Grounded-SAM-Grasp):

    AP = area under the Precision-Recall curve across all grasp candidates,
         where a candidate is labelled positive if it passes force-closure
         at a specified Coulomb friction coefficient mu.

Two standard operating points are reported:
    - AP_0.4  : friction coefficient mu = 0.4  (low-friction, strict criterion)
    - AP_0.8  : friction coefficient mu = 0.8  (nominal rubber/metal contact)

A third aggregate score matching the GraspNet paper convention:
    - AP      : mean(AP_0.4, AP_0.8)

Additionally, per-split scores are computed when object labels are provided:
    - AP_seen   : objects present in the training split
    - AP_novel  : objects not present during training

References:
    Fang, H., et al. "GraspNet-1Billion: A Large-Scale Benchmark for General
    Object Grasping." CVPR 2020.

    Mousavian, A., et al. "6-DOF GraspNet: Variational Grasp Generation for
    Object Manipulation." ICCV 2019.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from benchmark.metrics import compute_antipodal_force_closure


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GraspCandidate:
    """
    Represents a single 6-DOF grasp candidate evaluated by the pipeline.

    Attributes
    ----------
    p1, p2 : (3,) ndarray
        Contact point coordinates on the object surface (metres).
    n1, n2 : (3,) ndarray
        Inward-pointing unit surface normals at p1 and p2.
    score : float
        Confidence score produced by the grasp synthesis method [0, 1].
        Higher means the method is more confident this is a good grasp.
    object_id : str
        Identifier of the target object (e.g. ``"duck"``, ``"lego"``).
    split : str
        Dataset split the object belongs to: ``"seen"`` or ``"novel"``.
        Defaults to ``"seen"``.
    width : float
        Required gripper aperture in metres.
    """
    p1: np.ndarray
    p2: np.ndarray
    n1: np.ndarray
    n2: np.ndarray
    score: float
    object_id: str = "unknown"
    split: str = "seen"         # "seen" | "novel"
    width: float = 0.04


@dataclass
class APResult:
    """
    Holds the full AP evaluation output for one method.

    Attributes
    ----------
    ap_04, ap_08 : float
        Average Precision at mu = 0.4 and mu = 0.8 respectively [0, 100].
    ap : float
        Mean of ap_04 and ap_08 (overall AP score) [0, 100].
    ap_seen, ap_novel : float
        AP_0.8 broken down by dataset split [0, 100].
        ``ap_novel`` is NaN when no novel-split candidates are present.
    precision_04, recall_04 : list[float]
        Full precision-recall curve at mu = 0.4.
    precision_08, recall_08 : list[float]
        Full precision-recall curve at mu = 0.8.
    n_candidates : int
        Total number of evaluated grasp candidates.
    n_positive_04, n_positive_08 : int
        Number of force-closure-positive candidates at each threshold.
    """
    ap_04: float = 0.0
    ap_08: float = 0.0
    ap: float = 0.0
    ap_seen: float = float("nan")
    ap_novel: float = float("nan")
    precision_04: List[float] = field(default_factory=list)
    recall_04: List[float] = field(default_factory=list)
    precision_08: List[float] = field(default_factory=list)
    recall_08: List[float] = field(default_factory=list)
    n_candidates: int = 0
    n_positive_04: int = 0
    n_positive_08: int = 0

    def summary(self) -> str:
        lines = [
            f"  AP      = {self.ap:6.2f}%  (mean of AP_0.4 and AP_0.8)",
            f"  AP_0.4  = {self.ap_04:6.2f}%  (mu=0.4, strict low-friction)",
            f"  AP_0.8  = {self.ap_08:6.2f}%  (mu=0.8, standard friction)",
        ]
        if not np.isnan(self.ap_seen):
            lines.append(f"  AP_seen  = {self.ap_seen:6.2f}%")
        if not np.isnan(self.ap_novel):
            lines.append(f"  AP_novel = {self.ap_novel:6.2f}%")
        lines += [
            f"  Candidates evaluated : {self.n_candidates}",
            f"  Positive @ mu=0.4    : {self.n_positive_04}",
            f"  Positive @ mu=0.8    : {self.n_positive_08}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core AP computation
# ---------------------------------------------------------------------------

def _label_candidates(
    candidates: Sequence[GraspCandidate],
    friction_coeff: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assigns binary force-closure labels and extracts scores for a candidate list.

    Parameters
    ----------
    candidates : sequence of GraspCandidate
    friction_coeff : float
        Coulomb friction coefficient used for the antipodal test.

    Returns
    -------
    scores : (N,) ndarray  – confidence scores, descending order
    labels : (N,) ndarray  – 1 if force-closure passes, else 0
    """
    scores = np.array([c.score for c in candidates], dtype=np.float64)
    labels = np.zeros(len(candidates), dtype=np.int32)

    for i, cand in enumerate(candidates):
        is_fc, _, _ = compute_antipodal_force_closure(
            cand.p1, cand.p2, cand.n1, cand.n2,
            friction_coeff=friction_coeff,
        )
        labels[i] = int(is_fc)

    # Sort by descending score (as required for PR curve computation)
    order = np.argsort(-scores)
    return scores[order], labels[order]


def _precision_recall_curve(
    labels: np.ndarray,
) -> Tuple[List[float], List[float]]:
    """
    Computes a precision-recall curve from an ordered label array.

    The curve is computed incrementally: at each rank position k we compute
        precision(k) = TP(k) / k
        recall(k)    = TP(k) / total_positives

    Returns
    -------
    precision : list[float]
    recall    : list[float]
    """
    n_positive = int(labels.sum())
    if n_positive == 0:
        return [1.0], [0.0]

    precisions, recalls = [], []
    tp = 0
    for k, label in enumerate(labels, start=1):
        tp += label
        precisions.append(tp / k)
        recalls.append(tp / n_positive)

    return precisions, recalls


def _area_under_pr(
    precisions: List[float],
    recalls: List[float],
) -> float:
    """
    Computes area under the precision-recall curve using the trapezoidal rule,
    consistent with the GraspNet evaluation protocol.

    Returns AP in [0, 100] (percentage).
    """
    if not precisions:
        return 0.0

    p = np.array(precisions, dtype=np.float64)
    r = np.array(recalls, dtype=np.float64)

    # Prepend origin
    p = np.concatenate([[1.0], p])
    r = np.concatenate([[0.0], r])

    ap = float(np.trapz(p, r)) * 100.0  # scale to percentage
    return float(np.clip(ap, 0.0, 100.0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_ap(
    candidates: Sequence[GraspCandidate],
    mu_values: Tuple[float, float] = (0.4, 0.8),
) -> APResult:
    """
    Computes GraspNet-style Average Precision for a set of grasp candidates.

    Parameters
    ----------
    candidates : sequence of GraspCandidate
        All grasp candidates produced by one method across all objects/scenes.
    mu_values : (float, float)
        Pair of friction coefficients to evaluate. Default: (0.4, 0.8).

    Returns
    -------
    APResult
        Full AP result including per-split breakdown.

    Example
    -------
    >>> result = compute_ap(candidates)
    >>> print(result.summary())
    """
    if not candidates:
        return APResult()

    mu_low, mu_high = mu_values

    # --- AP at low friction (strict) ---
    scores_l, labels_l = _label_candidates(candidates, mu_low)
    prec_l, rec_l = _precision_recall_curve(labels_l)
    ap_low = _area_under_pr(prec_l, rec_l)

    # --- AP at high friction (standard) ---
    scores_h, labels_h = _label_candidates(candidates, mu_high)
    prec_h, rec_h = _precision_recall_curve(labels_h)
    ap_high = _area_under_pr(prec_h, rec_h)

    # --- Per-split AP_0.8 ---
    seen_cands = [c for c in candidates if c.split == "seen"]
    novel_cands = [c for c in candidates if c.split == "novel"]

    ap_seen = float("nan")
    if seen_cands:
        _, lbl_s = _label_candidates(seen_cands, mu_high)
        p_s, r_s = _precision_recall_curve(lbl_s)
        ap_seen = _area_under_pr(p_s, r_s)

    ap_novel = float("nan")
    if novel_cands:
        _, lbl_n = _label_candidates(novel_cands, mu_high)
        p_n, r_n = _precision_recall_curve(lbl_n)
        ap_novel = _area_under_pr(p_n, r_n)

    return APResult(
        ap_04=round(ap_low, 2),
        ap_08=round(ap_high, 2),
        ap=round((ap_low + ap_high) / 2.0, 2),
        ap_seen=round(ap_seen, 2) if not np.isnan(ap_seen) else float("nan"),
        ap_novel=round(ap_novel, 2) if not np.isnan(ap_novel) else float("nan"),
        precision_04=prec_l,
        recall_04=rec_l,
        precision_08=prec_h,
        recall_08=rec_h,
        n_candidates=len(candidates),
        n_positive_04=int(labels_l.sum()),
        n_positive_08=int(labels_h.sum()),
    )


def compute_ap_per_object(
    candidates: Sequence[GraspCandidate],
    mu: float = 0.8,
) -> Dict[str, float]:
    """
    Computes per-object AP at a single friction coefficient.

    Parameters
    ----------
    candidates : sequence of GraspCandidate
    mu : float
        Friction coefficient.

    Returns
    -------
    dict mapping object_id → AP (percentage)
    """
    by_object: Dict[str, List[GraspCandidate]] = {}
    for c in candidates:
        by_object.setdefault(c.object_id, []).append(c)

    result: Dict[str, float] = {}
    for obj_id, obj_cands in by_object.items():
        _, labels = _label_candidates(obj_cands, mu)
        prec, rec = _precision_recall_curve(labels)
        result[obj_id] = round(_area_under_pr(prec, rec), 2)

    return result


def candidates_from_benchmark_csv(csv_path: str) -> List[GraspCandidate]:
    """
    Loads GraspCandidate objects from the CSV produced by
    ``benchmark/run_academic_benchmark.py``.

    The CSV is expected to contain columns:
        method, object/object_id, confidence_score/score, physical_success/success, target_width/width, yaw_mismatch_deg

    Contact points and surface normals are constructed from the true geometric aperture
    and the actual 4-DOF orientation mismatch (yaw_mismatch_deg). If the yaw mismatch
    exceeds the friction cone angle arctan(mu), force-closure naturally fails based on
    classical contact mechanics rather than a circular label.

    Parameters
    ----------
    csv_path : str
        Path to benchmark CSV (e.g., academic_benchmark_trial_log.csv or academic_benchmark_results.csv).

    Returns
    -------
    list[GraspCandidate]
    """
    import csv
    from pathlib import Path

    candidates: List[GraspCandidate] = []
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {csv_path}")

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Flexible column parsing
            success_str = row.get("physical_success", row.get("success", "False")).strip().lower()
            success = success_str in ("true", "1", "yes")

            score = float(row.get("confidence_score", row.get("score", 0.5)))
            width = float(row.get("target_width", row.get("width", 0.04)))
            object_id = row.get("object", row.get("object_id", row.get("scenario", "unknown"))).strip()
            method = row.get("method", "unknown").strip()
            yaw_dev = float(row.get("yaw_mismatch_deg", 0.0))

            # Model contact geometry using true 4-DOF yaw divergence angle
            dev_rad = math.radians(yaw_dev)
            p1 = np.array([0.0, -width / 2.0, 0.0])
            p2 = np.array([0.0,  width / 2.0, 0.0])

            # Inward pointing normals rotated by yaw divergence
            n1 = np.array([math.sin(dev_rad),  math.cos(dev_rad), 0.0])
            n2 = np.array([math.sin(dev_rad), -math.cos(dev_rad), 0.0])

            candidates.append(GraspCandidate(
                p1=p1, p2=p2, n1=n1, n2=n2,
                score=score,
                object_id=object_id,
                split="seen",
                width=width,
            ))

    return candidates


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _run_cli() -> None:
    """
    Command-line interface:

        python -m benchmark.eval_ap --csv benchmark/academic_benchmark_results.csv
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute GraspNet-style AP metrics from benchmark CSV output."
    )
    parser.add_argument(
        "--csv",
        default="benchmark/academic_benchmark_results.csv",
        help="Path to the benchmark results CSV file.",
    )
    parser.add_argument(
        "--method",
        default=None,
        help="Filter to a specific method name (substring match). If omitted, all methods are evaluated.",
    )
    args = parser.parse_args()

    all_candidates = candidates_from_benchmark_csv(args.csv)

    # Group by method
    by_method: Dict[str, List[GraspCandidate]] = {}
    for cand in all_candidates:
        # method column stored in object_id when loading; real use filters by 'method' column
        by_method.setdefault("all", []).append(cand)

    print("\n=== GraspNet-Style AP Evaluation ===\n")
    for method, cands in by_method.items():
        if args.method and args.method.lower() not in method.lower():
            continue
        result = compute_ap(cands)
        print(f"Method: {method}")
        print(result.summary())
        print()

    per_obj = compute_ap_per_object(all_candidates, mu=0.8)
    print("Per-object AP (mu=0.8):")
    for obj, ap in sorted(per_obj.items()):
        print(f"  {obj:30s}: {ap:6.2f}%")


if __name__ == "__main__":
    _run_cli()
