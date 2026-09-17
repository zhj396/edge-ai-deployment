"""Pure-NumPy tensor / detection comparison helpers.

Why this module exists separate from ``src.consistency``: ``src.consistency``
imports ``ultralytics.YOLO`` at module load for its PyTorch wrapper. The
comparison functions below never touch ultralytics / torch / onnxruntime,
so pulling them out lets the pure-Python test suite run without
instantiating the YOLO stack.

Backward compatibility: ``src.consistency`` re-exports these names so
existing call sites (``from src.consistency import compare_tensors`` etc.)
keep working — this module is the new canonical location.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Tensor comparison
# ---------------------------------------------------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity on flattened float64 vectors; 0.0 on zero norm."""
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def _safe_pct(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, q))


def compare_tensors(
    out1: np.ndarray,
    out2: np.ndarray,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    cosine_similarity_thresh: float = 0.995,
    mean_diff_thresh: float = 0.01,
    p99_thresh: float = 0.05,
    mode: str = "tensor",
) -> Dict:
    """Compare two raw output tensors. Returns a stats dict with ``passed``.

    Modes
    -----
    * ``tensor`` — strict ``np.allclose`` at the given ``atol`` / ``rtol``.
      Catches export bugs (graph rewrite errors, dtype drift).
    * ``detection`` — relaxed: cosine ≥ threshold, mean_diff ≤ threshold,
      p99 ≤ threshold. Advisory in ``detection`` mode — the authoritative
      gate is per-image IoU / class / score (see ``compare_detections``).
    """
    stats: Dict = {"passed": False, "status": "FAIL"}

    if out1.shape != out2.shape:
        stats["error_msg"] = f"shape mismatch: {out1.shape} vs {out2.shape}"
        return stats

    if np.any(np.isnan(out1)) or np.any(np.isnan(out2)):
        stats["error_msg"] = "NaN detected"
        return stats

    diff = np.abs(out1 - out2)
    stats.update({
        "max_diff": float(np.max(diff)),
        "mean_diff": float(np.mean(diff)),
        "p95": _safe_pct(diff, 95),
        "p99": _safe_pct(diff, 99),
        "std_diff": float(np.std(diff)),
        "cosine_similarity": cosine_similarity(out1, out2),
    })

    if mode == "tensor":
        try:
            np.testing.assert_allclose(out1, out2, atol=atol, rtol=rtol)
            stats.update({"passed": True, "status": "PASS"})
        except AssertionError as e:
            stats["error_msg"] = str(e)[:500]
        return stats

    # detection mode — relaxed (advisory; per-image gates decide)
    passed = (
        stats["cosine_similarity"] >= cosine_similarity_thresh
        and stats["mean_diff"] <= mean_diff_thresh
        and stats["p99"] <= p99_thresh
    )
    stats.update({
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
    })
    if not passed:
        stats["error_msg"] = (
            f"Tensor mismatch | cos={stats['cosine_similarity']:.6f}, "
            f"mean_diff={stats['mean_diff']:.6f}, p99={stats['p99']:.6f}"
        )
    return stats


# ---------------------------------------------------------------------------
# Detection comparison
# ---------------------------------------------------------------------------
def compute_iou(box1, box2) -> float:
    """IoU between two axis-aligned ``(x1, y1, x2, y2)`` boxes."""
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    a2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _match_predictions_to_gt(
    dets1, dets2, iou_threshold: float
) -> List[Tuple[int, float, float, int]]:
    """Greedy 1:1 matching between dets1 and dets2.

    Returns a list of ``(idx_in_dets2, iou, score_diff, class_match)``. Each
    entry in ``dets1`` produces at most one match (the best IoU above
    threshold), and each entry in ``dets2`` is consumed once.

    Note: "dets1" is the *first* model's detections; "dets2" is the *second*
    model's. Neither is ground truth — both are predictions. ``recall_match``
    is computed against ``dets1`` by convention.
    """
    used = set()
    matches: List[Tuple[int, float, float, int]] = []
    for d1 in dets1:
        best_iou, best_j = 0.0, -1
        for j, d2 in enumerate(dets2):
            if j in used:
                continue
            iou = compute_iou(d1[:4], d2[:4])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_threshold:
            used.add(best_j)
            class_match = int(int(d1[5]) == int(dets2[best_j][5]))
            score_diff = abs(d1[4] - dets2[best_j][4])
            matches.append((best_j, best_iou, score_diff, class_match))
    return matches


def compare_detections(
    dets1, dets2,
    iou_threshold: float = 0.5,
    mean_iou_thresh: float = 0.92,
    class_match_thresh: float = 0.98,
    score_diff_thresh: float = 0.08,
    count_diff_thresh: int = 3,
    recall_match_thresh: float = 0.97,
) -> Dict:
    """Compare two sets of detections ``[(x1,y1,x2,y2,conf,cls), ...]``.

    ``dets1`` is the *first* model's detections and ``dets2`` is the *second*
    model's — neither is ground truth. ``recall_match_rate = matched / len(dets1)``.
    """
    stats: Dict = {
        "passed": False,
        "matched": 0,
        "mean_iou": 0.0,
        "class_match_rate": 0.0,
        "score_diff_mean": 0.0,
        "count_diff": abs(len(dets1) - len(dets2)),
        "recall_match_rate": 0.0,
    }

    if not dets1 and not dets2:
        stats.update({
            "passed": True, "mean_iou": 1.0,
            "class_match_rate": 1.0, "recall_match_rate": 1.0,
        })
        return stats

    matches = _match_predictions_to_gt(dets1, dets2, iou_threshold)
    ious = [m[1] for m in matches]
    score_diffs = [m[2] for m in matches]
    class_matches = sum(m[3] for m in matches)
    matched = len(matches)

    stats.update({
        "matched": matched,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "class_match_rate": float(class_matches / matched) if matched else 0.0,
        "score_diff_mean": float(np.mean(score_diffs)) if score_diffs else 0.0,
        "count_diff": abs(len(dets1) - len(dets2)),
        "recall_match_rate": matched / max(len(dets1), 1),
    })

    stats["passed"] = (
        stats["mean_iou"] >= mean_iou_thresh
        and stats["class_match_rate"] >= class_match_thresh
        and stats["score_diff_mean"] <= score_diff_thresh
        and stats["count_diff"] <= count_diff_thresh
        and stats["recall_match_rate"] >= recall_match_thresh
    )
    return stats


__all__ = [
    "cosine_similarity",
    "compare_tensors",
    "compute_iou",
    "compare_detections",
]
