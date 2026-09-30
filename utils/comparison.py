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


def _infer_nc(out: np.ndarray) -> "int | None":
    """Infer the class-channel count for a YOLOv8 ``(*, 4+nc, N)`` output.

    YOLOv8 exports as ``(bs, 4+nc, N)`` with the channel dim on axis 1. Box coords occupy the
    first 4 channels (pixel-scale, large magnitude); class scores occupy channels ``4..4+nc``
    (post-sigmoid, 0..1). When the channel dim is recoverable we can split the two groups and
    report the class channel alone — otherwise we bail (a transposed ``(bs, N, 4+nc)`` tensor
    or a non-3D tensor has no safe split point).

    The channel dim is "axis 1, small, and smaller than axis 2": standard export has
    ``4+nc=16`` vs ``N=8400``. The ``[5, 64]`` bound rejects obviously-wrong inferences (e.g.
    a pruned head with >60 classes, or a non-YOLO tensor) — pass ``nc`` explicitly there.
    """
    if out.ndim != 3:
        return None
    c, n = out.shape[1], out.shape[2]
    if 5 <= c <= 64 and c < n:
        return c - 4
    return None


def _class_channel_stats(
    out1: np.ndarray, out2: np.ndarray, nc: int, conf_cliff: "float | None"
) -> Dict:
    """Per-channel-group stats that the full-tensor cosine masks.

    The full-tensor ``cosine_similarity`` is dominated by the 4 box-coord channels (pixel-scale,
    ~0..640) over the ``nc`` class-score channels (0..1). A divergence that shifts ~1e-3 on
    class scores can land right on a ``conf`` decision cliff and promote dozens of background
    boxes to detections, while the full-tensor cosine stays >0.999 — a false-pass in tensor mode.
    Splitting the class channel out surfaces that divergence directly, and counting boxes above
    the cliff per side predicts the post-conf-filter detection-count disparity.
    """
    cls1 = out1[:, 4:4 + nc, :].astype(np.float64)
    cls2 = out2[:, 4:4 + nc, :].astype(np.float64)
    cdiff = np.abs(cls1 - cls2)
    # Per-box max class score (the value the conf filter actually thresholds).
    m1 = cls1.max(axis=1)  # (bs, N)
    m2 = cls2.max(axis=1)
    stats: Dict = {
        "cls_max_diff": float(np.max(cdiff)) if cdiff.size else 0.0,
        "cls_mean_diff": float(np.mean(cdiff)) if cdiff.size else 0.0,
        "cls_p99": _safe_pct(cdiff, 99),
        "cls_cos": cosine_similarity(cls1, cls2),
    }
    if conf_cliff is not None and conf_cliff > 0:
        # Expected surviving-box count after the conf filter, per side — the metric that
        # actually explains an infer-time detection explosion. ``promoted``/``demoted`` are
        # directional (out1 → out2): boxes the divergence pushed across the cliff relative to
        # the reference. A near-zero raw-tensor divergence can still flip many boxes here because
        # the conf threshold sits on the score-distribution's uncertainty mode — a hard cliff that
        # amplifies FP32-noise-level (≤3.5e-4) cross-EP divergence into a 30× detection-count gap
        # that ``np.allclose`` cannot see (the divergence is within its tolerance at that value).
        stats["above_cliff_1"] = int(np.sum(m1 >= conf_cliff))
        stats["above_cliff_2"] = int(np.sum(m2 >= conf_cliff))
        stats["cliff_promoted_1to2"] = int(
            np.sum((m1 < conf_cliff) & (m2 >= conf_cliff))
        )
        stats["cliff_demoted_1to2"] = int(
            np.sum((m1 >= conf_cliff) & (m2 < conf_cliff))
        )
    return stats


def compare_tensors(
    out1: np.ndarray,
    out2: np.ndarray,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    cosine_similarity_thresh: float = 0.995,
    mean_diff_thresh: float = 0.01,
    p99_thresh: float = 0.05,
    mode: str = "tensor",
    nc: "int | None" = None,
    conf_cliff: "float | None" = None,
    cliff_count_thresh: float = 5.0,
) -> Dict:
    """Compare two raw output tensors. Returns a stats dict with ``passed``.

    Modes
    -----
    * ``tensor`` — strict ``np.allclose`` at the given ``atol`` / ``rtol``.
      Catches export bugs (graph rewrite errors, dtype drift).
    * ``detection`` — relaxed: cosine ≥ threshold, mean_diff ≤ threshold,
      p99 ≤ threshold. Advisory in ``detection`` mode — the authoritative
      gate is per-image IoU / class / score (see ``compare_detections``).

    Class-channel diagnostics & the conf-cliff gate (``nc`` / ``conf_cliff``)
    -----------------------------------------------------------------------
    When ``nc`` is given (or inferable from a ``(bs, 4+nc, N)`` layout), the class-score
    channels are split out and reported separately (``cls_max_diff`` / ``cls_cos`` /
    ``above_cliff_*``). The full-tensor cosine is dominated by the large-magnitude box coords
    and can mask a class-score divergence small enough to cross a ``conf`` cliff and inflate
    detection counts — these stats surface it.

    In ``tensor`` mode, when ``conf_cliff`` is set, the cliff gate is **authoritative alongside
    allclose**: if the per-image box-count flip across the cliff (``max(promoted, demoted) / bs``)
    exceeds ``cliff_count_thresh``, the comparison FAILS even when allclose passes. This catches
    the failure mode allclose structurally cannot — a cross-EP numeric divergence within FP32
    noise (≤3.5e-4 at a 0.25 cliff) that the hard conf threshold amplifies into a spurious
    detection explosion. ``cliff_count_thresh`` is per-image (normalized by batch size).

    The two verdicts are also exposed **side by side** (``allclose_passed`` /
    ``detection_consistent``) rather than merged into ``passed``. For a reduced-precision side
    (FP16/INT8) raw allclose structurally cannot pass at any tolerance that still catches the
    real regression, yet the cliff gate stays clean — the detections ARE consistent. Reporting
    both truths (raw FAIL + detection PASS) instead of one merged verdict is the
    engineering-honest posture: it neither masks the raw divergence nor masks the detection
    consistency. The composite ``passed`` (allclose AND cliff) stays the strict raw-parity gate.
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

    # Class-channel diagnostics (informational). Bypassed on a shape mismatch or a
    # non-YOLOv8 layout — the allclose / advisory gate below is the source of truth for
    # ``passed``; these fields just make a cliff-amplified divergence visible in the report.
    resolved_nc = nc if nc is not None else _infer_nc(out1)
    if resolved_nc is not None and _infer_nc(out2) == resolved_nc:
        try:
            stats.update(_class_channel_stats(out1, out2, resolved_nc, conf_cliff))
            stats["nc"] = resolved_nc
        except Exception:
            pass

    if mode == "tensor":
        allclose_ok = True
        allclose_err = ""
        try:
            np.testing.assert_allclose(out1, out2, atol=atol, rtol=rtol)
        except AssertionError as e:
            allclose_ok = False
            allclose_err = str(e)[:500]

        # Conf-cliff gate — authoritative in tensor mode alongside allclose. allclose cannot
        # see a divergence that lives within FP32 noise yet crosses the hard conf threshold
        # (the divergence is ≤ allclose's own tolerance at the cliff value, so it passes). The
        # per-image box-count flip across the cliff is the only signal that catches it. On a
        # stable pair (same EP / CPU) promoted≈demoted≈0; on a cross-EP pair that destabilizes
        # the cliff it jumps to dozens. ``cliff_count_thresh`` is per image (÷ batch size).
        cliff_ok = True
        cliff_err = ""
        if conf_cliff is not None and "above_cliff_1" in stats:
            bs = max(out1.shape[0], 1)
            flip = max(stats["cliff_promoted_1to2"], stats["cliff_demoted_1to2"]) / bs
            if flip > cliff_count_thresh:
                cliff_ok = False
                cliff_err = (
                    f"conf-cliff destabilized: {flip:.1f} boxes/img flipped across "
                    f"conf={conf_cliff} (promoted={stats['cliff_promoted_1to2']}, "
                    f"demoted={stats['cliff_demoted_1to2']}, bs={bs}); "
                    f"above_cliff {stats['above_cliff_1']}→{stats['above_cliff_2']}. "
                    f"allclose alone passes (cls_max_diff={stats['cls_max_diff']:.2e}) "
                    f"because the divergence sits within its tolerance at the cliff value — "
                    f"expect a spurious-detection explosion at this conf in deployment."
                )

        # Dual verdict: raw-tensor allclose and the detection-level conf-cliff gate are
        # reported SIDE BY SIDE, not merged. For a reduced-precision side (FP16/INT8) raw
        # allclose structurally cannot pass — FP16 box-coord divergence on the head's
        # unbounded accumulation exceeds any tolerance loose enough to still catch the real
        # regression — yet the cliff gate stays clean (the detections ARE consistent). Merging
        # would either mask the raw divergence (cliff-authoritative) or mask the detection
        # consistency (allclose-authoritative). The composite ``passed`` below stays the
        # strict raw-parity gate (allclose AND cliff) for back-compat; the two explicit fields
        # let the report show both truths rather than one misleading PASS/FAIL.
        stats["allclose_passed"] = allclose_ok
        if conf_cliff is not None and "above_cliff_1" in stats:
            stats["detection_consistent"] = cliff_ok

        if allclose_ok and cliff_ok:
            stats.update({"passed": True, "status": "PASS"})
        else:
            stats["passed"] = False
            stats["status"] = "FAIL"
            stats["error_msg"] = " | ".join(m for m in (allclose_err, cliff_err) if m)
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
