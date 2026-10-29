"""Pure-NumPy tensor / detection comparison helpers.

No ultralytics / torch / onnxruntime imports. New generic comparison
helpers belong here.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tolerance selection
# ---------------------------------------------------------------------------
# The strict pair catches export bugs / dtype drift on full-precision pairs
# (PT <-> ONNX FP32); the loose pair absorbs reduced-precision noise
# (INT8 / FP16). See CLAUDE.md invariant 12.
TOLERANCE_PAIRS: Dict[str, Tuple[float, float]] = {
    "strict": (1e-4, 1e-3),
    "loose": (1e-3, 1e-2),
}

# Stem / backend-prefix markers that classify a side as reduced precision.
REDUCED_PRECISION_MARKERS = ("int8", "int4", "fp16", "f16", "half", "quant")
# The backend prefixes ModelWrapper parses (mirror of the class constants in
# src/consistency.py — kept as a literal so this module stays import-light).
_BACKEND_PREFIXES = ("openvino_int8:", "openvino:", "ort_cpp:", "tensorrt:")
# Suffixes a full-precision artifact can carry. .xml (OpenVINO IR — FP16 by
# default at convert time) and .engine (TensorRT) are deliberately absent.
_FULL_PRECISION_SUFFIXES = (".pt", ".pth", ".onnx")


def _strip_backend_prefix(spec: str) -> Tuple[str, str]:
    """Split a ``backend:path`` spec into ``(prefix, path)``; a plain path
    yields ``("", spec)``."""
    spec = str(spec)
    for prefix in _BACKEND_PREFIXES:
        if spec.startswith(prefix):
            return prefix, spec[len(prefix):]
    return "", spec


def _side_is_full_precision(spec: str) -> bool:
    """True iff one model spec looks full-precision, from its text alone.

    All three must hold: the backend prefix carries no reduced-precision
    marker; the path suffix is in the full-precision allowlist; the path
    stem (not the directories — a dataset dir named ``int8_set/`` must not
    misclassify) carries no marker. Marker scan over ``prefix + stem`` also
    classifies precision-labeled prefixes (``openvino_int8:``) without a
    special case.
    """
    prefix, path = _strip_backend_prefix(spec)
    if any(m in prefix.lower() for m in REDUCED_PRECISION_MARKERS):
        return False
    if not path.lower().endswith(_FULL_PRECISION_SUFFIXES):
        return False
    stem = Path(path).stem.lower()
    return not any(m in stem for m in REDUCED_PRECISION_MARKERS)


def select_tolerance_pair(model1_spec: str, model2_spec: str) -> Tuple[str, float, float]:
    """Pick the consistency tolerance pair from the model-pair stems.

    Returns ``(level, atol, rtol)``: the strict pair (atol 1e-4 / rtol
    1e-3) when *both* sides look full-precision, the loose pair (1e-3 /
    1e-2) otherwise. Symmetric by construction — swapping the two specs
    cannot change the result. An unmarked ``.onnx`` pairs strict (the
    repo's FP32 exports are named ``*_fp32.onnx``; ``fp32`` matches no
    marker). Deterministic: pure string classification, no I/O.
    """
    level = (
        "strict"
        if _side_is_full_precision(model1_spec) and _side_is_full_precision(model2_spec)
        else "loose"
    )
    atol, rtol = TOLERANCE_PAIRS[level]
    return level, atol, rtol


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
    """Per-channel-group stats over the class-score channels.

    ``cls_max_diff`` / ``cls_mean_diff`` / ``cls_p99`` / ``cls_cos`` are computed over channels
    ``[4:4+nc]``. With ``conf_cliff`` set, per-side above-cliff counts and promoted/demoted
    flips are also reported.
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
        # Per-side counts above conf_cliff, plus directional flips (below→above
        # / above→below).
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

    In ``tensor`` mode, when ``conf_cliff`` is set, the comparison also fails when the per-image
    box-count flip across the cliff (``max(promoted, demoted) / bs``) exceeds
    ``cliff_count_thresh``, even when allclose passes. ``cliff_count_thresh`` is per-image
    (normalized by batch size).

    The two verdicts are exposed separately (``allclose_passed`` /
    ``detection_consistent``); ``passed`` requires both.
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

    # Class-channel diagnostics (informational). Skipped when nc cannot be
    # inferred from either output.
    resolved_nc = nc if nc is not None else _infer_nc(out1)
    if resolved_nc is not None and _infer_nc(out2) == resolved_nc:
        try:
            stats.update(_class_channel_stats(out1, out2, resolved_nc, conf_cliff))
            stats["nc"] = resolved_nc
        except Exception:
            logger.debug(
                "class-channel diagnostics skipped (shapes %s / %s, nc=%s)",
                out1.shape, out2.shape, resolved_nc, exc_info=True,
            )

    if mode == "tensor":
        allclose_ok = True
        allclose_err = ""
        try:
            np.testing.assert_allclose(out1, out2, atol=atol, rtol=rtol)
        except AssertionError as e:
            allclose_ok = False
            allclose_err = str(e)[:500]

        # Tensor-mode gate: fail when per-image cliff flips exceed
        # cliff_count_thresh (normalized by batch size).
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
                    f"above_cliff {stats['above_cliff_1']}→{stats['above_cliff_2']}."
                )
                if allclose_ok:
                    cliff_err += (
                        f" allclose passes (cls_max_diff={stats['cls_max_diff']:.2e}): "
                        f"the divergence sits within its tolerance at the cliff value — "
                        f"expect spurious detections at this conf in deployment."
                    )

        # Report allclose_passed and detection_consistent separately;
        # passed = allclose AND cliff gate.
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
    model's. Neither is ground truth — both are predictions.
    ``recall_match_rate`` in the caller is matched/len(dets1).
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
        # best_j is -1 when no candidate overlaps at all; require a real
        # candidate so iou_threshold == 0 cannot match the sentinel.
        if best_j >= 0 and best_iou >= iou_threshold:
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
    model's — neither is ground truth. ``recall_match_rate = matched /
    max(len(dets1), 1)`` (the ``max`` guards the empty-``dets1`` case).
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
    "select_tolerance_pair",
    "TOLERANCE_PAIRS",
    "REDUCED_PRECISION_MARKERS",
]
