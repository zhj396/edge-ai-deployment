"""Tests for the backend-agnostic COCO mAP evaluator (utils/map_eval.py).

Model-free: no .pt / .onnx / GPU required. The IoU + AP functions are pure
NumPy and are exercised with hand-computed synthetic scenarios. The GT-loader
test is data-guarded (skips if data/images/val is absent). Mirrors the project's
test posture (CLAUDE.md: pytest -q runs in ~5s, no model).
"""
from __future__ import annotations

import numpy as np
import pytest

from utils.map_eval import box_iou, compute_map


# ---------------------------------------------------------------------------
# box_iou
# ---------------------------------------------------------------------------
def test_box_iou_known():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10],   # identical → 1.0
                  [5, 5, 15, 15],   # overlap 5x5=25, union 100+100-25=175 → 25/175
                  [20, 20, 30, 30]])  # no overlap → 0.0
    iou = box_iou(a, b)
    assert iou.shape == (1, 3)
    np.testing.assert_allclose(iou[0], [1.0, 25.0 / 175.0, 0.0], atol=1e-9)


def test_box_iou_empty_guard():
    # Empty either side → zeros with the right shape, never NaN.
    assert box_iou(np.zeros((0, 4)), np.array([[0, 0, 1, 1]])).shape == (0, 1)
    assert box_iou(np.array([[0, 0, 1, 1]]), np.zeros((0, 4))).shape == (1, 0)
    assert box_iou(np.zeros((0, 4)), np.zeros((0, 4))).shape == (0, 0)


def test_box_iou_degenerate_box():
    # A zero-area box (x2<=x1) must give IoU 0, not NaN.
    a = np.array([[5, 5, 5, 5]], dtype=float)  # zero area
    b = np.array([[0, 0, 10, 10]], dtype=float)
    iou = box_iou(a, b)
    assert iou.shape == (1, 1)
    assert iou[0, 0] == 0.0
    assert not np.isnan(iou).any()


# ---------------------------------------------------------------------------
# compute_map — hand-computed COCO AP
# ---------------------------------------------------------------------------
def _dets(stem, cls, boxes_conf):
    """Build detections {stem: [(cls, x1,y1,x2,y2, conf), ...]}."""
    return {stem: [(cls, x1, y1, x2, y2, conf) for (x1, y1, x2, y2, conf) in boxes_conf]}


def _gts(stem, cls_boxes, cls=0):
    """Build gts {stem: (N,5) [cls,x1,y1,x2,y2]}."""
    arr = np.array([[cls, x1, y1, x2, y2] for (x1, y1, x2, y2) in cls_boxes],
                   dtype=float)
    return {stem: arr if len(arr) else np.zeros((0, 5))}


def test_compute_map_perfect():
    # 1 class, 1 image, 1 GT box, 1 detection matching it (IoU 1.0).
    dets = _dets("img", 0, [(10, 10, 20, 20, 0.9)])
    gts = _gts("img", [(10, 10, 20, 20)])
    map50, map5095, rows = compute_map(dets, gts, num_classes=1)
    assert map50 == pytest.approx(1.0)
    assert map5095 == pytest.approx(1.0)
    assert rows[0]["mAP50"] == pytest.approx(1.0)


def test_compute_map_missed_gt():
    # 1 class, 1 image, 2 GT boxes, 1 detection matching one GT (IoU 1.0).
    # recall = 1/2 = 0.5; the single PR point is (recall=0.5, precision=1.0).
    # COCO 101-point: r in [0, 0.5] → precision 1.0 (51 points); r in
    # (0.5, 1.0] → 0 (50 points). AP = 51/101 ≈ 0.50495.
    dets = _dets("img", 0, [(10, 10, 20, 20, 0.9)])
    gts = _gts("img", [(10, 10, 20, 20), (100, 100, 110, 110)])
    map50, map5095, rows = compute_map(dets, gts, num_classes=1)
    assert map50 == pytest.approx(51.0 / 101.0, abs=1e-3)
    # mAP50-95: at IoU 0.5 the same AP (0.505); at higher IoU thresholds the
    # detection still matches the one GT at IoU 1.0 (>= all thresholds), so
    # recall stays 0.5 → AP is 51/101 at every threshold → mAP50-95 == map50.
    assert map5095 == pytest.approx(51.0 / 101.0, abs=1e-3)


def test_compute_map_fp_after_tp_keeps_ap_one():
    # 1 GT, 2 detections: TP@conf0.9 (matches GT), then FP@conf0.5 (GT already
    # matched). The COCO monotonic envelope keeps AP=1.0 — a lower-conf FP at
    # recall=1.0 does not lower the max precision at any recall.
    dets = _dets("img", 0, [(10, 10, 20, 20, 0.9),   # TP
                            (12, 12, 22, 22, 0.5)])  # FP (GT already matched)
    gts = _gts("img", [(10, 10, 20, 20)])
    map50, _, _ = compute_map(dets, gts, num_classes=1)
    assert map50 == pytest.approx(1.0)


def test_compute_map_no_detections_for_gt_class():
    # Class with GT but zero detections → recall 0, AP 0 → counts as 0 in mean.
    gts = _gts("img", [(10, 10, 20, 20)])
    map50, map5095, _ = compute_map({}, gts, num_classes=1)
    assert map50 == pytest.approx(0.0)
    assert map5095 == pytest.approx(0.0)


def test_compute_map_multiple_classes_and_zero_gt_excluded():
    # 3 classes: class 0 has GT+dets (perfect → AP 1.0); class 1 has GT but no
    # dets (AP 0); class 2 has NO GT in the val set → excluded from the mean
    # (row None, not counted as 0).
    dets = {
        "img": [(0, 10, 10, 20, 20, 0.9)],   # class 0: TP
    }
    gts = {
        "img": np.array([
            [0, 10, 10, 20, 20],            # class 0 GT
            [1, 100, 100, 110, 110],         # class 1 GT (no detection)
        ], dtype=float),
    }
    map50, map5095, rows = compute_map(dets, gts, num_classes=3)
    # mean over classes-with-GT = classes 0 and 1 → (1.0 + 0.0) / 2 = 0.5.
    assert map50 == pytest.approx(0.5)
    assert map5095 == pytest.approx(0.5)
    by_cls = {r["class"]: r for r in rows}
    assert by_cls[0]["mAP50"] == pytest.approx(1.0)
    assert by_cls[1]["mAP50"] == pytest.approx(0.0)
    assert by_cls[2]["mAP50"] is None        # zero-GT class excluded


def test_compute_map_cross_image_matching():
    # Two images; a detection in img2 must not match a GT in img1.
    dets = {
        "img1": [],
        "img2": [(0, 10, 10, 20, 20, 0.9)],
    }
    gts = {
        "img1": np.array([[0, 10, 10, 20, 20]], dtype=float),  # GT in img1
        "img2": np.zeros((0, 5)),
    }
    # The img2 detection has no GT in img2 → FP; the img1 GT is missed → recall 0.
    map50, _, _ = compute_map(dets, gts, num_classes=1)
    assert map50 == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Import-purity is enforced by the existing
# test_pure_python_tests_dont_pull_ultralytics (tests/test_consistency.py),
# which now imports this module — so importing utils.map_eval must not pull
# ultralytics. No separate purge-and-reload test here: purging torch from
# sys.modules mid-session corrupts torch-type identity for downstream tests
# (test_postprocess), and is unnecessary given the shared invariant test.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GT loader — data-guarded (skips if data/images/val absent)
# ---------------------------------------------------------------------------
def test_load_ground_truth_aligned(data_dir):
    val_dir = data_dir / "images" / "val"
    if not val_dir.exists():
        pytest.skip("data/images/val not present (optional data set)")
    from utils.map_eval import load_ground_truth

    paths, gts, names = load_ground_truth(data_dir / "data.yaml")
    assert len(paths) == len(gts)            # every image gets an entry
    assert all(p.stem in gts for p in paths)
    assert len(names) == 12                  # COCO-subset 12 classes
    nonempty = [s for s, v in gts.items() if v.shape[0] > 0]
    assert nonempty, "expected at least one labeled val image"
    sample = gts[nonempty[0]]
    assert sample.shape[1] == 5              # [cls, x1, y1, x2, y2]
    assert (sample[:, 1:] >= 0).all()       # pixel coords non-negative
    assert (sample[:, 3] >= sample[:, 1]).all()  # x2 >= x1
    assert (sample[:, 4] >= sample[:, 2]).all()  # y2 >= y1
