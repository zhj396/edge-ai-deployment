"""Unit tests for comparison helpers.

Imports ``utils.comparison`` (pure-NumPy) so the suite is independent of the
ultralytics stack. ``src.consistency`` re-exports the same names for callers
that already depend on them — see
``test_src_consistency_reexports_comparison_helpers`` below.
"""
import numpy as np
import pytest

from utils.comparison import (
    compare_detections,
    compare_tensors,
    compute_iou,
)


# ---------------------------------------------------------------------------
def test_compute_iou_identical():
    b = [0, 0, 10, 10]
    assert compute_iou(b, b) == pytest.approx(1.0)


def test_compute_iou_no_overlap():
    assert compute_iou([0, 0, 5, 5], [10, 10, 15, 15]) == 0.0


def test_compute_iou_partial():
    # 5x5 boxes shifted by (3, 3): intersection is 2x2 = 4.
    # union = 25 + 25 - 4 = 46 -> IoU = 4/46
    iou = compute_iou([0, 0, 5, 5], [3, 3, 8, 8])
    assert iou == pytest.approx(4 / 46)


# ---------------------------------------------------------------------------
def test_compare_tensors_identical_passes():
    a = np.random.rand(2, 4).astype(np.float32)
    stats = compare_tensors(a, a.copy(), mode="tensor")
    assert stats["passed"] is True


def test_compare_tensors_shape_mismatch():
    a = np.zeros((1, 4))
    b = np.zeros((1, 5))
    stats = compare_tensors(a, b, mode="tensor")
    assert stats["passed"] is False
    assert "shape mismatch" in stats["error_msg"]


def test_compare_tensors_nan_detected():
    a = np.array([1.0, np.nan, 3.0])
    stats = compare_tensors(a, a.copy(), mode="tensor")
    assert stats["passed"] is False
    assert "NaN" in stats["error_msg"]


def test_compare_tensors_detection_mode_relaxed():
    a = np.array([1.0, 2.0, 3.0])
    b = a + 0.001  # tiny diff
    stats = compare_tensors(a, b, mode="detection")
    assert stats["passed"] is True
    assert stats["cosine_similarity"] > 0.999


# ---------------------------------------------------------------------------
def test_compare_detections_empty_match():
    stats = compare_detections([], [])
    assert stats["passed"] is True
    assert stats["mean_iou"] == 1.0


def test_compare_detections_identical_match():
    dets = [(10, 10, 50, 50, 0.9, 0)]
    stats = compare_detections(dets, dets)
    assert stats["passed"] is True
    assert stats["mean_iou"] == pytest.approx(1.0)
    assert stats["class_match_rate"] == 1.0


def test_compare_detections_no_match_fails():
    dets1 = [(0, 0, 10, 10, 0.9, 0)]
    dets2 = [(100, 100, 110, 110, 0.9, 0)]
    stats = compare_detections(dets1, dets2, iou_threshold=0.5)
    assert stats["passed"] is False
    assert stats["matched"] == 0


def test_compare_detections_count_diff_penalized():
    dets1 = [(0, 0, 10, 10, 0.9, 0)]
    dets2 = [(0, 0, 10, 10, 0.9, 0)] * 5
    stats = compare_detections(dets1, dets2, count_diff_thresh=1)
    assert stats["count_diff"] == 4


# ---------------------------------------------------------------------------
# Backward-compat: src.consistency must re-export the pure helpers so any
# downstream caller that wrote ``from src.consistency import compare_tensors``
# keeps working. The actual implementation now lives in utils.comparison.
# ---------------------------------------------------------------------------
def test_src_consistency_reexports_comparison_helpers():
    from utils import comparison as uc
    from src import consistency as sc

    for name in (
        "cosine_similarity",
        "compare_tensors",
        "compute_iou",
        "compare_detections",
    ):
        assert getattr(sc, name) is getattr(uc, name), name


def test_pure_python_tests_dont_pull_ultralytics():
    """Importing the pure-Python test modules must not import ultralytics.

    The comparison helpers live in utils.comparison (pure NumPy); the only
    test-time import of ultralytics comes from src.postprocess (which needs
    non_max_suppression + scale_boxes). The test_consistency/test_quantize/
    test_utils modules themselves must stay ultralytics-free.
    """
    import sys

    # Pre-clean so prior imports don't pollute the assertion.
    for k in list(sys.modules):
        if k.startswith("test_") or k.startswith("ultralytics"):
            sys.modules.pop(k, None)

    sys.path.insert(0, "tests")
    import test_consistency   # noqa: F401
    import test_quantize      # noqa: F401
    import test_utils         # noqa: F401

    assert "ultralytics" not in sys.modules, (
        "pure-Python test modules must not import ultralytics; "
        f"loaded: {[k for k in sys.modules if 'ultralytics' in k][:5]}"
    )


def test_consistency_parser_exposes_report_path():
    """--report-path gives tensor and detection runs separate JSON reports —
    the shared default path means the second of two back-to-back runs
    replaces the first one's report."""
    import argparse
    from pathlib import Path

    from cli.consistency import add_parser

    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers(dest="command"))

    args = parser.parse_args(["consistency"])
    assert args.report_path == Path("results/consistency_report.json")

    args = parser.parse_args(["consistency", "--report-path", "results/tensor.json"])
    assert args.report_path == Path("results/tensor.json")
