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
    import test_map_eval      # noqa: F401  (utils.map_eval — pure-NumPy mAP)

    assert "ultralytics" not in sys.modules, (
        "pure-Python test modules must not import ultralytics; "
        f"loaded: {[k for k in sys.modules if 'ultralytics' in k][:5]}"
    )


def test_consistency_parser_exposes_report_path():
    """--report-path selects the report stem + directory; run() inserts a
    per-run <TS> between stem and suffix (consistency_report.json ->
    consistency_report_<TS>.json), so each run writes its own report and
    distinct stems keep tensor / detection reports side by side."""
    import argparse
    from pathlib import Path

    from cli.consistency import add_parser

    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers(dest="command"))

    args = parser.parse_args(["consistency"])
    assert args.report_path == Path("results/consistency_report.json")

    args = parser.parse_args(["consistency", "--report-path", "results/tensor.json"])
    assert args.report_path == Path("results/tensor.json")


# ---------------------------------------------------------------------------
# OpenVINO backend-selector prefixes (consistency-vs-IR support)
# ---------------------------------------------------------------------------
def test_model_wrapper_openvino_prefix_parses_lazily(tmp_path):
    """A prefixed wrapper constructs without loading the IR.

    Compilation is deferred to the first forward so ModelWrapper stays
    constructible on boxes without the openvino package; only the .xml path
    itself is checked up front.
    """
    from src.consistency import ModelWrapper

    xml = tmp_path / "yolov8s_openvino.xml"
    xml.touch()

    for prefix in ("openvino:", "openvino_int8:"):
        w = ModelWrapper(f"{prefix}{xml}", "cpu")
        assert w.type == "openvino"
        assert w.model is None           # not compiled yet — lazy
        assert w.model_path == str(xml)  # prefix stripped


def test_model_wrapper_openvino_missing_ir_fails_fast():
    """A typo'd IR path fails at construction with FileNotFoundError."""
    from src.consistency import ModelWrapper

    with pytest.raises(FileNotFoundError):
        ModelWrapper("openvino:/no/such/model.xml", "cpu")


def test_model_wrapper_openvino_forward_requires_openvino(monkeypatch, tmp_path):
    """Forward without openvino installed raises the install-hint ImportError
    (same contract as benchmark._run_openvino)."""
    import torch

    xml = tmp_path / "yolov8s_openvino.xml"
    xml.touch()

    monkeypatch.setattr("src.openvino_available", lambda: False)
    from src.consistency import ModelWrapper

    w = ModelWrapper(f"openvino:{xml}", "cpu")
    with pytest.raises(ImportError, match="requirements-openvino"):
        w.forward(torch.zeros(1, 3, 8, 8))


def test_resolve_consistency_model_prefix_passthrough(tmp_path):
    """Prefixed specs pass through verbatim (ModelWrapper parses them);
    plain paths and defaults resolve like any other path arg."""
    from pathlib import Path

    from cli.consistency import _resolve_consistency_model

    real = tmp_path / "yolov8s.pt"
    real.touch()

    spec = "openvino_int8:models/yolov8s_openvino_int8.xml"
    assert _resolve_consistency_model(spec, str(real)) == spec
    assert _resolve_consistency_model(str(real), str(real)) == str(real)
    assert Path(_resolve_consistency_model(None, str(real))) == real


def test_model_wrapper_prepare_validates_imgsz_list(monkeypatch, tmp_path):
    """prepare() compiles the IR before the measurement loop and rejects an
    img-sizes list the IR cannot serve with a run-level ValueError (the
    measurement loop itself only reports batch-level failures)."""
    xml = tmp_path / "yolov8s_openvino.xml"
    xml.touch()

    monkeypatch.setattr("src.openvino_available", lambda: True)

    class _StubEngine:
        """Stands in for OpenVINOEngine: construction records the imgsz;
        spatial_size reports the IR's conversion-pinned size."""

        def __init__(self, model_path, device, imgsz):
            self.imgsz = imgsz
            self.spatial_size = 640

    monkeypatch.setattr("src.OpenVINOEngine", _StubEngine)

    from src.consistency import ModelWrapper

    w = ModelWrapper(f"openvino:{xml}", "cpu")
    w.prepare([640])  # every requested size servable — compiles, no error
    assert isinstance(w.model, _StubEngine)

    # The engine rejects a mismatched FIRST size at construction; prepare's
    # own check covers a mismatched LATER size in the (already compiled) list.
    with pytest.raises(ValueError, match="imgsz=640"):
        w.prepare([640, 480])


def test_model_wrapper_prepare_skips_size_check_for_dynamic_ir(
    monkeypatch, tmp_path,
):
    """A dynamically-shaped IR accepts any img size — prepare must not
    reject it (only conversion-pinned static spatial dims are validated)."""
    xml = tmp_path / "yolov8s_openvino.xml"
    xml.touch()

    monkeypatch.setattr("src.openvino_available", lambda: True)

    class _StubEngine:
        def __init__(self, model_path, device, imgsz):
            self.spatial_size = None  # dynamic spatial dims

    monkeypatch.setattr("src.OpenVINOEngine", _StubEngine)

    from src.consistency import ModelWrapper

    w = ModelWrapper(f"openvino:{xml}", "cpu")
    w.prepare([480, 640])  # no ValueError — dynamic IR serves any size
    assert w.model is not None


def test_model_wrapper_bare_ir_path_hints_at_prefix(tmp_path):
    """A bare .xml path (missing the openvino: backend selector) fails
    with the prefix hint, not an opaque ORT InvalidProtobuf."""
    from src.consistency import ModelWrapper

    xml = tmp_path / "yolov8s_openvino.xml"
    xml.touch()

    with pytest.raises(ValueError, match="openvino:"):
        ModelWrapper(str(xml), "cpu")
