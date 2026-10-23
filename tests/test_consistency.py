"""Unit tests for comparison helpers.

The comparison-helper tests import ``utils.comparison`` (pure NumPy) and run
without the ultralytics stack. The remaining tests need the heavy stack and
skip when it is absent: ``test_src_consistency_reexports_comparison_helpers``
(``src.consistency`` imports ultralytics at module level),
``test_consistency_parser_exposes_report_path`` (``cli.consistency`` resolves
``validate_consistency`` through the same lazy import), and the OpenVINO
dispatch tests (they import ``src.consistency`` for ``ModelWrapper``).
"""
import numpy as np
import pytest

from utils.comparison import (
    compare_detections,
    compare_tensors,
    compute_iou,
    REDUCED_PRECISION_MARKERS,
    select_tolerance_pair,
    TOLERANCE_PAIRS,
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
# src.consistency re-exports the pure helpers from utils.comparison; the
# re-export must stay identity (same objects).
# ---------------------------------------------------------------------------
def test_src_consistency_reexports_comparison_helpers():
    # src.consistency pulls ultralytics / torch / onnxruntime at module level;
    # skip gracefully when the heavy stack is absent (same opt-in as
    # test_cli_output_paths._heavy_cli).
    pytest.importorskip("ultralytics")
    from utils import comparison as uc
    from src import consistency as sc

    for name in (
        "cosine_similarity",
        "compare_tensors",
        "compute_iou",
        "compare_detections",
        "select_tolerance_pair",
        "TOLERANCE_PAIRS",
        "REDUCED_PRECISION_MARKERS",
    ):
        assert getattr(sc, name) is getattr(uc, name), name


# ---------------------------------------------------------------------------
# Tolerance auto-selection (select_tolerance_pair)
# ---------------------------------------------------------------------------
def test_select_tolerance_pair_strict_pt_vs_fp32_onnx():
    """The README default pair (.pt vs *_fp32.onnx) auto-selects strict."""
    assert select_tolerance_pair(
        "models/yolov8s.pt", "models/yolov8s_fp32.onnx"
    ) == ("strict", 1e-4, 1e-3)


def test_select_tolerance_pair_strict_swapped_order():
    """Order-independent: swapping model1/model2 cannot change the pair."""
    a = select_tolerance_pair("models/yolov8s.pt", "models/yolov8s_fp32.onnx")
    b = select_tolerance_pair("models/yolov8s_fp32.onnx", "models/yolov8s.pt")
    assert a == b == ("strict", 1e-4, 1e-3)


def test_select_tolerance_pair_strict_pth_and_plain_onnx():
    """.pth aliases .pt (ModelWrapper's suffix branch); a marker-free .onnx
    pairs strict — the 'full-precision unless marked' default."""
    assert select_tolerance_pair(
        "models/yolov8s.pth", "models/yolov8s.onnx"
    ) == ("strict", 1e-4, 1e-3)


def test_select_tolerance_pair_loose_int8_stem():
    assert select_tolerance_pair(
        "models/yolov8s.pt", "models/yolov8s_int8.onnx"
    ) == ("loose", 1e-3, 1e-2)


def test_select_tolerance_pair_loose_fp16_stem():
    assert select_tolerance_pair(
        "models/yolov8s_fp32.onnx", "models/yolov8s_fp16.onnx"
    ) == ("loose", 1e-3, 1e-2)


def test_select_tolerance_pair_loose_openvino_int8_prefix():
    """The openvino_int8: prefix carries the int8 marker (plus a .xml
    suffix, which the allowlist rejects independently)."""
    assert select_tolerance_pair(
        "openvino_int8:models/yolov8s_openvino_int8.xml", "models/yolov8s.pt"
    ) == ("loose", 1e-3, 1e-2)


def test_select_tolerance_pair_loose_plain_openvino_prefix():
    """A plain openvino: IR is never strict: convert defaults to FP16, so
    .xml is not in the full-precision suffix allowlist."""
    assert select_tolerance_pair(
        "openvino:models/yolov8s_openvino.xml", "models/yolov8s.pt"
    ) == ("loose", 1e-3, 1e-2)


def test_select_tolerance_pair_prefix_stripping_ort_cpp():
    """ort_cpp: is a transparent backend selector — the stripped .onnx stem
    classifies as before."""
    assert select_tolerance_pair(
        "ort_cpp:models/yolov8s_fp32.onnx", "models/yolov8s.pt"
    ) == ("strict", 1e-4, 1e-3)


def test_select_tolerance_pair_loose_tensorrt_engine():
    """.engine is not in the full-precision allowlist (and the fp16 stem
    marker would trip regardless)."""
    assert select_tolerance_pair(
        "tensorrt:models/yolov8s_fp16.engine", "models/yolov8s_fp32.onnx"
    ) == ("loose", 1e-3, 1e-2)


def test_select_tolerance_pair_marker_and_prefix_tuples_pinned():
    """The classification vocabulary is a pinned contract: adding a marker
    (e.g. 'int8' -> also 'int16'?) or dropping a suffix changes verdicts on
    existing model pairs — do it deliberately, here."""
    assert TOLERANCE_PAIRS == {"strict": (1e-4, 1e-3), "loose": (1e-3, 1e-2)}
    assert REDUCED_PRECISION_MARKERS == (
        "int8", "int4", "fp16", "f16", "half", "quant",
    )


def test_consistency_parser_tolerance_defaults_are_none():
    """--atol/--rtol default to None (deferred to the auto-selector in
    run()); an explicit flag overrides its half only."""
    pytest.importorskip("ultralytics")
    import argparse

    from cli.consistency import add_parser

    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers(dest="command"))

    args = parser.parse_args(["consistency"])
    assert args.atol is None and args.rtol is None

    args = parser.parse_args(["consistency", "--atol", "1e-4"])
    assert args.atol == 1e-4 and args.rtol is None

    args = parser.parse_args(["consistency", "--rtol", "1e-2"])
    assert args.atol is None and args.rtol == 1e-2


def test_consistency_report_records_tolerances():
    """validate_consistency records the resolved pair (and its source, when
    the CLI passes one) under ``tolerances`` in the final report."""
    pytest.importorskip("ultralytics")
    import inspect

    from src import consistency

    src = inspect.getsource(consistency.validate_consistency)
    assert '"tolerances"' in src
    assert "tolerance_source" in src


def test_pure_python_tests_dont_pull_ultralytics():
    """Importing the pure-Python test modules must not import ultralytics.

    The comparison helpers live in utils.comparison (pure NumPy). Within
    these four modules the only ultralytics pull would come transitively
    through src imports (e.g. src.postprocess, which needs
    non_max_suppression + scale_boxes) — the modules themselves must stay
    ultralytics-free.
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
    # cli.consistency resolves validate_consistency through the lazy src
    # import, so this test needs the same heavy stack as the re-export test.
    pytest.importorskip("ultralytics")
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
# OpenVINO prefix dispatch (ModelWrapper)
# ---------------------------------------------------------------------------
def test_openvino_prefix_strips_head_not_tail():
    """``spec[len(prefix):]`` removes exactly ``len(prefix)`` characters
    from the head; the path text stays intact."""
    pytest.importorskip("ultralytics")
    from src.consistency import ModelWrapper

    spec = "openvino:models/yolov8s_openvino.xml"
    prefix = "openvino:"
    assert spec[len(prefix):] == "models/yolov8s_openvino.xml"
    # Pin the dispatch constant's contents: each prefix is an exact string
    # form matched via startswith, so an accidental edit surfaces here.
    assert ModelWrapper.OPENVINO_PREFIXES == ("openvino_int8:", "openvino:")


def test_openvino_missing_xml_fails_fast_with_stripped_path(tmp_path):
    """A missing IR raises FileNotFoundError naming the real path, before
    any OpenVINOEngine import is attempted."""
    pytest.importorskip("ultralytics")
    from src.consistency import ModelWrapper

    missing = tmp_path / "no_such_ir.xml"
    with pytest.raises(FileNotFoundError) as ei:
        ModelWrapper(f"openvino:{missing}", "cpu")
    assert str(missing) in str(ei.value)
    assert "openvino:" not in str(ei.value)


def test_openvino_non_xml_suffix_rejected(tmp_path):
    """``openvino:`` + a non-.xml suffix raises PathValidationError naming
    the actual path, ahead of the lazy engine import."""
    pytest.importorskip("ultralytics")
    from src.consistency import ModelWrapper
    from utils import PathValidationError

    # A stub file satisfies the exists() check; the suffix check fires
    # before any model content is read.
    fake = tmp_path / "fake.onnx"
    fake.write_bytes(b"\x00" * 8)
    with pytest.raises(PathValidationError):
        ModelWrapper(f"openvino:{fake}", "cpu")


def test_openvino_forward_dispatch_uses_host_numpy():
    """``ModelWrapper.forward`` on an openvino-type wrapper copies the torch
    tensor to a host float32 numpy batch before calling ``engine._forward``."""
    pytest.importorskip("torch")
    import numpy as np
    import torch

    from src.consistency import ModelWrapper

    mw = ModelWrapper.__new__(ModelWrapper)
    mw.type = "openvino"

    class _StubEngine:
        def _forward(self, batch):
            assert isinstance(batch, np.ndarray) and batch.dtype == np.float32
            assert batch.ndim == 4
            return np.zeros((batch.shape[0], 16, 8400), dtype=np.float32)

    mw.model = _StubEngine()
    out = ModelWrapper.forward(mw, torch.rand(2, 3, 64, 64))
    assert out.shape == (2, 16, 8400)


def test_openvino_bare_prefix_keeps_empty_stripped_path():
    """A bare "openvino:" (empty path) keeps the empty stripped path —
    the prefixed spec must not resurface in self.path / error messages."""
    pytest.importorskip("ultralytics")
    from src.consistency import ModelWrapper
    from utils import PathValidationError

    # Path("") resolves to the cwd (exists() is True), so the empty path
    # surfaces through the .xml suffix check as a PathValidationError.
    with pytest.raises((PathValidationError, FileNotFoundError)) as ei:
        ModelWrapper("openvino:", "cpu")
    assert "openvino:" not in str(ei.value)


def test_consistency_report_includes_openvino_device_provenance():
    """``validate_consistency`` wires an OpenVINO side's requested/actual
    device pair into the final report under ``openvino_devices``."""
    pytest.importorskip("ultralytics")
    import inspect

    from src import consistency

    src = inspect.getsource(consistency.validate_consistency)
    assert "openvino_provenance" in src
    assert '"openvino_devices"' in src


def test_consistency_plain_openvino_suffix_rejected(tmp_path):
    """A bare .xml (no openvino: prefix) routes to the PT/ONNX branch — the
    suffix gate rejects it there, pointing at the openvino: prefixes."""
    pytest.importorskip("ultralytics")
    from src.consistency import ModelWrapper
    from utils import PathValidationError

    fake = tmp_path / "ir.xml"
    fake.write_text("<net/>")
    with pytest.raises(PathValidationError, match="openvino:"):
        ModelWrapper(str(fake), "cpu")


# ---------------------------------------------------------------------------
# conf-cliff gate and dual verdict (utils/comparison.py, tensor mode)


def _yolo_layout(bs: int, nc: int, n: int, cls_scores) -> np.ndarray:
    """Build a (bs, 4+nc, N) tensor: box coords ~0..640, class scores given."""
    out = np.zeros((bs, 4 + nc, n), dtype=np.float32)
    out[:, :4, :] = np.linspace(0, 640, n, dtype=np.float32)
    out[:, 4:, :] = np.asarray(cls_scores, dtype=np.float32).reshape(bs, nc, n)
    return out


def test_compare_tensors_cliff_gate_clean_when_detections_stable():
    """A small class-score divergence that flips no boxes across the cliff
    leaves the cliff gate clean."""
    # 8 boxes with score just above conf=0.25 on both sides (no flips).
    scores = np.full((1, 2, 8), 0.30)
    out1 = _yolo_layout(1, 2, 8, scores)
    out2 = out1.copy()
    out2[:, 4:, :] += 1e-4  # FP32-noise-level; 0.30±1e-4 stays above 0.25
    stats = compare_tensors(out1, out2, mode="tensor", nc=2,
                            conf_cliff=0.25, atol=1e-3, rtol=1e-3)
    assert stats["allclose_passed"] is True
    assert stats["detection_consistent"] is True
    assert stats["passed"] is True
    assert stats["above_cliff_1"] == stats["above_cliff_2"] == 8


def test_compare_tensors_cliff_gate_fails_on_spurious_flip():
    """A divergence within allclose tolerance that flips many boxes across the
    cliff FAILS tensor mode even though allclose passes."""
    scores = np.full((1, 2, 8), 0.249)  # just below the cliff on side 1
    out1 = _yolo_layout(1, 2, 8, scores)
    out2 = out1.copy()
    out2[:, 4:, :] += 0.003  # 0.252 — above the cliff; within atol=1e-2
    stats = compare_tensors(out1, out2, mode="tensor", nc=2,
                            conf_cliff=0.25, atol=1e-2, rtol=1e-2,
                            cliff_count_thresh=1.0)
    assert stats["allclose_passed"] is True
    assert stats["detection_consistent"] is False
    assert stats["passed"] is False
    assert "conf-cliff" in stats["error_msg"]


def test_compare_tensors_dual_verdict_reduced_precision_pair():
    """FP16-style divergence: allclose cannot pass at a tolerance that still
    catches the regression, yet the cliff gate stays clean — both verdicts
    carried separately."""
    out1 = _yolo_layout(1, 2, 8, np.full((1, 2, 8), 0.30))
    out2 = out1.copy()
    out2[:, :4, :] += 2.0   # box coords diverge by px-scale (FP16 head noise)
    out2[:, 4:, :] += 1e-3  # class scores still above the 0.25 cliff
    stats = compare_tensors(out1, out2, mode="tensor", nc=2,
                            conf_cliff=0.25, atol=1e-3, rtol=1e-3)
    assert stats["allclose_passed"] is False
    assert stats["detection_consistent"] is True
    assert stats["passed"] is False


def test_compare_tensors_cliff_gate_skipped_without_inferable_nc():
    """A non-3D layout (no inferable nc, no explicit nc) skips the cliff
    gate entirely — no above_cliff stats, no detection_consistent field."""
    # Deterministic input (repo rule: unit tests stay deterministic)
    # spanning a wide magnitude range (0-700).
    a = np.linspace(0.0, 700.0, 320, dtype=np.float32).reshape(8, 40)
    b = a + 1e-4
    stats = compare_tensors(a, b, mode="tensor", conf_cliff=0.25,
                            atol=1e-3, rtol=1e-3)
    assert "above_cliff_1" not in stats
    assert "detection_consistent" not in stats
    assert stats["passed"] is True
