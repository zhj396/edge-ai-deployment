"""Unit tests for the consistency tolerance auto-selector.

Pure logic only — ``cli/consistency_tol`` is intentionally free of any ``src`` import
so it can be exercised without loading the ultralytics stack. The selector picks strict
(1e-4 / 1e-3) for PT↔FP32 and loose (1e-3 / 1e-2) whenever a reduced-precision model
(INT8 or FP16) is on either side; an explicit pair overrides; passing only one of the
two is rejected.
"""
from pathlib import Path

import pytest

from cli.consistency_tol import (
    LOOSE_ATOL,
    LOOSE_RTOL,
    STRICT_ATOL,
    STRICT_RTOL,
    involves_fp16,
    involves_int8,
    needs_loose_pair,
    resolve_tolerances,
)


def test_involves_int8_detects_int8_stem():
    assert involves_int8(Path("models/yolov8s.pt"), Path("models/yolov8s_int8.onnx"))
    assert involves_int8(Path("models/yolov8s_int8.onnx"), Path("models/yolov8s_fp32.onnx"))


def test_involves_int8_false_for_pt_fp32():
    assert not involves_int8(Path("models/yolov8s.pt"), Path("models/yolov8s_fp32.onnx"))


def test_involves_fp16_detects_fp16_stem():
    assert involves_fp16(
        Path("models/yolov8s_fp32.onnx"), Path("models/yolov8s_fp16.engine")
    )
    assert involves_fp16(
        Path("models/yolov8s_fp16.engine"), Path("models/yolov8s_fp32.onnx")
    )


def test_involves_fp16_false_for_fp32_and_int8():
    # fp32 stems must not trip the fp16 matcher; int8 is a different precision lane.
    assert not involves_fp16(
        Path("models/yolov8s_fp32.onnx"), Path("models/yolov8s.pt")
    )
    assert not involves_fp16(
        Path("models/yolov8s_int8.onnx"), Path("models/yolov8s_fp32.onnx")
    )


def test_needs_loose_pair_or_of_int8_and_fp16():
    assert needs_loose_pair("yolov8s.pt", "yolov8s_int8.onnx")
    assert needs_loose_pair("yolov8s_fp32.onnx", "yolov8s_fp16.engine")
    assert needs_loose_pair("yolov8s_fp16.engine", "yolov8s_int8.engine")
    assert not needs_loose_pair("yolov8s.pt", "yolov8s_fp32.onnx")


def test_resolve_strict_for_pt_fp32():
    atol, rtol = resolve_tolerances(None, None, "yolov8s.pt", "yolov8s_fp32.onnx")
    assert (atol, rtol) == (STRICT_ATOL, STRICT_RTOL)


def test_resolve_loose_when_int8_involved():
    atol, rtol = resolve_tolerances(None, None, "yolov8s_fp32.onnx", "yolov8s_int8.onnx")
    assert (atol, rtol) == (LOOSE_ATOL, LOOSE_RTOL)


def test_resolve_loose_when_fp16_involved():
    # FP16 box coords diverge to single digits on a 0-640 scale — strict allclose
    # cannot pass; the loose pair keeps the advisory tensor stats informative while
    # the conf-cliff gate stays authoritative.
    atol, rtol = resolve_tolerances(
        None, None, "yolov8s_fp32.onnx", "tensorrt:models/yolov8s_fp16.engine"
    )
    assert (atol, rtol) == (LOOSE_ATOL, LOOSE_RTOL)


def test_resolve_explicit_overrides_auto():
    # Explicit beats auto even for the PT↔FP32 pair that would otherwise be strict.
    atol, rtol = resolve_tolerances(1e-2, 1e-1, "yolov8s.pt", "yolov8s_fp32.onnx")
    assert (atol, rtol) == (1e-2, 1e-1)


def test_resolve_rejects_only_one_tolerance():
    # The whole point of the cliff fix is that atol AND rtol move together; mixing a strict
    # atol with a loose rtol would reintroduce the false-pass. Reject it loudly.
    with pytest.raises(SystemExit, match="both --atol and --rtol"):
        resolve_tolerances(1e-4, None, "yolov8s.pt", "yolov8s_fp32.onnx")
    with pytest.raises(SystemExit, match="both --atol and --rtol"):
        resolve_tolerances(None, 1e-3, "yolov8s.pt", "yolov8s_fp32.onnx")


# --- ort_cpp: backend prefix routing ----------------------------------------
# The C++ ORT backend is selected by an 'ort_cpp:' prefix on --model1/--model2.
# Tolerance selection is path-stem based, so the prefix must not break INT8
# detection: 'ort_cpp:.../yolov8s_int8.onnx' still selects the loose INT8 pair,
# and an all-FP32 ort_cpp run still gets strict.

def test_involves_int8_detects_int8_behind_ort_cpp_prefix():
    assert involves_int8("models/yolov8s_fp32.onnx", "ort_cpp:models/yolov8s_int8.onnx")
    assert involves_int8("ort_cpp:models/yolov8s_int8.onnx", "ort_cpp:models/yolov8s_fp32.onnx")


def test_involves_int8_false_for_ort_cpp_fp32_pair():
    assert not involves_int8(
        "models/yolov8s_fp32.onnx", "ort_cpp:models/yolov8s_fp32.onnx"
    )


def test_resolve_loose_when_ort_cpp_int8_involved():
    atol, rtol = resolve_tolerances(
        None, None, "ort_cpp:models/yolov8s_fp32.onnx",
        "ort_cpp:models/yolov8s_int8.onnx",
    )
    assert (atol, rtol) == (LOOSE_ATOL, LOOSE_RTOL)


def test_resolve_strict_when_ort_cpp_fp32_vs_fp32():
    atol, rtol = resolve_tolerances(
        None, None, "models/yolov8s_fp32.onnx",
        "ort_cpp:models/yolov8s_fp32.onnx",
    )
    assert (atol, rtol) == (STRICT_ATOL, STRICT_RTOL)
