"""Tests for the benchmark mAP plumbing (src/benchmark.py).

The benchmark's --validation path needs a model and dataset to execute, so
these tests exercise the plumbing only: ``run_all``'s dispatch target exists,
and the unsupported-backend path returns without constructing any engine.
No model, no dataset, no GPU. ``Benchmark.__new__`` skips the ``__init__``
that requires data.yaml + the calibration sampler, and the unsupported
backend path returns before any engine is constructed (so a nonexistent
model path cannot make it fail).
"""
from __future__ import annotations

from src.benchmark import Benchmark

# The vocabulary run_all writes into the summary CSV's mAP_source column.
SUPPORTED_SOURCES = {
    "native evaluator",
    "unsupported backend (no Python engine to drive)",
    "failed: native evaluator crashed (see log)",
    "no GT: every class has zero ground truth in the val set",
    "disabled (--validation off)",
}


def test_compute_map_native_exists():
    # run_all dispatches to this exact name; renaming it without updating
    # the call site crashes every --validation run with AttributeError.
    assert callable(getattr(Benchmark, "_compute_map_native", None))


def test_compute_map_native_unsupported_backend():
    # A backend with no Python engine reports (None, "unsupported ...")
    # without constructing any engine (the model path below does not exist).
    b = Benchmark.__new__(Benchmark)
    result, source = b._compute_map_native(
        "ort_cpp", "models/does_not_exist.onnx"
    )
    assert result is None
    assert source in SUPPORTED_SOURCES
    assert source.startswith("unsupported backend")


def test_run_all_continues_past_failed_backend(monkeypatch):
    """A backend whose run raises is logged and omitted; the loop continues
    and the remaining backends land in the summary — the same
    omit-and-summarize contract as OpenVINOEngine.infer's failed batches."""
    b = Benchmark.__new__(Benchmark)
    b.validation = False
    b.timestamp_suffix = "_test"  # summary_csv_path interpolates this

    # summary_csv_path points at results/ — stub the CSV write so the test
    # stays hermetic (the DataFrame under test is the return value below).
    written = {}
    import pandas as pd

    def _fake_to_csv(self, path, **kwargs):
        written["path"] = str(path)

    monkeypatch.setattr(pd.DataFrame, "to_csv", _fake_to_csv)

    calls = []

    def _ok(backend, model_path):
        calls.append(backend)
        return {
            "device": "cpu", "batch_size": 1, "mean_total_s": 1.0,
            "mean_batch_s": 0.5, "mean_image_s": 0.5, "fps": 2.0,
            "latency_ms_mean": 500.0,
        }

    def _boom(backend, model_path):
        raise RuntimeError("openvino is not installed")

    monkeypatch.setattr(b, "_run_pytorch", _ok)
    monkeypatch.setattr(b, "_run_openvino", _boom)
    monkeypatch.setattr(b, "_run_onnx", _ok)

    df = b.run_all({
        "pytorch": "models/yolov8s.pt",
        "openvino": "models/yolov8s_openvino.xml",
        "onnx_fp32": "models/yolov8s_fp32.onnx",
    })

    assert calls == ["pytorch", "onnx_fp32"]
    assert list(df["Backend"]) == ["pytorch", "onnx_fp32"]
