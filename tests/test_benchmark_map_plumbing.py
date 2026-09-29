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
    # A backend with no Python engine must report (None, "unsupported ...")
    # and must not construct any engine on the way (the model path below
    # deliberately does not exist).
    b = Benchmark.__new__(Benchmark)
    result, source = b._compute_map_native(
        "ort_cpp", "models/does_not_exist.onnx"
    )
    assert result is None
    assert source in SUPPORTED_SOURCES
    assert source.startswith("unsupported backend")
