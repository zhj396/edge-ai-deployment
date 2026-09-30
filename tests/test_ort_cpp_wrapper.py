"""Model-free tests for the ``ort_cpp`` backend's Python glue.

The ORT-C++ backend (the built ``ort_cpp`` exe) is wired into ``benchmark`` +
``consistency`` via a thin subprocess wrapper. These tests guard the *contract*
the harnesses rely on, without needing the built exe or a staged model:

* ``resolve_ort_cpp_exe`` honors the ``ORT_CPP_PATH`` env override and raises a
  "build it first" ``RuntimeError`` (pointing at ``cmake --build cpp/build``)
  when the exe is absent;
* ``cli.consistency._resolve_consistency_model`` passes an ``ort_cpp:`` spec
  through verbatim so ``ModelWrapper`` can parse the prefix;
* ``ModelWrapper`` detects the ``ort_cpp:`` prefix **lazily** — no in-process
  ORT session is loaded and the exe is not resolved at construction time, so
  importing / constructing without a built exe (e.g. CI, the test suite) works;
  only an actual ``forward`` call resolves the exe (and may raise
  the "build it first" error).

Heavy: imports ``src.consistency`` + ``src.benchmark`` (→ ultralytics), like
``test_openvino_optional_imports``. No ``.pt`` / ``.onnx`` / exe / GPU required.
"""
from __future__ import annotations

import pytest

from src.benchmark import resolve_ort_cpp_exe
from src.consistency import ModelWrapper
from cli.consistency import _resolve_consistency_model


# --- resolve_ort_cpp_exe ---------------------------------------------------
def test_resolve_ort_cpp_exe_env_override(tmp_path, monkeypatch):
    # ORT_CPP_PATH (an existing file) wins over the build-candidate lookup.
    fake = tmp_path / "fake_ort_cpp"
    fake.touch()
    monkeypatch.setenv("ORT_CPP_PATH", str(fake))
    assert resolve_ort_cpp_exe() == str(fake)


def test_resolve_ort_cpp_exe_missing_raises(monkeypatch):
    # Deterministic on machines where the exe happens to be built: drop the env
    # override AND force the build-candidate .is_file() check to fail.
    monkeypatch.delenv("ORT_CPP_PATH", raising=False)
    import pathlib
    monkeypatch.setattr(pathlib.Path, "is_file", lambda self: False)

    with pytest.raises(RuntimeError) as exc:
        resolve_ort_cpp_exe()
    msg = str(exc.value)
    assert "cmake --build" in msg
    assert "ORT_CPP_PATH" in msg


# --- _resolve_consistency_model -------------------------------------------
def test_resolve_consistency_model_passthrough_ort_cpp():
    spec = "ort_cpp:models/yolov8s_fp32.onnx"
    # The 'ort_cpp:' prefix is a backend selector, not a path — it must survive
    # resolve_path_arg verbatim so ModelWrapper can parse it.
    assert _resolve_consistency_model(spec, "models/yolov8s.pt") == spec


def test_resolve_consistency_model_plain_path_passthrough(tmp_path):
    # A supplied plain (non-prefixed) path is returned as-is — resolve_path_arg
    # only existing-validates the *default*, never a user-supplied value.
    p = tmp_path / "m.onnx"
    p.touch()
    assert _resolve_consistency_model(str(p), "models/yolov8s.pt") == str(p)


# --- ModelWrapper prefix detection ----------------------------------------
def test_model_wrapper_ort_cpp_prefix_is_lazy():
    w = ModelWrapper("ort_cpp:models/yolov8s_fp32.onnx", "cpu")
    assert w.type == "ort_cpp"
    assert w.model_path == "models/yolov8s_fp32.onnx"
    # Lazy by design: no in-process session, and the exe is NOT resolved at
    # construction — so this works without a built exe.
    assert w.model is None
    assert w._exe is None


def test_model_wrapper_ort_cpp_device_cuda_coercion_off():
    # device='cuda' coerces to 'cpu' when CUDA is unavailable (the common CI /
    # laptop case) — the C++ path is CPU-only regardless.
    w = ModelWrapper("ort_cpp:models/yolov8s_fp32.onnx", "cuda")
    assert w.device in ("cpu", "cuda")
