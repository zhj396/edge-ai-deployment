"""Model-free tests for the ``trt_cpp`` backend's Python glue.

The TensorRT-C++ backend (the built ``_trt_cpp`` pybind11 module) is wired into
``benchmark`` + ``consistency`` in-process (NOT a subprocess — unlike
``ort_cpp``). It mirrors the Python ``tensorrt:`` path exactly (native mAP +
``kernel_latency_ms``, apples-to-apples) so the ``trt_cpp*`` vs ``tensorrt*``
comparison isolates the C++-binding vs cuda-python overhead. These tests guard
the *contract* the harnesses rely on, without needing the built module, a
staged ``.engine``, or a GPU:

* the new public names resolve through ``src``'s PEP-562 ``_LAZY`` map (the
  original OpenVINO/TRT wiring bug: unregistered names raised ``AttributeError``
  at import time);
* ``resolve_trt_cpp_module`` honors the ``TRT_CPP_PATH`` env override and raises
  a helpful "build it first" ``RuntimeError`` (pointing at the cmake build
  command) when the module is absent;
* ``cli.consistency._resolve_consistency_model`` passes a ``trt_cpp:`` spec
  through verbatim (does not mangle it via ``resolve_path_arg``);
* ``ModelWrapper`` detects the ``trt_cpp:`` prefix **lazily** — no module is
  loaded at construction, so importing / constructing without the built module
  (CI, the test suite) does not blow up; only an actual ``forward`` call
  resolves it (and may raise the "build it first" error);
* when the module is absent, constructing ``TensorRTEngineCpp`` raises
  ``ImportError`` with a helpful message rather than ``AttributeError`` /
  ``NameError`` (mirrors the tensorrt/openvino optional-dep posture).

Heavy: imports ``src.consistency`` + ``src.benchmark`` (-> ultralytics), like
``test_tensorrt_optional_imports`` / ``test_ort_cpp_wrapper``. No ``.pt`` /
``.onnx`` / ``.engine`` / built module / GPU required.
"""
import pytest

import src
from src.benchmark import resolve_trt_cpp_module
from src.consistency import ModelWrapper
from cli.consistency import _resolve_consistency_model


# ---------------------------------------------------------------------------
# _LAZY registration — the names must resolve through the package surface.
# ---------------------------------------------------------------------------
LAZY_NAMES = ["TensorRTEngineCpp", "trt_cpp_available"]


def test_public_names_resolve_through_lazy():
    """Each name is reachable via ``getattr(src, name)`` (PEP 562 ``_LAZY``)."""
    for name in LAZY_NAMES:
        assert hasattr(src, name), (
            f"{name!r} is not resolvable via src — missing from _LAZY?"
        )
        assert name in src._LAZY, f"{name!r} missing from src._LAZY map"


# ---------------------------------------------------------------------------
# resolve_trt_cpp_module — env override + helpful "build it first" error.
# (Mirrors test_resolve_ort_cpp_exe_* in test_ort_cpp_wrapper.py.)
# ---------------------------------------------------------------------------
def test_resolve_trt_cpp_module_env_override(tmp_path, monkeypatch):
    # TRT_CPP_PATH (an existing file) wins over the build-candidate glob. The
    # contract under test is the SEARCH ORDER (env first), not a successful
    # load of a fake file — so we stub _load_pybind_module to record the path
    # it was handed and return a sentinel. resolve_trt_cpp_module resolves the
    # name from module globals at call time, so patching bm._load_pybind_module
    # affects the call inside resolve_trt_cpp_module.
    fake = tmp_path / "fake_trt_cpp.so"
    fake.touch()
    monkeypatch.setenv("TRT_CPP_PATH", str(fake))
    import src.benchmark as bm
    seen = {}

    def fake_load(path, name):
        seen["path"] = path
        return "MOD"

    orig = bm._load_pybind_module
    bm._load_pybind_module = fake_load
    try:
        out = resolve_trt_cpp_module()
    finally:
        bm._load_pybind_module = orig
    assert out == "MOD"
    assert seen["path"] == str(fake)


def test_resolve_trt_cpp_module_missing_raises_helpful(monkeypatch, tmp_path):
    # Deterministic on machines where the module happens to be built: drop the
    # env override AND force the build-dir candidate check to fail (point
    # repo_root's cpp/build/tensorrt at an empty temp dir by monkeypatching
    # Path.is_dir on the build dir). Simpler: monkeypatch os.path.isfile (env)
    # and Path.is_dir (build dir) both to False.
    monkeypatch.delenv("TRT_CPP_PATH", raising=False)
    import os.path
    monkeypatch.setattr(os.path, "isfile", lambda p: False)
    import pathlib
    monkeypatch.setattr(pathlib.Path, "is_dir", lambda self: False)

    with pytest.raises(RuntimeError) as exc:
        resolve_trt_cpp_module()
    msg = str(exc.value)
    assert "cmake" in msg
    assert "TRT_CPP_PATH" in msg
    assert "BUILD_TRT_CPP" in msg


# ---------------------------------------------------------------------------
# cli.consistency prefix passthrough.
# ---------------------------------------------------------------------------
def test_resolve_consistency_model_passthrough_trt_cpp():
    spec = "trt_cpp:models/yolov8s_fp16.engine"
    # The 'trt_cpp:' prefix is a backend selector, not a path — it must survive
    # resolve_path_arg verbatim so ModelWrapper can parse it.
    assert _resolve_consistency_model(spec, "models/yolov8s.pt") == spec


def test_resolve_consistency_model_plain_path_passthrough(tmp_path):
    # A supplied plain (non-trt_cpp:) path is returned as-is — resolve_path_arg
    # only existing-validates the *default*, never a user-supplied value.
    p = tmp_path / "m.engine"
    p.touch()
    assert _resolve_consistency_model(str(p), "models/yolov8s.pt") == str(p)


# ---------------------------------------------------------------------------
# ModelWrapper prefix detection — lazy (mirrors test_ort_cpp_wrapper + the
# tensorrt: lazy test).
# ---------------------------------------------------------------------------
def test_model_wrapper_trt_cpp_prefix_is_lazy():
    w = ModelWrapper("trt_cpp:models/yolov8s_fp16.engine", "cuda")
    assert w.type == "trt_cpp"
    assert w.model_path == "models/yolov8s_fp16.engine"
    # The whole point of the lazy design: no module is loaded at construction —
    # so this works without the _trt_cpp module built or the .engine staged.
    assert w.model is None


def test_model_wrapper_trt_cpp_numeric_device_selects_gpu():
    # A numeric --device (GPU id) selects that GPU for the TensorRTEngineCpp,
    # while self.device (the "cpu"/"cuda" mode the ORT reference + preprocess
    # use) stays coerced — so those paths are unchanged by a numeric --device.
    w_int = ModelWrapper("trt_cpp:models/yolov8s_fp16.engine", 1)
    assert w_int.type == "trt_cpp"
    assert w_int._trt_gpu_id == 1                       # int GPU id honoured
    assert w_int.device in ("cpu", "cuda")              # ORT-ref mode unchanged
    w_str = ModelWrapper("trt_cpp:models/yolov8s_fp16.engine", "1")
    assert w_str._trt_gpu_id == 1                       # numeric string honoured
    assert ModelWrapper("trt_cpp:m.engine", "cpu")._trt_gpu_id == 0
    assert ModelWrapper("trt_cpp:m.engine", "cuda")._trt_gpu_id == 0


# ---------------------------------------------------------------------------
# Optional-dependency contract: absent module -> ImportError, not crash.
# Skipped (not failed) when the module IS built, so the suite stays green on a
# box that happens to have it.
# ---------------------------------------------------------------------------
TRT_CPP_INSTALLED = src.trt_cpp_available()


@pytest.mark.skipif(TRT_CPP_INSTALLED, reason="trt_cpp module is built; "
                    "ImportError path cannot be exercised")
def test_engine_raises_importerror_without_module():
    TensorRTEngineCpp = src.TensorRTEngineCpp
    with pytest.raises(ImportError, match="trt_cpp"):
        TensorRTEngineCpp(model_path="missing.engine")


def test_trt_cpp_available_is_bool():
    # trt_cpp_available() must return a bool (True/False), never raise on a box
    # without the module — the lazy probe catches resolve_trt_cpp_module's
    # RuntimeError and caches False.
    assert isinstance(TRT_CPP_INSTALLED, bool)
