"""Model-free tests for the TensorRT backend's optional-dependency surface.

tensorrt + cuda-python are *optional* — the base pin set (requirements-cpu.txt
/ requirements-kaggle.txt) does not include them. These tests guard the contract the rest of
the toolchain relies on without needing a GPU or the TensorRT wheels:

* the public names resolve through ``src``'s PEP-562 ``_LAZY`` map (an
  unregistered name breaks the import with ``ImportError``);
* the ``cli.tensorrt`` subcommand module imports cleanly and exposes
  ``add_parser`` / ``run`` (it does ``from src import ...`` at module top,
  so a broken ``_LAZY`` entry would surface here);
* the ``TensorRT*Config`` dataclasses construct;
* when tensorrt is absent, constructing ``TensorRTEngine`` and calling the
  build / export entry points raise ``ImportError`` with a helpful message,
  never ``AttributeError`` / ``NameError``;
* ``cli.consistency._resolve_consistency_model`` passes a ``tensorrt:`` spec
  through verbatim (a prefixed spec is a backend selector, not a path);
* ``src.consistency.trt_forward_capped`` caps the consistency forward at the
  engine's per-forward bound (stub-engine tested, no TRT required);
* ``ModelWrapper`` detects the ``tensorrt:`` prefix lazily — no engine is
  deserialized at construction, so importing / constructing without TRT
  installed (CI, the test suite) does not fail; only an actual ``forward``
  call resolves the engine (and may raise the ``ImportError``).

Heavy: imports ``src.consistency`` (-> ultralytics), like
``test_openvino_optional_imports`` / ``test_ort_cpp_wrapper``. No
``.pt`` / ``.onnx`` / ``.engine`` / GPU required.
"""
import numpy as np
import pytest
import torch

import src
from cli import tensorrt as cli_tensorrt
from cli import (
    TensorRTBuildConfig,
    TensorRTExportConfig,
    TensorRTRunConfig,
)
from cli.consistency import _resolve_consistency_model
from src.consistency import ModelWrapper, trt_forward_capped


# ---------------------------------------------------------------------------
# _LAZY registration — the names must resolve through the package surface.
# ---------------------------------------------------------------------------
LAZY_NAMES = [
    "TensorRTEngine",
    "tensorrt_available",
    "build_tensorrt_engine",
    "export_tensorrt_ultralytics",
    "trt_build_available",
]


def test_public_names_resolve_through_lazy():
    """Each name is reachable via ``getattr(src, name)`` (PEP 562 ``_LAZY``)."""
    for name in LAZY_NAMES:
        assert hasattr(src, name), (
            f"{name!r} is not resolvable via src — missing from _LAZY?"
        )
        assert name in src._LAZY, f"{name!r} missing from src._LAZY map"


def test_cli_tensorrt_exposes_parser_and_run():
    """The subcommand module imports without error and is CLI-shaped."""
    assert callable(getattr(cli_tensorrt, "add_parser", None))
    assert callable(getattr(cli_tensorrt, "run", None))


# ---------------------------------------------------------------------------
# Dataclasses construct with the fields the CLI populates.
# ---------------------------------------------------------------------------
def test_tensorrt_build_config_constructs():
    cfg = TensorRTBuildConfig(
        model="a.onnx", output="a.engine", imgsz=640, precision="fp16",
        max_batch=8, workspace_bytes=8 * 1024 ** 3, device=0,
    )
    assert cfg.precision == "fp16"


def test_tensorrt_export_config_constructs():
    cfg = TensorRTExportConfig(
        model="a.pt", output="a.engine", imgsz=640, precision="fp16",
        device=0,
    )
    assert cfg.data_yaml is None


def test_tensorrt_run_config_constructs():
    cfg = TensorRTRunConfig(
        model="a.engine", imgs_input="data", imgsz=640, device=0,
        max_imgs=32, batch_size=8, conf=0.25, iou=0.45,
        output_dir="results/predictions/tensorrt",
    )
    assert cfg.device == 0


def test_cli_default_engine_output_is_precision_aware(tmp_path):
    """``build`` / ``export`` share one precision-aware default .engine name:
    INT8 lands at yolov8s_int8.engine; fp16 and fp32 keep the _fp16 name.
    Both subparsers carry --output default=None and resolve at run() time."""
    import argparse
    from pathlib import Path
    from cli import (
        DEFAULT_MODEL_TENSORRT_FP16,
        DEFAULT_MODEL_TENSORRT_INT8,
    )
    from cli.tensorrt import _precision_default_output

    assert _precision_default_output("int8") == Path(DEFAULT_MODEL_TENSORRT_INT8)
    assert _precision_default_output("fp16") == Path(DEFAULT_MODEL_TENSORRT_FP16)
    assert _precision_default_output("fp32") == Path(DEFAULT_MODEL_TENSORRT_FP16)

    onnx = tmp_path / "a.onnx"
    onnx.write_bytes(b"")  # --model is existing_validation-gated at parse time
    pt = tmp_path / "a.pt"
    pt.write_bytes(b"")
    parser = cli_tensorrt.add_parser(argparse.ArgumentParser().add_subparsers())
    args_build = parser.parse_args(["build", "--model", str(onnx)])
    args_export = parser.parse_args(["export", "--model", str(pt)])
    assert args_build.output is None
    assert args_export.output is None


def test_tensorrt_build_config_exclude_head_default_on():
    # Head FP32 protection defaults ON for INT8; --no-exclude-head is the
    # opt-out.
    cfg = TensorRTBuildConfig(
        model="a.onnx", output="a.engine", imgsz=640, precision="int8",
        max_batch=8, workspace_bytes=8 * 1024 ** 3, device=0,
    )
    assert cfg.exclude_head is True
    cfg2 = TensorRTBuildConfig(
        model="a.onnx", output="a.engine", imgsz=640, precision="int8",
        max_batch=8, workspace_bytes=8 * 1024 ** 3, device=0,
        exclude_head=False,
    )
    assert cfg2.exclude_head is False


def test_select_head_layer_names_picks_model_22_prefix():
    """The head-exclusion policy is a pure string-prefix match — unit-testable
    without TRT/GPU."""
    from src.tensorrt_build import _select_head_layer_names, HEAD_NAME_PREFIXES
    names = [
        "/model.21/cv2/conv/Conv",                 # backbone — skip
        "/model.22/cv2.0/cv2.0.0/conv/Conv",       # head box-branch conv — pin
        "/model.22/dfl/Reshape",                   # head DFL — pin
        "/model.22/Sigmoid",                      # head activation — pin
        "/model.22.0/Const",                       # NOT head (prefix segment differs)
    ]
    assert _select_head_layer_names(names) == {
        "/model.22/cv2.0/cv2.0.0/conv/Conv",
        "/model.22/dfl/Reshape",
        "/model.22/Sigmoid",
    }
    assert HEAD_NAME_PREFIXES == ["/model.22/"]


# ---------------------------------------------------------------------------
# QDQ-vs-plain ONNX detection — the testable seam for the default-QDQ-vs-
# calibrator branch in build_tensorrt_engine. Pure (onnx only, no TRT).
# The default INT8 path builds a QDQ ONNX (via src.quantize.quantize_onnx_to_int8,
# head excluded) and feeds it to TRT with the INT8 flag + NO calibrator (head
# FP32 by QDQ omission); a plain ONNX takes the --calibrator ablation. This test
# is model-free (synthetic ONNX via onnx.helper) — no .pt / GPU required.
# ---------------------------------------------------------------------------
def _make_plain_onnx(path):
    import onnx
    from onnx import TensorProto, helper
    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", 1])
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["n", 1])
    add = helper.make_node("Add", ["x", "x"], ["y"])
    graph = helper.make_graph([add], "g", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, str(path))


def _make_qdq_onnx(path):
    import onnx
    from onnx import TensorProto, helper
    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", 1])
    D = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["n", 1])
    scale = helper.make_tensor("scale", TensorProto.FLOAT, [1], [1.0])
    zp = helper.make_tensor("zp", TensorProto.UINT8, [1], [0])
    ql = helper.make_node("QuantizeLinear", ["x", "scale", "zp"], ["q"])
    dql = helper.make_node("DequantizeLinear", ["q", "scale", "zp"], ["y"])
    graph = helper.make_graph(
        [ql, dql], "g", [X], [D], initializer=[scale, zp]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, str(path))


def test_is_qdq_onnx_detects_quantize_linear(tmp_path):
    """_is_qdq_onnx is True iff the graph carries QuantizeLinear/DequantizeLinear
    nodes — the pure testable half of the QDQ-vs-calibrator branch."""
    from src.tensorrt_build import _is_qdq_onnx
    plain = tmp_path / "plain.onnx"
    qdq = tmp_path / "qdq.onnx"
    _make_plain_onnx(plain)
    _make_qdq_onnx(qdq)
    assert _is_qdq_onnx(plain) is False
    assert _is_qdq_onnx(qdq) is True


def test_input_batch_build_error_gates_static_mismatches():
    """The ONNX input batch dim / --static mismatches each get a pointed
    message at the build gate; the buildable combinations get None."""
    from src.tensorrt_build import _input_batch_build_error

    # Buildable: dynamic dim either way.
    assert _input_batch_build_error(-1, static=False) is None
    # Buildable: static batch=1 ONNX under --static.
    assert _input_batch_build_error(1, static=True) is None

    # --static with a dynamic dim names the re-export route.
    msg = _input_batch_build_error(-1, static=True)
    assert msg is not None and "--no-dynamic" in msg and ".pt" in msg
    # --static with a non-1 static dim: same gate.
    assert _input_batch_build_error(4, static=True) is not None

    # Static dim without --static: the dynamic profile cannot be set on it.
    msg = _input_batch_build_error(1, static=False)
    assert msg is not None and "--static" in msg and ".pt" in msg
    # Non-1 static dim without --static: neither route fits this ONNX as-is.
    msg = _input_batch_build_error(4, static=False)
    assert msg is not None and "dynamic batch" in msg


def test_tensorrt_build_config_calibrator_default_false():
    # The default INT8 path is QDQ (no calibrator); --calibrator opts into
    # the calibrator-based INT8 path.
    cfg = TensorRTBuildConfig(
        model="a.onnx", output="a.engine", imgsz=640, precision="int8",
        max_batch=8, workspace_bytes=8 * 1024 ** 3, device=0,
    )
    assert cfg.calibrator is False
    cfg2 = TensorRTBuildConfig(
        model="a.onnx", output="a.engine", imgsz=640, precision="int8",
        max_batch=8, workspace_bytes=8 * 1024 ** 3, device=0,
        calibrator=True,
    )
    assert cfg2.calibrator is True


def test_tensorrt_build_config_calib_method_default():
    # The QDQ quantization step defaults to MinMax (matches the ORT path).
    # --calib-method Entropy opts into KL calibration (may fit tight activation
    # distributions better under forced symmetric quantization; Entropy holds
    # all intermediate tensors in host memory, so --max-cal-samples must be
    # reduced when using it).
    cfg = TensorRTBuildConfig(
        model="a.onnx", output="a.engine", imgsz=640, precision="int8",
        max_batch=8, workspace_bytes=8 * 1024 ** 3, device=0,
    )
    assert cfg.calib_method == "MinMax"
    cfg2 = TensorRTBuildConfig(
        model="a.onnx", output="a.engine", imgsz=640, precision="int8",
        max_batch=8, workspace_bytes=8 * 1024 ** 3, device=0,
        calib_method="Entropy",
    )
    assert cfg2.calib_method == "Entropy"


# ---------------------------------------------------------------------------
# Optional-dependency contract: absent tensorrt -> ImportError, not crash.
# Skipped (not failed) when the package IS installed, so the suite stays
# green on a box that happens to have tensorrt.
# ---------------------------------------------------------------------------
TRT_INSTALLED = src.tensorrt_available()


@pytest.mark.skipif(TRT_INSTALLED, reason="tensorrt is installed; "
                    "ImportError path cannot be exercised")
def test_engine_raises_importerror_without_tensorrt():
    TensorRTEngine = src.TensorRTEngine
    with pytest.raises(ImportError, match="tensorrt"):
        TensorRTEngine(model_path="missing.engine")


@pytest.mark.skipif(TRT_INSTALLED, reason="tensorrt is installed")
def test_build_raises_importerror_without_tensorrt():
    with pytest.raises(ImportError, match="tensorrt"):
        src.build_tensorrt_engine(
            onnx_path="missing.onnx", output_path="out.engine",
        )


@pytest.mark.skipif(TRT_INSTALLED, reason="tensorrt is installed")
def test_export_raises_importerror_without_tensorrt():
    with pytest.raises(ImportError, match="tensorrt"):
        src.export_tensorrt_ultralytics(
            pt_path="missing.pt", engine_path="out.engine",
        )


# ---------------------------------------------------------------------------
# Consistency prefix passthrough + lazy ModelWrapper detection.
# ---------------------------------------------------------------------------
def test_resolve_consistency_model_passthrough_tensorrt():
    spec = "tensorrt:models/yolov8s_fp16.engine"
    # The 'tensorrt:' prefix is a backend selector, not a path — the
    # resolver passes it through verbatim so ModelWrapper can parse it.
    assert _resolve_consistency_model(spec, "models/yolov8s.pt") == spec


def test_model_wrapper_tensorrt_prefix_is_lazy():
    w = ModelWrapper("tensorrt:models/yolov8s_fp16.engine", "cuda")
    assert w.type == "tensorrt"
    assert w.model_path == "models/yolov8s_fp16.engine"
    # No engine is deserialized at construction, so this works without TRT
    # installed or the .engine staged (the lazy-load contract).
    assert w.model is None


def test_model_wrapper_tensorrt_device_cuda_coercion_off():
    # device='cuda' coerces to 'cpu' when CUDA is unavailable (the common CI /
    # laptop case) and stays 'cuda' on a CUDA box — the TRT path only matters
    # on a real GPU.
    w = ModelWrapper("tensorrt:models/yolov8s_fp16.engine", "cuda")
    if torch.cuda.is_available():
        assert w.device == "cuda"
    else:
        assert w.device == "cpu"


def test_model_wrapper_tensorrt_numeric_device_selects_gpu():
    # A numeric --device (GPU id) selects that GPU for the TensorRTEngine, while
    # self.device (the "cpu"/"cuda" mode the ORT reference + preprocess use)
    # stays coerced — so those paths are unchanged by a numeric --device. The
    # TRT engine is lazy, so this is model-free (no engine deserialised).
    w_int = ModelWrapper("tensorrt:models/yolov8s_fp16.engine", 1)
    assert w_int.type == "tensorrt"
    assert w_int._trt_gpu_id == 1                       # int GPU id honoured
    assert w_int.device in ("cpu", "cuda")             # ORT-ref mode unchanged
    w_str = ModelWrapper("tensorrt:models/yolov8s_fp16.engine", "1")
    assert w_str._trt_gpu_id == 1                      # numeric string honoured
    # "cpu"/"cuda" -> TRT GPU 0 (the engine is always GPU; default).
    assert ModelWrapper("tensorrt:m.engine", "cpu")._trt_gpu_id == 0
    assert ModelWrapper("tensorrt:m.engine", "cuda")._trt_gpu_id == 0


class _StubEngine:
    """Engine stand-in for :func:`trt_forward_capped` — records chunk sizes,
    tags each chunk's output with its batch count."""

    max_batch = 8

    def __init__(self):
        self.batches = []

    def _effective_batch(self, requested):
        return max(1, min(requested, self.max_batch))

    def raw_forward(self, x):
        self.batches.append(x.shape[0])
        return np.full((x.shape[0], 16, 5), float(x.shape[0]), dtype=np.float32)


def test_trt_forward_capped_subloops_above_max_batch():
    # A batch above the engine's max_batch raises inside _forward; the capped
    # path sub-loops at the effective bound and stacks the chunks in input
    # order (the harness's per-batch except would swallow the raise, so the
    # cap is what keeps every forward within the bound; the tail partial
    # batch runs directly).
    engine = _StubEngine()
    x = np.zeros((20, 3, 4, 4), dtype=np.float32)
    out = trt_forward_capped(engine, x)
    assert engine.batches == [8, 8, 4]
    assert out.shape == (20, 16, 5)
    assert (out[:16] == 8).all() and (out[16:] == 4).all()


def test_trt_forward_capped_single_pass_within_max_batch():
    # A batch within the bound goes through as one forward (no chunking).
    engine = _StubEngine()
    out = trt_forward_capped(engine, np.zeros((3, 3, 4, 4), dtype=np.float32))
    assert engine.batches == [3]
    assert out.shape == (3, 16, 5)


def test_trt_forward_capped_copies_each_chunk_out_of_staging():
    # raw_forward returns a view of the engine's reused host output buffer;
    # the capped path copies each chunk, so a later forward cannot overwrite
    # an earlier chunk's contribution to the stacked result.
    class _StagingEngine(_StubEngine):
        def __init__(self):
            super().__init__()
            self._stage = np.zeros((self.max_batch, 16, 5), dtype=np.float32)

        def raw_forward(self, x):
            self.batches.append(x.shape[0])
            self._stage[: x.shape[0]] = float(x.shape[0])
            return self._stage[: x.shape[0]]

    engine = _StagingEngine()
    x = np.zeros((20, 3, 4, 4), dtype=np.float32)
    out = trt_forward_capped(engine, x)
    assert engine.batches == [8, 8, 4]
    assert out.shape == (20, 16, 5)
    assert (out[:16] == 8).all() and (out[16:] == 4).all()
