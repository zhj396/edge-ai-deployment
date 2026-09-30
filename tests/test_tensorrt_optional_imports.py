"""Model-free tests for the TensorRT backend's optional-dependency surface.

tensorrt + cuda-python are *optional* — the base pin set (requirements-cpu.txt
/ -gpu.txt) does not include them. These tests guard the contract the rest of
the toolchain relies on without needing a GPU or the TensorRT wheels:

* the new public names resolve through ``src``'s PEP-562 ``_LAZY`` map (the
  original OpenVINO wiring bug: unregistered names raised ``AttributeError``
  at import time);
* the ``cli.tensorrt`` subcommand module imports cleanly and exposes
  ``add_parser`` / ``run`` (it does ``from src import ...`` at module top,
  so a broken ``_LAZY`` entry would surface here);
* the ``TensorRT*Config`` dataclasses construct;
* when tensorrt is absent, constructing ``TensorRTEngine`` and calling the
  build / export entry points raise ``ImportError`` with a helpful message
  rather than ``AttributeError`` / ``NameError``;
* ``cli.consistency._resolve_consistency_model`` passes a ``tensorrt:`` spec
  through verbatim (does not mangle it via ``resolve_path_arg``);
* ``ModelWrapper`` detects the ``tensorrt:`` prefix **lazily** — no engine is
  deserialized at construction, so importing / constructing without TRT
  installed (CI, the test suite) does not blow up; only an actual ``forward``
  call resolves the engine (and may raise the helpful error).

Heavy: imports ``src.consistency`` + ``src.benchmark`` (-> ultralytics), like
``test_openvino_optional_imports`` / ``test_ort_cpp_wrapper``. No
``.pt`` / ``.onnx`` / ``.engine`` / GPU required.
"""
import pytest

import src
from cli import tensorrt as cli_tensorrt
from cli import (
    TensorRTBuildConfig,
    TensorRTExportConfig,
    TensorRTRunConfig,
)
from cli.consistency import _resolve_consistency_model
from src.consistency import ModelWrapper


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


def test_tensorrt_build_config_exclude_head_default_on():
    # Head FP32 protection defaults ON for INT8 (mirrors ORT invariant #3);
    # the CLI --no-exclude-head is the opt-out (ablation).
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
    """The head-exclusion policy is a pure string-prefix match — the testable
    seam (mirrors src.quantize._resolve_node_names_by_prefix). No TRT/GPU."""
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
# legacy-calibrator branch in build_tensorrt_engine. Pure (onnx only, no TRT).
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


def test_tensorrt_build_config_calibrator_default_false():
    # The default INT8 path is QDQ (no calibrator); --calibrator opts into the
    # legacy ablation that does NOT hold the head FP32 on T4/TRT 10.4.
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
    # --calib-method Entropy opts into the KL-calibration experiment (the lever
    # for the ~18-33% recall loss; Entropy OOMs on the 300-sample set -- the
    # calibration forward device is cpu, host-side histogram cache is the
    # bottleneck; cut --max-cal-samples).
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
    # The 'tensorrt:' prefix is a backend selector, not a path — it must
    # survive resolve_path_arg verbatim so ModelWrapper can parse it.
    assert _resolve_consistency_model(spec, "models/yolov8s.pt") == spec


def test_model_wrapper_tensorrt_prefix_is_lazy():
    w = ModelWrapper("tensorrt:models/yolov8s_fp16.engine", "cuda")
    assert w.type == "tensorrt"
    assert w.model_path == "models/yolov8s_fp16.engine"
    # The whole point of the lazy design: no engine is deserialized at
    # construction — so this works without TRT installed or the .engine staged.
    assert w.model is None


def test_model_wrapper_tensorrt_device_cuda_coercion_off():
    # device='cuda' coerces to 'cpu' when CUDA is unavailable (the common CI /
    # laptop case) — the TRT path only matters on a real GPU.
    w = ModelWrapper("tensorrt:models/yolov8s_fp16.engine", "cuda")
    assert w.device in ("cpu", "cuda")


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
