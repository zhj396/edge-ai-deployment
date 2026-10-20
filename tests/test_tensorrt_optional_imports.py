"""Model-free tests for the TensorRT backend's optional-dependency surface.

tensorrt + cuda-python are *optional* — the base pin set (requirements-cpu.txt
/ requirements-kaggle.txt) does not include them. These tests guard the contract
the rest of the toolchain relies on without needing a GPU or the TensorRT
wheels:

* the public names resolve through ``src``'s PEP-562 ``_LAZY`` map (an
  unregistered name breaks the import with ``ImportError``);
* when tensorrt is absent, constructing ``TensorRTEngine`` and calling the
  build / export entry points raise ``ImportError`` with a helpful message,
  never ``AttributeError`` / ``NameError``;
* the build-side pure seams (head-exclusion name match, QDQ detection, the
  ``--static`` batch-dim gate) are policy unit tests on plain strings /
  ``onnx`` graphs — no TRT / GPU.

No ``.pt`` / ``.onnx`` / ``.engine`` / GPU required.
"""
import pytest

import src


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
