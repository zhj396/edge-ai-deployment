"""Unit tests for the ``src.consistency`` session/forward helpers (CPU path).

Builds a minimal single-node ONNX model in memory and drives it through
``get_ort_session`` + ``ort_forward``, so the torch -> numpy input conversion
and the session wiring stay covered without a model artifact or GPU. The
CUDA DLPack branch and its cross-stream barrier require a GPU; the default
suite covers the CPU path only.
"""
import numpy as np
import onnx
import torch

from src.consistency import get_ort_session, ort_forward


def _write_plus_one_model(path):
    """Minimal ONNX graph ``Y = X + 1`` (1-D float input, dynamic length)."""
    x = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [None])
    y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [None])
    one = onnx.helper.make_tensor("one", onnx.TensorProto.FLOAT, [1], [1.0])
    node = onnx.helper.make_node("Add", ["X", "one"], ["Y"])
    graph = onnx.helper.make_graph([node], "plus_one", [x], [y], [one])
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def test_ort_forward_cpu_path(tmp_path):
    model_path = tmp_path / "plus_one.onnx"
    _write_plus_one_model(model_path)

    session = get_ort_session(str(model_path), "cpu")

    # float64 in, float32 out: the CPU branch casts before session.run.
    x = torch.tensor([1.0, 2.5, 4.0], dtype=torch.float64)
    out = ort_forward(session, x, "cpu")

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [2.0, 3.5, 5.0])
