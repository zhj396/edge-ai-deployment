"""YOLOv8s ONNX Runtime toolchain — domain layer.

Imports are kept **lazy** on purpose: submodules pull in heavy optional deps
(torch, torchvision, onnx, onnxruntime). PEP 562 ``__getattr__`` loads a
submodule only when something actually asks for the attribute, so ``from
src import X`` — including the pure-logic unit tests — does not pull in the
full dependency graph.

New domain submodules (``sampler``, ``benchmark``, ``engine`` ...) register
here by adding a ``public_name -> ".submodule"`` entry to ``_LAZY`` as they
land; ``__all__`` follows automatically.
"""
from __future__ import annotations

import importlib
from typing import Any

from ._version import __version__  # noqa: F401  (re-exported for src.__version__)


# Map of public name -> submodule (relative to this package), only for
# submodules that exist. When a caller does ``from src import model_export``
# or ``src.model_export``, ``__getattr__`` looks the name up here, imports the
# submodule, and returns the attribute. The resolved attribute is also bound
# to the package (``globals()`` below), so subsequent ``src.model_export``
# lookups bypass ``__getattr__`` entirely.

_LAZY: dict[str, str] = {
    "model_export": ".export",
    "preprocess_single": ".preprocess",
    "preprocess_imgs": ".preprocess",
    "preprocess_frames": ".preprocess",
    "post_process": ".postprocess",
    "CalibrationSampler": ".sampler",
    "quantize_onnx_to_int8": ".quantize",
    "YOLOv8Engine": ".engine",
    "validate_consistency": ".consistency",
    "Benchmark": ".benchmark",
    # OpenVINO backend (optional — openvino/nncf not in the base pin set)
    "convert_onnx_to_openvino_ir": ".openvino_convert",
    "nncf_quantize_openvino": ".openvino_convert",
    "openvino_conversion_available": ".openvino_convert",
    "nncf_available": ".openvino_convert",
    "OpenVINOEngine": ".openvino_engine",
    "OpenVINOAsyncEngine": ".openvino_engine",
    "openvino_available": ".openvino_engine",
    # TensorRT backend (optional — tensorrt/cuda-python not in the base pin set)
    "TensorRTEngine": ".tensorrt_engine",
    "tensorrt_available": ".tensorrt_engine",
    "build_tensorrt_engine": ".tensorrt_build",
    "export_tensorrt_ultralytics": ".tensorrt_build",
    "trt_build_available": ".tensorrt_build",
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        mod = importlib.import_module(_LAZY[name], __package__)
        attr = getattr(mod, name)
        # Bind to this package so repeated attribute access and
        # ``from . import X`` after a real import resolve without re-dispatch.
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
