"""YOLOv8s ONNX Runtime toolchain — domain layer.

Imports are kept **lazy** on purpose: each submodule (``sampler``, ``benchmark``, ``engine`` ...)
pulls in heavy optional deps (torchvision, pandas, psutil, ORT, torch). Eagerly importing them in
``__init__`` would force every ``from src.X import Y`` — including the pure-logic unit tests —
to load the full dependency graph. Instead we use PEP 562 ``__getattr__`` so submodules are only
loaded when something actually asks for the attribute.
"""
from __future__ import annotations

import importlib
from typing import Any

from ._version import __version__  # noqa: F401  (re-exported for src.__version__)


# Map of public name -> submodule (relative to this package). When a caller does ``from
# src import YOLOv8Engine`` or ``src.YOLOv8Engine``, ``__getattr__`` looks the name up
# here, imports the submodule, and returns the attribute. The resolved attribute is
# also bound to the package (``setattr`` below), so subsequent ``src.YOLOv8Engine``
# lookups bypass ``__getattr__`` entirely.

_LAZY: dict[str, str] = {
    "post_process": ".postprocess",
    "preprocess_single": ".preprocess",
    "preprocess_imgs": ".preprocess",
    "preprocess_frames": ".preprocess",
    "CalibrationSampler": ".sampler",
    "model_export": ".export",
    "validate_consistency": ".consistency",
    "quantize_onnx_to_int8": ".quantize",
    "YOLOv8Engine": ".engine",
    "Benchmark": ".benchmark",
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        mod = importlib.import_module(_LAZY[name], __package__)
        # Bind the submodule to this package so repeated attribute access and
        # ``from . import X`` after a real import resolve without re-dispatch.

        setattr(__import__(__name__), name, getattr(mod, name))
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
