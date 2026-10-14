"""CLI package surface.

Path flags are validated at parse time via argparse ``type=`` callables
(``existing_validation``, and ``onnx_only`` for ONNX-only inputs), so a
missing artifact fails during parsing with the ARTIFACTS.md hint.
"""
from __future__ import annotations

from src import __version__  # noqa: F401  (re-exported for cli.__version__)

from .schema import ExportConfig, QuantizeConfig

# ---------------------------------------------------------------------------
# Documented default model paths. Centralized so a rename only needs to be
# updated here + README §Installation + ARTIFACTS.md (which pins the artifact
# name).
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PT = "models/yolov8s.pt"
DEFAULT_MODEL_FP32 = "models/yolov8s_fp32.onnx"
DEFAULT_MODEL_INT8 = "models/yolov8s_int8.onnx"

# Calibration / validation image directory (must contain data.yaml + images/val).
DEFAULT_DATA_DIR = "data"


__all__ = [
    "ExportConfig",
    "QuantizeConfig",
    "DEFAULT_MODEL_PT",
    "DEFAULT_MODEL_FP32",
    "DEFAULT_MODEL_INT8",
    "DEFAULT_DATA_DIR",
]
