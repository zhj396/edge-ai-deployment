"""CLI package surface.

Path flags are validated at parse time via argparse ``type=`` callables
(``existing_validation``, and ``onnx_only`` for ONNX-only inputs), so a
missing artifact fails during parsing with the ARTIFACTS.md hint. Optional
path flags (``consistency``) declare ``default=None`` and apply the same
validation to their documented default at ``run()`` time, after logging is
configured.
"""
from __future__ import annotations

from datetime import datetime

from src import __version__  # noqa: F401  (re-exported for cli.__version__)

from .schema import (
    ExportConfig,
    QuantizeConfig,
    InferConfig,
    ConsistencyConfig,
    BenchmarkConfig,
)

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


def run_timestamp() -> str:
    """Per-run artifact timestamp suffix ``_%Y%m%d_%H%M%S_<ms>``.

    Local time, sortable, Windows-safe (no colons); the millisecond
    distinguishes runs started within the same second.
    """
    now = datetime.now()
    return now.strftime("_%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"


__all__ = [
    "ExportConfig",
    "QuantizeConfig",
    "InferConfig",
    "ConsistencyConfig",
    "BenchmarkConfig",
    "DEFAULT_MODEL_PT",
    "DEFAULT_MODEL_FP32",
    "DEFAULT_MODEL_INT8",
    "DEFAULT_DATA_DIR",
    "run_timestamp",
]
