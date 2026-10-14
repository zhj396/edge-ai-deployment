"""CLI package surface.

Optional path flags are declared ``default=None`` in argparse and resolved
at ``run()`` time via :func:`resolve_path_arg`, which applies
``existing_validation`` to the documented default after logging is
configured.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from src import __version__  # noqa: F401  (re-exported for cli.__version__)
from utils import existing_validation

from .schema import (
    ExportConfig,
    QuantizeConfig,
    InferConfig,
    BenchmarkConfig,
    ConsistencyConfig,
)

# ---------------------------------------------------------------------------
# Documented default model paths. Centralized so a future rename (e.g. yolov8s_v2.pt)
# only needs to be updated here + README §Installation + ARTIFACTS.md (which pins
# the artifact name).
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PT = "models/yolov8s.pt"
DEFAULT_MODEL_FP32 = "models/yolov8s_fp32.onnx"
DEFAULT_MODEL_INT8 = "models/yolov8s_int8.onnx"

# Calibration / validation image directory (must contain data.yaml + images/val).
DEFAULT_DATA_DIR = "data"


def resolve_path_arg(value: Optional[Path], default: str) -> Path:
    """Return the validated ``Path`` for a path flag.

    Returns ``value`` if the user supplied one; otherwise validates the
    documented default.
    """
    return value if value is not None else existing_validation(default)


# Alias of resolve_path_arg for model-path call sites.
resolve_model_arg = resolve_path_arg


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
    "BenchmarkConfig",
    "ConsistencyConfig",
    "DEFAULT_MODEL_PT",
    "DEFAULT_MODEL_FP32",
    "DEFAULT_MODEL_INT8",
    "DEFAULT_DATA_DIR",
    "resolve_path_arg",
    "resolve_model_arg",
    "run_timestamp",
]
