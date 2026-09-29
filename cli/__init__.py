"""CLI package surface.

Why this module exposes path defaults + a resolver, not just ``@dataclass`` configs: argparse
applies ``type=existing_validation`` to string defaults at parse time, so a default like
``models/yolov8s_fp32.onnx`` fails at parse time on a fresh clone (no ``models/`` dir) with a
cryptic "invalid ty value" message.

Instead, ``default=None`` goes to argparse and the documented default is resolved at
``run()`` time via :func:`resolve_path_arg`. The error message then comes from the actual
workflow ("Path does not exist: …" + the ARTIFACTS.md hint), and
the same helper is the single source of truth for which path each subcommand treats as its
default — rename ``yolov8s.pt`` to ``yolov8s_v2.pt`` here once, the README is the only other
place to update.
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
    OpenVINOConvertConfig,
    OpenVINOQuantizeConfig,
    OpenVINORunConfig,
)

# ---------------------------------------------------------------------------
# Documented default model paths. Centralized so a future rename (e.g. yolov8s_v2.pt)
# only needs to be updated here + README §Installation.
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PT = "models/yolov8s.pt"
DEFAULT_MODEL_FP32 = "models/yolov8s_fp32.onnx"
DEFAULT_MODEL_INT8 = "models/yolov8s_int8.onnx"
DEFAULT_MODEL_OPENVINO = "models/yolov8s_openvino.xml"
DEFAULT_MODEL_OPENVINO_INT8 = "models/yolov8s_openvino_int8.xml"

# Calibration / validation image directory (must contain data.yaml + images/val).
DEFAULT_DATA_DIR = "data"


def resolve_path_arg(value: Optional[Path], default: str) -> Path:
    """Return the validated ``Path`` for a path flag.

    Flow: pass ``default=None`` to argparse so it never calls ``type=`` on the default string. At
    ``run()`` time, return ``value`` if the user supplied one; otherwise validate the documented
    default. Fresh-clone errors ("Path does not exist: …" + the ARTIFACTS.md hint) surface at
    the same code level as the workflow.
    """
    return value if value is not None else existing_validation(default)


# Back-compat alias — ``resolve_model_arg`` is what older call sites used.
resolve_model_arg = resolve_path_arg


def run_timestamp() -> str:
    """Per-run artifact timestamp suffix, single source for all subcommands.

    Format: ``_%Y%m%d_%H%M%S_<ms>`` — local time + millisecond, sortable,
    Windows-safe (no colons). One timestamp per CLI run so every artifact a
    run writes (consistency report + failed-sample dir, benchmark summary +
    per-class CSVs) shares the same suffix; the millisecond distinguishes
    runs started within the same second.
    """
    now = datetime.now()
    return now.strftime("_%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"


__all__ = [
    "ExportConfig",
    "QuantizeConfig",
    "InferConfig",
    "BenchmarkConfig",
    "ConsistencyConfig",
    "OpenVINOConvertConfig",
    "OpenVINOQuantizeConfig",
    "OpenVINORunConfig",
    "DEFAULT_MODEL_PT",
    "DEFAULT_MODEL_FP32",
    "DEFAULT_MODEL_INT8",
    "DEFAULT_MODEL_OPENVINO",
    "DEFAULT_MODEL_OPENVINO_INT8",
    "DEFAULT_DATA_DIR",
    "resolve_path_arg",
    "resolve_model_arg",
    "run_timestamp",
]
