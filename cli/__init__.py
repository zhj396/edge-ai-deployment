"""CLI package surface.

Why this module exposes path defaults + a resolver, not just ``@dataclass`` configs: argparse
applies ``type=existing_validation`` to string defaults at parse time, so a default like
``models/yolov8s_fp32.onnx`` would crash on a fresh clone (no ``models/`` dir) with a cryptic
"invalid ty value" message.

We avoid that by passing ``default=None`` to argparse and resolving the documented default at
``run()`` time via :func:`resolve_path_arg`. The error message then comes from the actual
workflow ("Path does not exist: …" + the ARTIFACTS.md hint) rather than from the parser, and
the same helper is the single source of truth for which path each subcommand treats as its
default — rename ``yolov8s.pt`` to ``yolov8s_v2.pt`` here once, the README is the only other
place to update.
"""
from __future__ import annotations

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
# only needs to be updated here + README §Installation.
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PT = "models/yolov8s.pt"
DEFAULT_MODEL_FP32 = "models/yolov8s_fp32.onnx"
DEFAULT_MODEL_INT8 = "models/yolov8s_int8.onnx"

# Calibration / validation image directory (must contain data.yaml + images/val).
DEFAULT_DATA_DIR = "data"


def resolve_path_arg(value: Optional[Path], default: str) -> Path:
    """Return the validated ``Path`` for a path flag.

    Flow: pass ``default=None`` to argparse so it never calls ``type=`` on the default string. At
    ``run()`` time, return ``value`` if the user supplied one; otherwise validate the documented
    default. This keeps fresh-clone errors ("Path does not exist: …" + the ARTIFACTS.md hint)
    at the same code level as the workflow, instead of slipping out of argparse as a parse
    error.
    """
    return value if value is not None else existing_validation(default)


# Back-compat alias — ``resolve_model_arg`` is what older call sites used.
resolve_model_arg = resolve_path_arg


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
]
