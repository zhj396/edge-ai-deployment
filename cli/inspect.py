"""CLI: ``inspect`` — print ONNX model metadata as JSON."""
import json
from pathlib import Path
from typing import Union

from utils import (
    PathValidationError,
    compare_models,
    existing_validation,
    get_logger,
    inspect_onnx,
)

logger = get_logger(__name__)


def onnx_only(path: Union[str, Path]) -> Path:
    """Validate that ``path`` is an existing .onnx file.

    Raises ``PathValidationError`` (an ``argparse.ArgumentTypeError``) with a
    suffix-specific message, so argparse ``type=`` callables print it verbatim.
    """
    p = existing_validation(path)
    if p.suffix.lower() != ".onnx":
        suffix = p.suffix or "no suffix"
        raise PathValidationError(
            f"--model/--compare expects a .onnx file, got {suffix!r} ({p}). "
            f"Use Ultralytics' YOLO export for a PyTorch checkpoint first."
        )
    return p


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "inspect",
        help="Inspect an ONNX model's metadata",
        epilog="Example: python main.py inspect --model models/yolov8s_fp32.onnx",
    )
    parser.add_argument("--model", type=onnx_only, required=True)
    parser.add_argument(
        "--compare", type=onnx_only, default=None,
        help="Optional second ONNX model for side-by-side comparison",
    )
    return parser


def run(args):
    meta = inspect_onnx(args.model)
    logger.info("Model metadata:\n%s", json.dumps(meta.to_dict(), indent=2))

    if args.compare:
        comp = compare_models(args.model, args.compare)
        logger.info("Comparison:\n%s", json.dumps(comp, indent=2))
